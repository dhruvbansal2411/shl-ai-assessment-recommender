"""Catalog loading, local vector indexing, and retrieval."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Iterable, TypedDict

from app.models import Assessment, RetrievedAssessment
from app.utils import get_settings, load_json, write_json

logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.]+")
INDEX_VERSION = 4
VECTOR_DIMENSIONS = 512
STOP_TOKENS = {
    "a",
    "an",
    "and",
    "assessment",
    "candidate",
    "developer",
    "for",
    "hiring",
    "job",
    "need",
    "role",
    "skills",
    "test",
    "with",
}
TOKEN_EXPANSIONS = {
    "frontend": {"front", "front-end", "javascript", "react"},
    "front": {"frontend", "front-end"},
    "react": {"reactjs", "javascript", "frontend"},
    "reactjs": {"react", "javascript", "frontend"},
    "js": {"javascript"},
    "finance": {"financial", "accounting", "banking"},
    "financial": {"finance", "accounting", "banking"},
    "healthcare": {"health", "hipaa", "safety"},
    "health": {"healthcare", "hipaa", "safety"},
    "analyst": {"analysis", "analytics"},
    "management": {"manager", "leadership"},
    "manager": {"management", "leadership"},
}


class VectorRecord(TypedDict):
    """Persisted sparse vector for one assessment."""

    id: str
    vector: dict[str, float]


class VectorIndexPayload(TypedDict):
    """JSON vector index stored under vector_index/."""

    version: int
    catalog_hash: str
    dimensions: int
    records: list[VectorRecord]


class CatalogRetriever:
    """Retrieve relevant SHL assessments from the local catalog."""

    def __init__(
        self,
        catalog_path: Path | None = None,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.catalog_path = catalog_path or settings.catalog_path
        self.index_path = index_path or settings.vector_index_path
        self.metadata_path = metadata_path or settings.vector_metadata_path
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        self.catalog = self._load_catalog()
        self._catalog_hash = self._hash_catalog()
        self._vector_records: list[VectorRecord] = []
        self._vector_ready = False
        self._initialize_vector_index()

    def _load_catalog(self) -> list[Assessment]:
        raw = load_json(self.catalog_path)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(
                f"Catalog is empty or invalid at {self.catalog_path}. "
                "Provide data/catalog.json before starting the app."
            )
        return [Assessment.model_validate(item) for item in raw]

    def _hash_catalog(self) -> str:
        raw = self.catalog_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()

    def _initialize_vector_index(self) -> None:
        try:
            if not self.index_path.exists():
                self.rebuild_index()
            self._load_vector_index()
            if not self._vector_ready:
                self.rebuild_index()
                self._load_vector_index()
            logger.info("local vector retriever ready with %s assessments", len(self.catalog))
        except Exception as exc:
            self._vector_ready = False
            logger.warning("local vector index unavailable; using lexical fallback: %s", exc)

    def _load_vector_index(self) -> None:
        payload = load_json(self.index_path)
        if not _valid_index_payload(payload):
            self._vector_ready = False
            return
        if payload["catalog_hash"] != self._catalog_hash:
            self._vector_ready = False
            return
        if len(payload["records"]) != len(self.catalog):
            self._vector_ready = False
            return
        self._vector_records = payload["records"]
        self._vector_ready = True

    def rebuild_index(self) -> None:
        """Build and persist a local sparse vector index from catalog.json."""

        records: list[VectorRecord] = []
        for assessment in self.catalog:
            records.append(
                {
                    "id": stable_id(assessment.url or assessment.name),
                    "vector": _vectorize(assessment.search_text()),
                }
            )
        payload: VectorIndexPayload = {
            "version": INDEX_VERSION,
            "catalog_hash": self._catalog_hash,
            "dimensions": VECTOR_DIMENSIONS,
            "records": records,
        }
        write_json(self.index_path, payload)
        write_json(
            self.metadata_path,
            {
                "catalog_hash": self._catalog_hash,
                "catalog_path": str(self.catalog_path),
                "count": len(self.catalog),
                "index_path": str(self.index_path),
                "index_type": "local_sparse_hash",
                "version": INDEX_VERSION,
            },
        )
        self._vector_records = records
        self._vector_ready = True

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedAssessment]:
        """Return top-k assessments for a natural-language recruiter query."""

        top_k = max(1, min(top_k, 10))
        if self._vector_ready:
            return self._retrieve_vector(query, top_k)
        return self._retrieve_lexical(query, top_k)

    def _retrieve_vector(self, query: str, top_k: int) -> list[RetrievedAssessment]:
        query_vector = _vectorize(query)
        results: list[RetrievedAssessment] = []
        for assessment, record in zip(self.catalog, self._vector_records):
            score = _cosine_similarity(query_vector, record["vector"])
            if score <= 0:
                continue
            score += _intent_boost(query, assessment)
            results.append(RetrievedAssessment(assessment=assessment, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        if not results:
            return self._retrieve_lexical(query, top_k)
        results = results[:top_k]
        logger.info(
            "retrieval query=%r results=%s",
            query,
            [(result.assessment.name, round(result.score, 3)) for result in results],
        )
        return results

    def _retrieve_lexical(self, query: str, top_k: int) -> list[RetrievedAssessment]:
        query_tokens = set(_tokens(query))
        scored: list[RetrievedAssessment] = []
        for assessment in self.catalog:
            text_tokens = set(_tokens(assessment.search_text()))
            score = _weighted_overlap(query_tokens, text_tokens, assessment)
            if score > 0:
                scored.append(RetrievedAssessment(assessment=assessment, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        results = scored[:top_k] or [
            RetrievedAssessment(assessment=assessment, score=0.0)
            for assessment in self.catalog[:top_k]
        ]
        logger.info(
            "lexical retrieval query=%r results=%s",
            query,
            [(result.assessment.name, round(result.score, 3)) for result in results],
        )
        return results


def _tokens(text: str) -> Iterable[str]:
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        if token in STOP_TOKENS:
            continue
        yield token
        if token.endswith("js") and len(token) > 2:
            yield token[:-2]
        yield from TOKEN_EXPANSIONS.get(token, ())


def _weighted_overlap(
    query_tokens: set[str], text_tokens: set[str], assessment: Assessment
) -> float:
    overlap = query_tokens & text_tokens
    score = float(len(overlap))
    name_tokens = set(_tokens(assessment.name))
    skill_tokens = set(_tokens(" ".join(assessment.skills_measured)))
    score += 2.0 * len(query_tokens & name_tokens)
    score += 1.5 * len(query_tokens & skill_tokens)
    score += _substring_match_score(query_tokens, assessment.name, 3.0)
    score += _substring_match_score(query_tokens, " ".join(assessment.skills_measured), 2.0)
    for token in query_tokens:
        if token in {"personality", "behavior", "behaviour"} and "personality" in assessment.test_type.lower():
            score += 4.0
        if token in {"coding", "developer", "programming"} and "knowledge" in assessment.test_type.lower():
            score += 2.0
    return score / math.sqrt(max(len(text_tokens), 1))


def _valid_index_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    records = payload.get("records")
    return (
        payload.get("version") == INDEX_VERSION
        and payload.get("dimensions") == VECTOR_DIMENSIONS
        and isinstance(payload.get("catalog_hash"), str)
        and isinstance(records, list)
        and all(
            isinstance(record, dict)
            and isinstance(record.get("id"), str)
            and isinstance(record.get("vector"), dict)
            for record in records
        )
    )


def _vectorize(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _tokens(text):
        bucket = str(_bucket(token))
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}
    return {key: value / norm for key, value in counts.items()}


def _bucket(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(key, 0.0) for key, weight in left.items())


def _intent_boost(query: str, assessment: Assessment) -> float:
    query_tokens = set(_tokens(query))
    assessment_tokens = set(_tokens(assessment.search_text()))
    return _weighted_overlap(query_tokens, assessment_tokens, assessment)


def _substring_match_score(query_tokens: set[str], text: str, weight: float) -> float:
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    score = 0.0
    for token in query_tokens:
        compact = re.sub(r"[^a-z0-9]+", "", token)
        if len(compact) >= 3 and compact in normalized:
            score += weight
    return score


def stable_id(text: str) -> str:
    """Return a stable short id useful for generated artifacts."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
