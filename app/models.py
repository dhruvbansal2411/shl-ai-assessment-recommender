"""Pydantic models for the public API and catalog records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """Single stateless chat message supplied by the caller."""

    role: Role
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be empty")
        return stripped


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    """Stable public recommendation shape."""

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """Response body for POST /chat. Keep this schema stable."""

    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok"]


class Assessment(BaseModel):
    """Normalized SHL Individual Test Solution record."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    skills_measured: list[str] = Field(default_factory=list)
    test_type: str = "Assessment"
    duration: str = "See SHL catalog"
    remote_testing_support: bool = False
    adaptive_support: bool = False
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    url: str

    @model_validator(mode="before")
    @classmethod
    def normalize_catalog_record(cls, value: Any) -> Any:
        """Accept both legacy and current SHL catalog record shapes."""

        if not isinstance(value, dict):
            return value

        data = dict(value)
        keys = _as_list(data.get("keys"))

        if "url" not in data and data.get("link"):
            data["url"] = data["link"]
        if not _as_list(data.get("skills_measured")) and keys:
            data["skills_measured"] = keys
        if not str(data.get("test_type", "")).strip() and keys:
            data["test_type"] = ", ".join(keys)
        if "remote_testing_support" not in data and "remote" in data:
            data["remote_testing_support"] = _as_bool(data["remote"])
        if "adaptive_support" not in data and "adaptive" in data:
            data["adaptive_support"] = _as_bool(data["adaptive"])
        if not data.get("description"):
            data["description"] = data.get("name", "")

        return data

    @field_validator("description", "duration", "test_type", "url", mode="before")
    @classmethod
    def stringify_catalog_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("skills_measured", "job_levels", "languages", mode="before")
    @classmethod
    def normalize_catalog_lists(cls, value: Any) -> list[str]:
        return _as_list(value)

    def search_text(self) -> str:
        """Return dense text optimized for embedding and lexical fallback."""

        parts = [
            self.name,
            self.description,
            " ".join(self.skills_measured),
            self.test_type,
            self.duration,
            " ".join(self.job_levels),
            " ".join(self.languages),
            "remote testing" if self.remote_testing_support else "no remote testing",
            "adaptive" if self.adaptive_support else "non adaptive",
        ]
        return " | ".join(part for part in parts if part)


class RetrievedAssessment(BaseModel):
    """Assessment plus retrieval score used internally."""

    assessment: Assessment
    score: float


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "available", "check"}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).replace(";", ",").replace("|", ",").split(",")
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized
