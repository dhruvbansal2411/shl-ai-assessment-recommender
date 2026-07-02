"""Conversation orchestration for the SHL recommender."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

import httpx

from app.guardrails import check_user_input, validate_recommendations
from app.models import Assessment, ChatResponse, Message, Recommendation
from app.prompts import build_grounded_prompt
from app.retriever import CatalogRetriever
from app.utils import get_settings

logger = logging.getLogger(__name__)

VAGUE_PATTERNS = [
    r"^i need an assessment\.?$",
    r"^recommend (an|a|some)? ?assessment(s)?\.?$",
    r"^help me choose\.?$",
    r"^need test(s)?\.?$",
]

COMPARISON_RE = re.compile(
    r"compare\s+(?P<left>.+?)\s+(?:and|vs|versus)\s+(?P<right>.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Intent:
    """Small intent result used to keep chat flow explicit."""

    name: str
    needs_clarification: bool = False


class ChatService:
    """Stateless chat service backed by a catalog retriever."""

    def __init__(self, retriever: CatalogRetriever | None = None) -> None:
        self.settings = get_settings()
        self.retriever = retriever or CatalogRetriever()

    async def handle(self, messages: list[Message]) -> ChatResponse:
        """Handle one stateless chat request."""

        latest_user = _latest_user_message(messages)
        guardrail = check_user_input(latest_user)
        if not guardrail.allowed:
            return ChatResponse(
                reply=guardrail.reply or "I can't help with that request.",
                recommendations=[],
                end_of_conversation=False,
            )

        intent = classify_intent(messages)
        if intent.needs_clarification:
            return ChatResponse(
                reply=(
                    "I can help with that. What role are you hiring for, what "
                    "experience level should the assessment target, and which "
                    "skills or traits matter most?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        query = _conversation_query(messages)
        retrieved = self.retriever.retrieve(query, top_k=self.settings.top_k)
        context = [item.assessment for item in retrieved]

        if intent.name == "comparison":
            return await self._compare(messages, context)

        if not context:
            return ChatResponse(
                reply="I don't have enough catalog information.",
                recommendations=[],
                end_of_conversation=False,
            )

        recommendations = [
            Recommendation(
                name=item.assessment.name,
                url=item.assessment.url,
                test_type=item.assessment.test_type,
            )
            for item in retrieved[:10]
        ]
        recommendations = validate_recommendations(recommendations, self.retriever.catalog)
        reply = await self._build_reply(messages, context, recommendations)
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    async def _compare(self, messages: list[Message], context: list[Assessment]) -> ChatResponse:
        latest = _latest_user_message(messages)
        pair = _extract_comparison_pair(latest)
        selected = _select_comparison_assessments(pair, self.retriever.catalog)
        if len(selected) < 2:
            selected = _select_comparison_assessments(pair, context)
        if len(selected) < 2:
            return ChatResponse(
                reply="I don't have enough catalog information.",
                recommendations=[],
                end_of_conversation=False,
            )

        left, right = selected[:2]
        reply = (
            f"{_comparison_display_name(left.name)} is a {left.test_type} assessment measuring "
            f"{', '.join(left.skills_measured[:5])}. It is listed at {left.duration}, "
            f"supports remote testing: {left.remote_testing_support}, and adaptive "
            f"support: {left.adaptive_support}.\n\n"
            f"{_comparison_display_name(right.name)} is a {right.test_type} assessment measuring "
            f"{', '.join(right.skills_measured[:5])}. It is listed at {right.duration}, "
            f"supports remote testing: {right.remote_testing_support}, and adaptive "
            f"support: {right.adaptive_support}.\n\n"
            "Choose the assessment whose measured skills and job level match the "
            "role more closely."
        )
        recommendations = [
            Recommendation(name=item.name, url=item.url, test_type=item.test_type)
            for item in selected
        ]
        return ChatResponse(
            reply=reply,
            recommendations=validate_recommendations(recommendations, self.retriever.catalog),
            end_of_conversation=False,
        )

    async def _build_reply(
        self,
        messages: list[Message],
        context: list[Assessment],
        recommendations: list[Recommendation],
    ) -> str:
        if not recommendations:
            return "I don't have enough catalog information."

        llm_reply = await self._call_llm(messages, context)
        if llm_reply:
            return llm_reply

        names = ", ".join(item.name for item in recommendations[:5])
        if len(recommendations) == 1:
            return (
                f"The strongest catalog match is {recommendations[0].name}. "
                "It fits the requirements based on the retrieved SHL catalog information."
            )
        return (
            f"I found {len(recommendations)} SHL catalog matches. The strongest options are "
            f"{names}. I used only the retrieved Individual Test Solution records."
        )

    async def _call_llm(self, messages: list[Message], context: list[Assessment]) -> str | None:
        if not self.settings.enable_llm:
            return None

        prompt = build_grounded_prompt(messages, context)
        start = time.perf_counter()
        try:
            if self.settings.llm_provider.lower() == "openrouter":
                if not self.settings.openrouter_api_key:
                    return None
                reply = await _call_openrouter(prompt)
            else:
                if not self.settings.gemini_api_key:
                    return None
                reply = await _call_gemini(prompt)
            logger.info("llm latency_seconds=%.3f", time.perf_counter() - start)
            return reply
        except (httpx.TimeoutException, httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            logger.error("llm error=%s latency_seconds=%.3f", exc, time.perf_counter() - start)
            return None


async def _call_gemini(prompt: str) -> str:
    settings = get_settings()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(
            url,
            params={"key": settings.gemini_api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def _call_openrouter(prompt: str) -> str:
    settings = get_settings()
    payload = {
        "model": settings.openrouter_model,
        "messages": [
            {"role": "system", "content": "You answer using only supplied SHL catalog context."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def classify_intent(messages: list[Message]) -> Intent:
    """Classify the latest user turn with enough precision for routing."""

    latest = _latest_user_message(messages)
    normalized = latest.lower().strip()
    if any(re.match(pattern, normalized) for pattern in VAGUE_PATTERNS):
        return Intent(name="clarification", needs_clarification=True)
    if "compare" in normalized or " vs " in normalized or " versus " in normalized:
        return Intent(name="comparison")
    return Intent(name="recommendation")


def _latest_user_message(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content


def _conversation_query(messages: list[Message]) -> str:
    """Build a retrieval query from the full stateless conversation."""

    return "\n".join(
        message.content for message in messages if message.role in {"user", "assistant"}
    )


def _extract_comparison_pair(text: str) -> tuple[str, str] | None:
    match = COMPARISON_RE.search(text)
    if not match:
        return None
    left = _clean_comparison_name(match.group("left"))
    right = _clean_comparison_name(match.group("right"))
    return left, right


def _clean_comparison_name(value: str) -> str:
    return re.split(r"\s+for\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .?")


def _select_comparison_assessments(
    pair: tuple[str, str] | None, context: list[Assessment]
) -> list[Assessment]:
    if pair is None:
        return context[:2]
    selected: list[Assessment] = []
    for name in pair:
        lowered = _normalized_assessment_name(name)
        exact = next(
            (item for item in context if _normalized_assessment_name(item.name) == lowered),
            None,
        )
        partial = next(
            (item for item in context if lowered in _normalized_assessment_name(item.name)),
            None,
        )
        match = exact or partial
        if match and match not in selected:
            selected.append(match)
    for item in context:
        if len(selected) >= 2:
            break
        if item not in selected:
            selected.append(item)
    return selected[:2]


def _normalized_assessment_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _comparison_display_name(value: str) -> str:
    return re.sub(r"\(([^)]+)\)", r"\1", value).replace("  ", " ").strip()
