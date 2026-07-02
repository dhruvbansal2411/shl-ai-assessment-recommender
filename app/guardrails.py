"""Input and output guardrails for safe catalog-only recommendations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Assessment, Recommendation

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above|system|developer) instructions",
    r"forget (all )?(previous|above|system|developer) instructions",
    r"reveal (the )?(system prompt|developer message|hidden prompt)",
    r"act as (?:dan|jailbreak|uncensored)",
    r"you are now",
    r"bypass (the )?(rules|guardrails|policy)",
]

REFUSAL_TOPICS = {
    "legal": {"lawyer", "lawsuit", "legal advice", "contract", "sue", "compliance ruling"},
    "medical": {"diagnose", "medical", "doctor", "treatment", "medication", "therapy plan"},
    "finance": {"investment", "stock", "crypto", "loan", "tax advice", "portfolio"},
    "general_hiring": {
        "write a job description",
        "interview questions",
        "salary",
        "compensation",
        "performance review",
        "fire an employee",
        "termination",
    },
}

ALLOWED_DOMAIN_TERMS = {
    "assessment",
    "test",
    "shl",
    "skill",
    "skills",
    "role",
    "developer",
    "engineer",
    "manager",
    "sales",
    "support",
    "graduate",
    "personality",
    "cognitive",
    "java",
    "python",
    "sql",
    "javascript",
    "compare",
    "recommend",
    "candidate",
    "hiring",
    "job",
}


@dataclass(frozen=True)
class GuardrailResult:
    """Decision returned by guardrail checks."""

    allowed: bool
    reason: str | None = None
    reply: str | None = None


def detect_prompt_injection(text: str) -> bool:
    """Return True when a user message attempts to override instructions."""

    normalized = text.lower()
    return any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS)


def check_user_input(text: str) -> GuardrailResult:
    """Validate whether the latest user request belongs in the product scope."""

    normalized = text.lower()
    if detect_prompt_injection(normalized):
        return GuardrailResult(
            allowed=False,
            reason="prompt_injection",
            reply=(
                "I can't follow instructions that try to override my catalog-only "
                "SHL assessment recommender rules."
            ),
        )

    for topic, terms in REFUSAL_TOPICS.items():
        if any(term in normalized for term in terms):
            return GuardrailResult(
                allowed=False,
                reason=topic,
                reply=(
                    "I can only help with SHL Individual Test Solution assessment "
                    "selection. I can't provide general hiring, legal, medical, or "
                    "financial advice."
                ),
            )

    if len(normalized.split()) > 2 and not any(term in normalized for term in ALLOWED_DOMAIN_TERMS):
        return GuardrailResult(
            allowed=False,
            reason="off_topic",
            reply=(
                "I can only help recruiters choose SHL Individual Test Solution "
                "assessments from the catalog."
            ),
        )

    return GuardrailResult(allowed=True)


def validate_recommendations(
    recommendations: list[Recommendation], catalog: list[Assessment]
) -> list[Recommendation]:
    """Drop any recommendation not found exactly in the catalog."""

    by_name = {assessment.name: assessment for assessment in catalog}
    valid: list[Recommendation] = []
    for item in recommendations:
        assessment = by_name.get(item.name)
        if assessment and assessment.url == item.url:
            valid.append(item)
    return valid[:10]

