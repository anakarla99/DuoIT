"""Local LLM orchestration for pronunciation and dramatic coaching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from backend.knowledge.semantic_search import PhoneticSearchResult
from backend.scorer.pronunciation_scoring import PronunciationScoreResult

DEFAULT_COACH_MODEL = "distilgpt2"
DEFAULT_MAX_NEW_TOKENS = 160


class DramaCoachError(ValueError):
    """Raised when the dramatic-coach request or model response is invalid."""


@dataclass(frozen=True)
class DramaCoachRequest:
    """Inputs used to generate a focused dramatic-coaching response."""

    expected_text: str
    score_result: PronunciationScoreResult
    rules: tuple[PhoneticSearchResult, ...] = ()
    acting_direction: str = "Speak clearly and confidently."


@dataclass(frozen=True)
class DramaCoachResponse:
    """Generated coaching text and the prompt used to produce it."""

    feedback: str
    prompt: str
    model_name: str


class TextGenerationPipeline(Protocol):
    """Minimal protocol required from a Transformers text-generation pipeline."""

    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        return_full_text: bool,
    ) -> list[dict[str, Any]]:
        """Generate coaching text from a prompt."""


def build_coaching_prompt(request: DramaCoachRequest) -> str:
    """Build a bounded prompt from scoring output and retrieved rules."""
    if not request.expected_text.strip():
        raise DramaCoachError("Expected text cannot be empty.")
    if not request.score_result.words:
        raise DramaCoachError("Pronunciation scores cannot be empty.")
    if not request.acting_direction.strip():
        raise DramaCoachError("Acting direction cannot be empty.")

    failed_words = ", ".join(request.score_result.failed_words) or "none"
    rules = "\n".join(
        f"- {rule.metadata.get('title', rule.rule_id)}: {rule.document}"
        for rule in request.rules
    ) or "- No specific phonetic rule was retrieved."

    return (
        "You are a supportive pronunciation and drama coach.\n"
        "Give concise, actionable feedback in English.\n"
        "Do not invent phonetic rules. Address only the failed words and "
        "include one short acting direction.\n\n"
        f"Expected text: {request.expected_text.strip()}\n"
        f"Failed words: {failed_words}\n"
        f"Acting direction: {request.acting_direction.strip()}\n"
        f"Relevant phonetic rules:\n{rules}\n\n"
        "Response format:\n"
        "Pronunciation: <one or two sentences>\n"
        "Practice: <one concrete exercise>\n"
        "Performance: <one acting direction>"
    )


def create_local_coach_pipeline(
    model_name: str = DEFAULT_COACH_MODEL,
) -> TextGenerationPipeline:
    """Create a local text-generation pipeline lazily."""
    if not model_name.strip():
        raise DramaCoachError("The coach model name cannot be empty.")

    try:
        from transformers import pipeline
    except ImportError as error:
        raise DramaCoachError(
            "Transformers is required to create the local coach pipeline."
        ) from error

    try:
        return pipeline(task="text-generation", model=model_name)
    except (OSError, RuntimeError, ValueError) as error:
        raise DramaCoachError(
            f"Unable to load coach model {model_name!r}: {error}"
        ) from error


def generate_coaching(
    request: DramaCoachRequest,
    *,
    model_name: str | None = None,
    generator: TextGenerationPipeline | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> DramaCoachResponse:
    """Generate coaching feedback using an injected or local LLM pipeline."""
    if max_new_tokens <= 0:
        raise DramaCoachError("max_new_tokens must be positive.")

    prompt = build_coaching_prompt(request)
    resolved_model_name = model_name or os.getenv(
        "COACH_MODEL_NAME",
        DEFAULT_COACH_MODEL,
    )
    active_generator = generator or create_local_coach_pipeline(resolved_model_name)

    try:
        raw_response = active_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise DramaCoachError(f"Coach generation failed: {error}") from error

    if not isinstance(raw_response, list) or not raw_response:
        raise DramaCoachError("The coach model returned no generated response.")
    generated_text = raw_response[0].get("generated_text")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise DramaCoachError("The coach model returned empty feedback.")

    return DramaCoachResponse(
        feedback=generated_text.strip(),
        prompt=prompt,
        model_name=resolved_model_name,
    )
