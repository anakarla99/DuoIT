"""Per-word pronunciation and intelligibility scoring."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from backend.scorer.transcription import TranscriptionResult, WordTimestamp

DEFAULT_PRONUNCIATION_THRESHOLD = 0.70
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)


class PronunciationScoringError(ValueError):
    """Raised when pronunciation scoring input is invalid."""


@dataclass(frozen=True)
class WordPronunciationScore:
    """Score and alignment details for one expected word."""

    expected_word: str
    recognized_word: str | None
    score: float
    passed: bool
    start_seconds: float | None
    end_seconds: float | None


@dataclass(frozen=True)
class PronunciationScoreResult:
    """Per-word scores and the words requiring pronunciation feedback."""

    words: tuple[WordPronunciationScore, ...]
    failed_words: tuple[str, ...]
    threshold: float


def _normalize_word(word: str) -> str:
    return word.casefold().replace("’", "'").replace("–", "-")


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def _resolve_threshold(threshold: float | None) -> float:
    configured = (
        threshold
        if threshold is not None
        else float(
            os.getenv(
                "PRONUNCIATION_THRESHOLD",
                str(DEFAULT_PRONUNCIATION_THRESHOLD),
            )
        )
    )
    if not 0.0 <= configured <= 1.0:
        raise PronunciationScoringError(
            "The pronunciation threshold must be between 0 and 1."
        )
    return configured


def score_pronunciation(
    expected_text: str,
    transcription: TranscriptionResult,
    *,
    threshold: float | None = None,
) -> PronunciationScoreResult:
    """Score expected words against sequential Whisper word timestamps.

    This is an intelligibility score, not a phoneme-level GOP score. Exact
    recognized-word matches use Whisper confidence when available; without a
    confidence value they receive 1.0. Missing or mismatched words receive 0.
    """
    expected_words = _tokenize(expected_text)
    if not expected_words:
        raise PronunciationScoringError("Expected text must contain at least one word.")
    if not transcription.words:
        raise PronunciationScoringError("Transcription must contain at least one word.")

    resolved_threshold = _resolve_threshold(threshold)
    scored_words: list[WordPronunciationScore] = []

    for index, expected_word in enumerate(expected_words):
        recognized: WordTimestamp | None = (
            transcription.words[index] if index < len(transcription.words) else None
        )
        is_match = (
            recognized is not None
            and _normalize_word(expected_word) == _normalize_word(recognized.word)
        )
        score = (
            0.0
            if not is_match
            else 1.0 if recognized.confidence is None else recognized.confidence
        )
        scored_words.append(
            WordPronunciationScore(
                expected_word=expected_word,
                recognized_word=None if recognized is None else recognized.word,
                score=score,
                passed=score >= resolved_threshold,
                start_seconds=None if recognized is None else recognized.start_seconds,
                end_seconds=None if recognized is None else recognized.end_seconds,
            )
        )

    return PronunciationScoreResult(
        words=tuple(scored_words),
        failed_words=tuple(
            word.expected_word for word in scored_words if not word.passed
        ),
        threshold=resolved_threshold,
    )
