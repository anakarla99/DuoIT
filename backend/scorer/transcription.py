"""Whisper transcription with word-level timestamps for CAPT scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy as np

from backend.scorer.audio_preprocessing import (
    DEFAULT_SAMPLE_RATE,
    PreprocessedAudio,
    preprocess_audio,
)

DEFAULT_MODEL_NAME = "openai/whisper-small"


class TranscriptionError(ValueError):
    """Raised when transcription cannot produce valid timestamped words."""


@dataclass(frozen=True)
class WordTimestamp:
    """A transcribed word and its position in the source audio."""

    word: str
    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Timestamped transcription returned by the ASR model."""

    text: str
    words: tuple[WordTimestamp, ...]
    model_name: str


class SpeechRecognitionPipeline(Protocol):
    """Minimal protocol required from a Transformers ASR pipeline."""

    def __call__(
        self,
        audio: dict[str, Any],
        *,
        return_timestamps: str,
    ) -> dict[str, Any]:
        """Transcribe normalized audio."""


def create_whisper_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SpeechRecognitionPipeline:
    """Create a local Transformers Whisper pipeline on first use."""
    if not model_name.strip():
        raise TranscriptionError("The ASR model name cannot be empty.")

    try:
        from transformers import pipeline
    except ImportError as error:
        raise TranscriptionError(
            "Transformers is required to create the ASR pipeline."
        ) from error

    try:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_name,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise TranscriptionError(
            f"Unable to load ASR model {model_name!r}: {error}"
        ) from error


def _parse_word_timestamps(raw_result: dict[str, Any]) -> tuple[WordTimestamp, ...]:
    chunks = raw_result.get("chunks")
    if not isinstance(chunks, list):
        raise TranscriptionError(
            "The ASR model did not return word timestamps. "
            "Ensure return_timestamps='word' is supported."
        )

    words: list[WordTimestamp] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TranscriptionError("The ASR model returned an invalid timestamp chunk.")

        word = chunk.get("text")
        timestamp = chunk.get("timestamp")
        if not isinstance(word, str) or not word.strip():
            continue
        if (
            not isinstance(timestamp, (tuple, list))
            or len(timestamp) != 2
            or not all(isinstance(value, (int, float)) for value in timestamp)
        ):
            raise TranscriptionError(
                f"Invalid timestamp for transcribed word {word!r}."
            )

        start_seconds, end_seconds = float(timestamp[0]), float(timestamp[1])
        if not np.isfinite([start_seconds, end_seconds]).all():
            raise TranscriptionError(
                f"Non-finite timestamp for transcribed word {word!r}."
            )
        if start_seconds < 0 or end_seconds < start_seconds:
            raise TranscriptionError(
                f"Invalid timestamp range for transcribed word {word!r}."
            )

        confidence = chunk.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or not np.isfinite(confidence):
                raise TranscriptionError(
                    f"Invalid confidence for transcribed word {word!r}."
                )
            if not 0.0 <= float(confidence) <= 1.0:
                raise TranscriptionError(
                    f"Confidence must be between 0 and 1 for {word!r}."
                )

        words.append(
            WordTimestamp(
                word=word.strip(),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                confidence=None if confidence is None else float(confidence),
            )
        )

    if not words:
        raise TranscriptionError("The ASR model returned no transcribed words.")
    return tuple(words)


def transcribe_audio(
    audio_source: str | Path | BinaryIO | PreprocessedAudio,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    recognizer: SpeechRecognitionPipeline | None = None,
) -> TranscriptionResult:
    """Transcribe audio and return validated word-level timestamps.

    A recognizer can be injected for tests or for application-managed model
    lifecycles. When omitted, a local Whisper pipeline is created lazily.
    """
    audio = (
        audio_source
        if isinstance(audio_source, PreprocessedAudio)
        else preprocess_audio(audio_source)
    )
    active_recognizer = recognizer or create_whisper_pipeline(model_name)

    try:
        raw_result = active_recognizer(
            {"array": audio.samples, "sampling_rate": DEFAULT_SAMPLE_RATE},
            return_timestamps="word",
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise TranscriptionError(f"Audio transcription failed: {error}") from error

    if not isinstance(raw_result, dict):
        raise TranscriptionError("The ASR model returned an invalid response.")

    text = raw_result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TranscriptionError("The ASR model returned empty transcription text.")

    return TranscriptionResult(
        text=text.strip(),
        words=_parse_word_timestamps(raw_result),
        model_name=model_name,
    )
