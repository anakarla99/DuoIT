"""Audio loading and normalization for the CAPT scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1


class AudioPreprocessingError(ValueError):
    """Raised when an audio input cannot be safely prepared for scoring."""


@dataclass(frozen=True)
class PreprocessedAudio:
    """Normalized audio samples and the metadata required by scoring models."""

    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def duration_seconds(self) -> float:
        """Return the duration of the normalized audio."""
        return self.samples.size / self.sample_rate


def preprocess_audio(
    audio_source: str | Path | BinaryIO,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> PreprocessedAudio:
    """Load audio and normalize it to 16 kHz, mono, and finite float samples.

    ``audio_source`` may be a filesystem path or a seekable binary file object.
    Librosa performs channel mixing and resampling during loading so the output
    is ready for the Hugging Face audio model pipeline.
    """
    if target_sample_rate <= 0:
        raise AudioPreprocessingError("The target sample rate must be positive.")

    try:
        samples, sample_rate = librosa.load(
            audio_source,
            sr=target_sample_rate,
            mono=True,
        )
    except (OSError, EOFError, ValueError, RuntimeError) as error:
        raise AudioPreprocessingError(
            f"Unable to load audio input: {error}"
        ) from error

    if samples.size == 0:
        raise AudioPreprocessingError("Audio input is empty.")
    if not np.isfinite(samples).all():
        raise AudioPreprocessingError("Audio input contains non-finite samples.")
    if not np.any(samples):
        raise AudioPreprocessingError("Audio input contains no audible signal.")

    normalized_samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(normalized_samples)))
    if peak > 1.0:
        normalized_samples = normalized_samples / peak

    return PreprocessedAudio(
        samples=normalized_samples,
        sample_rate=sample_rate,
        channels=DEFAULT_CHANNELS,
    )
