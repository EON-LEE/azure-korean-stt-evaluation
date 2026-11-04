"""Shared data types.

Adapters only ever receive :class:`AudioInput`, which has no reference text field.
Reference transcripts stay in :class:`ManifestEntry` and are used only for scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioInput:
    utt_id: str
    split: str
    path: Path
    duration_s: float
    sample_rate: int
    sha256: str


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    split: str
    source_repo: str
    source_revision: str
    source_row: int
    original_audio_path: str
    audio_path: str
    audio_sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    source_sample_rate: int
    reference: str

    def to_audio_input(self, base_dir: Path) -> AudioInput:
        return AudioInput(
            utt_id=self.utt_id,
            split=self.split,
            path=(base_dir / self.audio_path).resolve(),
            duration_s=self.duration_s,
            sample_rate=self.sample_rate,
            sha256=self.audio_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ManifestEntry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass
class Transcription:
    """Adapter output for one utterance."""

    text: str
    raw: Any = None
    latency: dict[str, float] = field(default_factory=dict)
    partials: int = 0
    usage: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcription":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


class AdapterError(Exception):
    """Base class for adapter errors."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class TransientError(AdapterError):
    """Retryable error (throttling, 5xx, timeouts, dropped connections).

    ``ambiguous`` marks failures after audio was already sent, where the service may have
    processed (and billed) the request. The runner does not resend those by default.
    """

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 retry_after_s: float | None = None, ambiguous: bool = False):
        super().__init__(message, status=status, code=code)
        self.retry_after_s = retry_after_s
        self.ambiguous = ambiguous


class PermanentError(AdapterError):
    """Non-retryable error (bad request, auth, unsupported model/locale)."""


class ConfigError(Exception):
    """Model or experiment configuration is incomplete or invalid."""
