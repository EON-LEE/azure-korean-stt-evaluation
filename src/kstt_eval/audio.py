"""Audio helpers: decoding, deterministic resampling, WAV/PCM encoding."""

from __future__ import annotations

import hashlib
import io
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def decode_audio(data: bytes) -> tuple[np.ndarray, int, int]:
    """Decode encoded audio bytes. Returns (float32 samples [frames, channels], sr, channels)."""
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples, int(sr), int(samples.shape[1])


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def resample(mono: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return mono.astype(np.float32, copy=False)
    frac = Fraction(sr_out, sr_in)
    return resample_poly(mono, frac.numerator, frac.denominator).astype(np.float32)


def float_to_pcm16(mono: np.ndarray) -> np.ndarray:
    return (np.clip(mono, -1.0, 1.0) * 32767.0).round().astype("<i2")


def encode_wav_pcm16(mono: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, float_to_pcm16(mono), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_pcm16(path: Path, target_sr: int | None = None) -> tuple[bytes, int]:
    """Read a WAV file as mono little-endian PCM16 bytes, optionally resampled."""
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = to_mono(samples)
    if target_sr and target_sr != sr:
        mono = resample(mono, sr, target_sr)
        sr = target_sr
    return float_to_pcm16(mono).tobytes(), int(sr)


def pcm_chunks(pcm: bytes, sr: int, chunk_ms: int) -> list[bytes]:
    """Split mono PCM16 bytes into fixed-duration chunks. The last chunk may be shorter."""
    if sr <= 0 or chunk_ms <= 0:
        raise ValueError(f"sample rate and chunk_ms must be positive (sr={sr}, chunk_ms={chunk_ms})")
    if len(pcm) % 2:
        raise ValueError("PCM16 byte length must be even")
    step = int(sr * chunk_ms / 1000) * 2
    if step == 0:
        raise ValueError(f"chunk_ms={chunk_ms} is shorter than one sample at {sr} Hz")
    return [pcm[i : i + step] for i in range(0, len(pcm), step)]


def silence_pcm16(sr: int, seconds: float) -> bytes:
    return b"\x00\x00" * int(round(sr * seconds))
