import numpy as np
import pytest

from kstt_eval.audio import decode_audio, encode_wav_pcm16, pcm_chunks, read_pcm16, resample, silence_pcm16


def _tone(sr: int, seconds: float) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_wav_roundtrip_and_resample(tmp_path):
    wav = encode_wav_pcm16(_tone(16000, 0.5), 16000)
    samples, sr, ch = decode_audio(wav)
    assert (sr, ch, samples.shape[0]) == (16000, 1, 8000)
    path = tmp_path / "a.wav"
    path.write_bytes(wav)
    pcm, sr = read_pcm16(path)
    assert sr == 16000 and len(pcm) == 16000
    pcm24, sr24 = read_pcm16(path, 24000)
    assert sr24 == 24000 and len(pcm24) == 24000
    assert len(resample(_tone(16000, 1.0), 16000, 24000)) == 24000


def test_pcm_chunks_sizes_and_validation():
    pcm = b"\x00\x01" * 1000  # 1000 samples
    chunks = pcm_chunks(pcm, 16000, 20)  # 320 samples per chunk
    assert [len(c) for c in chunks] == [640, 640, 640, 80]
    assert b"".join(chunks) == pcm
    with pytest.raises(ValueError):
        pcm_chunks(pcm, 0, 20)
    with pytest.raises(ValueError):
        pcm_chunks(pcm, 16000, 0)
    with pytest.raises(ValueError):
        pcm_chunks(b"\x00", 16000, 20)
    with pytest.raises(ValueError):
        pcm_chunks(pcm, 10, 1)


def test_silence():
    assert silence_pcm16(16000, 0.5) == b"\x00\x00" * 8000
    assert silence_pcm16(16000, 0) == b""
