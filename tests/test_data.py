import numpy as np
import pytest

from kstt_eval.audio import encode_wav_pcm16
from kstt_eval.data import (
    SOURCES, DataError, RowCheck, _write_wav_entry, load_manifest, manifest_path, pick_smoke, utt_id_from_path,
)
from collections import Counter


def test_pick_smoke_is_deterministic_and_order_independent():
    items = [("a", f"id{i}") for i in range(10)] + [("b", f"id{i}") for i in range(10, 13)]
    picked = pick_smoke(items, lambda x: x[0], lambda x: x[1], {"a": 2, "b": 1})
    again = pick_smoke(list(reversed(items)), lambda x: x[0], lambda x: x[1], {"a": 2, "b": 1})
    assert picked == again
    assert [x[0] for x in picked] == ["a", "a", "b"]
    with pytest.raises(DataError):
        pick_smoke(items, lambda x: x[0], lambda x: x[1], {"b": 4})


def test_row_check_records_issues():
    spec = SOURCES["eval_clean"]
    check = RowCheck(spec, Counter(), [])
    good = spec.path_prefix + "KsponSpeech_E00001.wav"
    assert check.validate(0, good, "가나다") == "KsponSpeech_E00001"
    assert check.validate(1, good, "가나다") is None
    assert check.validate(2, "other/KsponSpeech_E00002.wav", "가나다") is None
    assert check.validate(3, spec.path_prefix + "KsponSpeech_E00003.wav", "") is None
    assert [i["issue"] for i in check.issues] == ["duplicate_id", "unexpected_path", "empty_reference"]


def test_utt_id_and_manifest_paths(tmp_path):
    assert utt_id_from_path("KsponSpeech_eval\\eval_other\\KsponSpeech_E03001.pcm") == "KsponSpeech_E03001"
    assert manifest_path(tmp_path, "smoke") == tmp_path / "manifests" / "smoke.jsonl"
    assert manifest_path(tmp_path, "full", "eval_clean") == tmp_path / "manifests" / "full" / "eval_clean.jsonl"
    with pytest.raises(ValueError):
        manifest_path(tmp_path, "full")


def test_write_wav_entry_converts_to_16k_mono(tmp_path):
    sr = 22050
    t = np.arange(sr // 2) / sr
    stereo = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 500 * t)], axis=1) * 0.2
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, stereo.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    spec = SOURCES["eval_other"]
    entry, src_sr, ch = _write_wav_entry(spec, tmp_path, "U1", 7, spec.path_prefix + "U1.pcm", "가나", buf.getvalue())
    assert (src_sr, ch) == (sr, 2)
    assert entry.sample_rate == 16000 and entry.channels == 1
    assert entry.duration_s == pytest.approx(0.5, abs=0.001)
    assert (tmp_path / entry.audio_path).exists()
    assert entry.audio_path == "audio/eval_other/U1.wav"
    with pytest.raises(ValueError):
        _write_wav_entry(spec, tmp_path, "U2", 8, "x", "가", encode_wav_pcm16(np.zeros(0, np.float32), 16000))


def test_load_manifest_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path, "smoke")
    p = manifest_path(tmp_path, "full", "eval_clean")
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    with pytest.raises(DataError):
        load_manifest(tmp_path, "full", ("eval_clean",))
