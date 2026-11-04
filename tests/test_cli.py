import json
import os
from pathlib import Path

import pytest

from kstt_eval.cli import main
from kstt_eval.types import ManifestEntry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_endpoints(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("KSTT_"):
            monkeypatch.delenv(name)
    return ["--config-dir", str(ROOT / "configs"), "--env-file", str(tmp_path / "absent.env"),
            "--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"),
            "--reports-dir", str(tmp_path / "reports")]


def _manifest(tmp_path, n=2):
    p = tmp_path / "data" / "manifests" / "smoke.jsonl"
    p.parent.mkdir(parents=True)
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            e = ManifestEntry(utt_id=f"u{i}", split="eval_clean", source_repo="r", source_revision="v", source_row=i,
                              original_audio_path="p", audio_path=f"audio/u{i}.wav", audio_sha256=f"s{i}",
                              duration_s=3.6, sample_rate=16000, channels=1, source_sample_rate=16000,
                              reference="가")
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def test_models_lists_routes(no_endpoints, capsys):
    assert main(no_endpoints + ["models"]) == 0
    out = capsys.readouterr().out
    assert "fast-azure" in out and "vl-azure-speech-vad" in out and "missing env" in out


def test_full_run_needs_budget(no_endpoints, capsys):
    assert main(no_endpoints + ["run", "--subset", "full"]) == 2
    assert "--max-audio-hours" in capsys.readouterr().err


def test_run_refuses_missing_env(no_endpoints, tmp_path, capsys):
    _manifest(tmp_path)
    assert main(no_endpoints + ["run", "--routes", "fast-azure"]) == 2
    assert "missing env" in capsys.readouterr().err


def test_dry_run_prints_plan(no_endpoints, tmp_path, capsys):
    _manifest(tmp_path)
    assert main(no_endpoints + ["run", "--routes", "fast-*,track:B", "--dry-run", "--run-id", "d1"]) == 0
    out = capsys.readouterr().out
    assert "run_id=d1" in out and "(dry run)" in out
    assert "fast-azure" in out and "sdk-azure" in out
    assert "total audio to send (before retries)" in out
    assert not (tmp_path / "runs" / "d1").exists()


def test_unknown_route_and_report_errors(no_endpoints, capsys):
    assert main(no_endpoints + ["run", "--routes", "nope", "--dry-run"]) == 2
    assert main(no_endpoints + ["report", "--run", "absent"]) == 2
