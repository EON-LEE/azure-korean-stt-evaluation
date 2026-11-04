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


def _manifest(tmp_path, n=2, splits=("eval_clean",)):
    p = tmp_path / "data" / "manifests" / "smoke.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for i in range(n):
            e = ManifestEntry(utt_id=f"u{i}", split=splits[i % len(splits)], source_repo="r", source_revision="v",
                              source_row=i, original_audio_path="p", audio_path=f"audio/u{i}.wav",
                              audio_sha256=f"s{i}", duration_s=3.6, sample_rate=16000, channels=1,
                              source_sample_rate=16000, reference="가")
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


def test_selection_options(no_endpoints, tmp_path, capsys):
    _manifest(tmp_path, n=6, splits=("eval_clean", "eval_other"))
    base = no_endpoints + ["run", "--routes", "fast-azure", "--dry-run"]
    assert main(base + ["--limit", "2", "--sample-per-split", "1"]) == 2
    assert "either --limit or --sample-per-split" in capsys.readouterr().err
    assert main(base + ["--resource-limit", "speech"]) == 2
    assert "NAME=N" in capsys.readouterr().err
    assert main(base + ["--sample-per-split", "2", "--resource-limit", "speech=4"]) == 0
    assert "items=4" in capsys.readouterr().out


def test_supervise_needs_run_id(no_endpoints, tmp_path, capsys):
    _manifest(tmp_path)
    assert main(no_endpoints + ["supervise", "--routes", "fast-azure"]) == 2
    assert "--run-id" in capsys.readouterr().err


def test_status(no_endpoints, tmp_path, capsys):
    assert main(no_endpoints + ["status", "--run", "absent"]) == 2
    run = tmp_path / "runs" / "s1"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "s1", "items": ["u0", "u1"], "routes": {"fast-azure": {}},
                                              "max_audio_hours": 1.0}), encoding="utf-8")
    (run / "results.jsonl").write_text(json.dumps({"route": "fast-azure", "utt_id": "u0", "status": "ok"}) + "\n",
                                       encoding="utf-8")
    capsys.readouterr()
    assert main(no_endpoints + ["status", "--run", "s1"]) == 0
    assert "run s1: stopped (no live lock)" in capsys.readouterr().out
    assert main(no_endpoints + ["status", "--run", "s1", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["routes"]["fast-azure"]["final"] == 1 and st["routes"]["fast-azure"]["remaining"] == 1


def test_real_run_checks_git_and_audio_before_sending(no_endpoints, tmp_path, capsys, monkeypatch):
    import kstt_eval.cli as cli_mod
    from kstt_eval.data import load_manifest

    monkeypatch.setenv("KSTT_SPEECH_KOREACENTRAL_ENDPOINT", "https://example.invalid")
    monkeypatch.setattr(cli_mod, "execute", None)  # any call to the runner would fail the test
    monkeypatch.setattr(cli_mod, "git_info", lambda: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(cli_mod, "load_manifest", lambda data_dir, subset: load_manifest(data_dir, "smoke"))
    _manifest(tmp_path)
    full = no_endpoints + ["run", "--routes", "fast-azure", "--subset", "full", "--max-audio-hours", "1",
                           "--run-id", "f1"]
    assert main(full) == 2
    assert "uncommitted changes" in capsys.readouterr().err
    assert main(full + ["--allow-dirty"]) == 2
    assert "do not match the manifest" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "f1").exists()
