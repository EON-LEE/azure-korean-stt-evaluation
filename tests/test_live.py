"""Live checks against Azure. Run with ``uv run pytest --live -m live`` after ``az login``.

These tests only call non-billable endpoints (token, model list, deployment status).
"""

from pathlib import Path

import pytest

from kstt_eval.cli import main

ROOT = Path(__file__).resolve().parents[1]
ARGS = ["--config-dir", str(ROOT / "configs"), "--env-file", str(ROOT / "configs" / "endpoints.env")]


@pytest.mark.live
def test_doctor_http_routes(capsys):
    code = main(ARGS + ["doctor", "--routes", "fast-azure,fast-mai2,aoai-gpt-4o-mini-transcribe"])
    out = capsys.readouterr().out
    assert code == 0, out
