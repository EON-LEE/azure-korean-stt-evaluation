import asyncio
import json

import pytest

import kstt_eval.runner as runner_mod
from kstt_eval.config import ModelConfig
from kstt_eval.runner import Cache, RunError, RunPlan, execute, load_results, plan_summary
from kstt_eval.types import ManifestEntry, PermanentError, Transcription, TransientError


class FakeAdapter:
    """Returns scripted outcomes per utt_id. A list is consumed one item per call."""

    def __init__(self, cfg, script, calls):
        self.cfg = cfg
        self.script = script
        self.calls = calls

    def signature(self):
        return {"adapter": "fake", "params": {k: v for k, v in sorted(self.cfg.params.items())
                                              if k not in ("concurrency", "max_retries")}}

    async def transcribe(self, audio):
        self.calls.append((self.cfg.key, audio.utt_id))
        outcome = self.script.get(audio.utt_id, "기본 결과")
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return Transcription(text=outcome, latency={"request_s": 0.1})

    async def aclose(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    state = {"script": {}, "calls": []}
    monkeypatch.setattr(runner_mod, "create_adapter", lambda cfg: FakeAdapter(cfg, state["script"], state["calls"]))
    monkeypatch.setattr(runner_mod, "backoff_s", lambda attempt: 0.0)
    return state


def _entries(n=3, duration=1.0):
    return [ManifestEntry(utt_id=f"u{i}", split="eval_clean", source_repo="r", source_revision="rev", source_row=i,
                          original_audio_path=f"p/u{i}.pcm", audio_path=f"audio/u{i}.wav", audio_sha256=f"sha{i}",
                          duration_s=duration, sample_rate=16000, channels=1, source_sample_rate=16000,
                          reference="참조")
            for i in range(n)]


def _plan(tmp_path, run_id="r1", entries=None, max_hours=None, use_cache=True, **params):
    params = {"concurrency": 1, "max_retries": 3, **params}
    cfg = ModelConfig(key="route", adapter="fake", track="A", params=params, model_id="m")
    return RunPlan(run_id=run_id, runs_dir=tmp_path / "runs", data_dir=tmp_path / "data", subset="smoke",
                   routes=[cfg], entries=entries if entries is not None else _entries(), max_audio_hours=max_hours,
                   use_cache=use_cache)


def _run(plan):
    return asyncio.run(execute(plan, log=lambda m: None))


def _statuses(plan):
    return {u: r["status"] for (_, u), r in load_results(plan.run_dir / "results.jsonl").items()}


def test_run_resume_and_cache(tmp_path, fake):
    plan = _plan(tmp_path)
    out = _run(plan)
    assert out["counts"]["route"] == {"ok": 3}
    assert len(fake["calls"]) == 3
    meta = json.loads((plan.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["items"] == ["u0", "u1", "u2"] and "route" in meta["routes"]
    assert meta["last_budget_used_audio_hours"] == pytest.approx(3 / 3600, abs=1e-6)

    _run(_plan(tmp_path))  # same run id: nothing left to do
    assert len(fake["calls"]) == 3

    plan2 = _plan(tmp_path, run_id="r2")
    _run(plan2)  # new run id: served from cache
    assert len(fake["calls"]) == 3
    recs = load_results(plan2.run_dir / "results.jsonl")
    assert all(r["cached"] and r["status"] == "ok" for r in recs.values())

    _run(_plan(tmp_path, run_id="r3", use_cache=False))
    assert len(fake["calls"]) == 6


def test_throttle_retry_then_success(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("HTTP 429", status=429, retry_after_s=0), "성공"]
    plan = _plan(tmp_path, entries=_entries(1))
    _run(plan)
    rec = load_results(plan.run_dir / "results.jsonl")[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["throttled"], rec["text"]) == ("ok", 2, 1, "성공")


def test_ambiguous_error_is_not_resent(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("read timeout", ambiguous=True), "never"]
    plan = _plan(tmp_path, entries=_entries(1))
    _run(plan)
    rec = load_results(plan.run_dir / "results.jsonl")[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["error"]["ambiguous"]) == ("failed", 1, True)


def test_retries_are_capped(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("503", status=503, retry_after_s=0) for _ in range(10)]
    plan = _plan(tmp_path, entries=_entries(1), max_retries=2)
    _run(plan)
    rec = load_results(plan.run_dir / "results.jsonl")[("route", "u0")]
    assert (rec["status"], rec["attempts"]) == ("failed", 3)


def test_breaker_after_permanent_errors_and_redaction(tmp_path, fake):
    secret = "Bearer abc.def.ghi"
    for i in range(5):
        fake["script"][f"u{i}"] = PermanentError(f"HTTP 401 {secret}", status=401)
    plan = _plan(tmp_path, entries=_entries(5))
    _run(plan)
    recs = load_results(plan.run_dir / "results.jsonl")
    statuses = [recs[("route", f"u{i}")]["status"] for i in range(5)]
    assert statuses == ["failed"] * 3 + ["skipped"] * 2
    assert recs[("route", "u4")]["reason"].startswith("route_disabled")
    assert len(fake["calls"]) == 3
    text = (plan.run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert "abc.def.ghi" not in text


def test_budget_exhausted(tmp_path, fake):
    plan = _plan(tmp_path, max_hours=1.5 / 3600)
    _run(plan)
    assert sorted(_statuses(plan).values()) == ["ok", "skipped", "skipped"]
    recs = load_results(plan.run_dir / "results.jsonl")
    assert {r["reason"] for r in recs.values() if r["status"] == "skipped"} == {"budget_exhausted"}


def test_empty_text_and_unexpected_exception(tmp_path, fake):
    fake["script"]["u0"] = "   "
    fake["script"]["u1"] = KeyError("bug")
    plan = _plan(tmp_path)
    _run(plan)
    recs = load_results(plan.run_dir / "results.jsonl")
    assert recs[("route", "u0")]["status"] == "empty"
    assert recs[("route", "u1")]["status"] == "failed" and recs[("route", "u1")]["error"]["unexpected"] is True
    assert recs[("route", "u2")]["status"] == "ok"


def test_failed_items_are_retried_on_resume(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("t", ambiguous=True), "다시 성공"]
    plan = _plan(tmp_path)
    _run(plan)
    assert _statuses(plan)["u0"] == "failed"
    _run(_plan(tmp_path))
    assert _statuses(plan)["u0"] == "ok"
    assert len(fake["calls"]) == 4


def test_signature_change_requires_new_run_id(tmp_path, fake):
    _run(_plan(tmp_path))
    with pytest.raises(RunError):
        _run(_plan(tmp_path, language="en"))
    with pytest.raises(RunError):
        _run(_plan(tmp_path, entries=_entries(2)))


def test_plan_summary_counts(tmp_path, fake):
    _run(_plan(tmp_path, entries=_entries(2)))
    rows = plan_summary(_plan(tmp_path, entries=_entries(2)))
    assert (rows[0]["done"], rows[0]["to_call"]) == (2, 0)
    rows = plan_summary(_plan(tmp_path, run_id="new", entries=_entries(3)))
    assert (rows[0]["done"], rows[0]["cached"], rows[0]["to_call"]) == (0, 2, 1)
    assert rows[0]["audio_hours_to_call"] == pytest.approx(1 / 3600, abs=1e-4)
    assert fake["calls"] == [("route", "u0"), ("route", "u1")]


def test_cache_path_depends_on_signature(tmp_path):
    cache = Cache(tmp_path)
    a = cache.path("r", {"x": 1}, "u", "sha")
    assert a == cache.path("r", {"x": 1}, "u", "sha")
    assert a != cache.path("r", {"x": 2}, "u", "sha")
    assert a != cache.path("r", {"x": 1}, "u", "sha2")
