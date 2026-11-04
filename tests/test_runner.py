import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest

import kstt_eval.runner as runner_mod
from kstt_eval.config import ModelConfig
from kstt_eval.runner import Cache, Control, LockHeld, Probes, ResourceLimiter, RunError, RunPlan, Timings, execute, \
    ledger_summary, load_results, plan_summary, select_entries
from kstt_eval.types import AuthError, ManifestEntry, NetworkError, PermanentError, Transcription, TransientError
from kstt_eval.util import CorruptLogError, seed_order

SECRET = "Bearer " + "abc.def.ghi"


class FakeAdapter:
    """Returns scripted outcomes per (route, utt_id) or utt_id. A list is consumed one item per call."""

    def __init__(self, cfg, state):
        self.cfg = cfg
        self.state = state

    def signature(self):
        skip = ("concurrency", "max_retries", "max_attempts", "resource")
        return {"adapter": "fake", "params": {k: v for k, v in sorted(self.cfg.params.items()) if k not in skip}}

    async def transcribe(self, audio):
        self.state["calls"].append((self.cfg.key, audio.utt_id))
        if self.state.get("on_call"):
            self.state["on_call"](self.cfg.key, audio.utt_id)
        if self.state.get("delay"):
            await asyncio.sleep(self.state["delay"])
        script = self.state["script"]
        outcome = script.get((self.cfg.key, audio.utt_id), script.get(audio.utt_id, "기본 결과"))
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome = outcome()
        if isinstance(outcome, BaseException):
            raise outcome
        return Transcription(text=outcome, latency={"request_s": 0.1})

    async def aclose(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    state = {"script": {}, "calls": [], "network_ok": True, "token_ok": [True], "logs": []}
    monkeypatch.setattr(runner_mod, "create_adapter", lambda cfg: FakeAdapter(cfg, state))
    monkeypatch.setattr(runner_mod, "backoff_s", lambda attempt: 0.0)
    return state


def _entries(n=3, duration=1.0, splits=("eval_clean",)):
    out = []
    for i in range(n):
        out.append(ManifestEntry(utt_id=f"u{i}", split=splits[i % len(splits)], source_repo="r",
                                 source_revision="rev", source_row=i, original_audio_path=f"p/u{i}.pcm",
                                 audio_path=f"audio/u{i}.wav", audio_sha256=f"sha{i}", duration_s=duration,
                                 sample_rate=16000, channels=1, source_sample_rate=16000, reference="참조"))
    return out


def _timings(**kw):
    base = dict(heartbeat_s=0.01, progress_s=0.05, network_probe_s=0.01, token_retry_s=0.01,
                route_cooldown_s=0.01, lock_refresh_s=0.05)
    return Timings(**{**base, **kw})


def _probes(state, **kw):
    async def network():
        return state["network_ok"]

    async def token():
        seq = state["token_ok"]
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return Probes(network=network, token=token, disk_free=lambda p: 10 ** 12, **kw)


def _cfg(key="route", **params):
    return ModelConfig(key=key, adapter="fake", track="A", params={"concurrency": 1, "max_retries": 3, **params},
                       model_id="m")


def _plan(tmp_path, state, run_id="r1", entries=None, max_hours=None, use_cache=True, routes=None, timings=None,
          probes=None, **params):
    return RunPlan(run_id=run_id, runs_dir=tmp_path / "runs", data_dir=tmp_path / "data", subset="smoke",
                   routes=routes or [_cfg(**params)], entries=entries if entries is not None else _entries(),
                   max_audio_hours=max_hours, use_cache=use_cache, timings=timings or _timings(),
                   probes=probes or _probes(state))


def _run(plan, state=None):
    sink = state["logs"].append if state is not None else (lambda m: None)
    return asyncio.run(execute(plan, log=sink))


def _results(plan):
    return load_results(plan.run_dir / "results.jsonl")


def _statuses(plan, route="route"):
    return {u: r["status"] for (k, u), r in _results(plan).items() if k == route}


# --- basic flow ---------------------------------------------------------------------------


def test_run_resume_and_cache(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    out = _run(plan)
    assert out["counts"]["route"] == {"ok": 3}
    assert len(fake["calls"]) == 3
    meta = json.loads((plan.run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["items"] == ["u0", "u1", "u2"] and "route" in meta["routes"]
    assert meta["last_budget_used_audio_hours"] == pytest.approx(3 / 3600, abs=1e-6)
    assert len(meta["invocations"]) == 1 and meta["invocations"][0]["exit_reason"] == "pass_done"
    assert not (plan.run_dir / "lock").exists()
    assert json.loads((plan.run_dir / "progress.json").read_text(encoding="utf-8"))["routes"]["route"]["final"] == 3
    assert list((plan.run_dir / "logs").glob("*.log"))

    out = _run(_plan(tmp_path, fake))  # same run id: nothing left to do
    assert len(fake["calls"]) == 3 and out["counts"]["route"] == {}
    meta = json.loads((plan.run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(meta["invocations"]) == 2

    plan2 = _plan(tmp_path, fake, run_id="r2")
    _run(plan2)  # new run id: served from cache
    assert len(fake["calls"]) == 3
    assert all(r["cached"] and r["status"] == "ok" for r in _results(plan2).values())

    _run(_plan(tmp_path, fake, run_id="r3", use_cache=False))
    assert len(fake["calls"]) == 6


def test_throttle_is_not_an_attempt(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("HTTP 429", status=429, retry_after_s=0), "성공"]
    plan = _plan(tmp_path, fake, entries=_entries(1))
    _run(plan)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["throttled"], rec["text"]) == ("ok", 1, 1, "성공")
    assert ledger_summary(plan.run_dir)["audio_hours"] == pytest.approx(1 / 3600)


def test_ambiguous_error_is_not_resent(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("read timeout", ambiguous=True), "never"]
    plan = _plan(tmp_path, fake, entries=_entries(1))
    _run(plan)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["error"]["ambiguous"], rec["error"]["kind"]) == \
        ("failed", 1, True, "ambiguous")


def test_retries_are_capped(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("503", status=503, retry_after_s=0) for _ in range(10)]
    plan = _plan(tmp_path, fake, entries=_entries(1), max_retries=2)
    _run(plan)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"]) == ("failed", 3)


def test_budget_exhausted(tmp_path, fake):
    plan = _plan(tmp_path, fake, max_hours=1.5 / 3600)
    out = _run(plan)
    assert sorted(_statuses(plan).values()) == ["ok", "skipped", "skipped"]
    assert {r["reason"] for r in _results(plan).values() if r["status"] == "skipped"} == {"budget_exhausted"}
    assert out["budget_exhausted"] is True


def test_budget_accumulates_across_invocations(tmp_path, fake):
    _run(_plan(tmp_path, fake, max_hours=2.5 / 3600))
    assert len(fake["calls"]) == 2
    _run(_plan(tmp_path, fake, max_hours=2.5 / 3600))  # ledger still holds 2 s of 2.5 s
    assert len(fake["calls"]) == 2
    plan = _plan(tmp_path, fake, max_hours=3 / 3600)
    out = _run(plan)
    assert len(fake["calls"]) == 3 and set(_statuses(plan).values()) == {"ok"}
    assert out["ledger_audio_hours"] == pytest.approx(3 / 3600)
    assert out["budget_used_audio_hours"] == pytest.approx(1 / 3600)


def test_empty_text_and_unexpected_exception(tmp_path, fake):
    fake["script"]["u0"] = "   "
    fake["script"]["u1"] = KeyError("bug")
    plan = _plan(tmp_path, fake)
    _run(plan)
    recs = _results(plan)
    assert recs[("route", "u0")]["status"] == "empty"
    assert recs[("route", "u1")]["status"] == "failed" and recs[("route", "u1")]["error"]["unexpected"] is True
    assert recs[("route", "u2")]["status"] == "ok"


def test_failed_items_are_retried_on_resume(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("t", ambiguous=True), "다시 성공"]
    plan = _plan(tmp_path, fake)
    _run(plan)
    assert _statuses(plan)["u0"] == "failed"
    _run(_plan(tmp_path, fake))
    assert _statuses(plan)["u0"] == "ok"
    assert len(fake["calls"]) == 4
    assert _results(plan)[("route", "u0")]["attempts"] == 2


def test_attempt_cap_makes_failure_final(tmp_path, fake):
    fake["script"]["u0"] = lambda: TransientError("t", ambiguous=True)
    for expected in ("failed", "failed_final", "failed_final"):
        plan = _plan(tmp_path, fake, entries=_entries(1))
        plan.max_attempts = 2
        _run(plan)
        assert _statuses(plan)["u0"] == expected
    assert len(fake["calls"]) == 2
    assert _results(plan)[("route", "u0")]["reason"] == "attempt_cap"


def test_permanent_error_final_after_two_and_redacted(tmp_path, fake):
    fake["script"]["u0"] = lambda: PermanentError(f"HTTP 400 {SECRET}", status=400)
    plan = _plan(tmp_path, fake, entries=_entries(1))
    _run(plan)
    assert _statuses(plan)["u0"] == "failed"
    _run(_plan(tmp_path, fake, entries=_entries(1)))
    assert _statuses(plan)["u0"] == "failed_final"
    for name in ("results.jsonl", "ledger.jsonl", "run.json"):
        assert "abc.def.ghi" not in (plan.run_dir / name).read_text(encoding="utf-8")
    log_text = "".join(p.read_text(encoding="utf-8") for p in (plan.run_dir / "logs").glob("*.log"))
    assert "abc.def.ghi" not in log_text


def test_signature_change_requires_new_run_id(tmp_path, fake):
    _run(_plan(tmp_path, fake))
    with pytest.raises(RunError):
        _run(_plan(tmp_path, fake, language="en"))
    with pytest.raises(RunError):
        _run(_plan(tmp_path, fake, entries=_entries(2)))


def test_plan_summary_counts(tmp_path, fake):
    _run(_plan(tmp_path, fake, entries=_entries(2)))
    rows = plan_summary(_plan(tmp_path, fake, entries=_entries(2)))
    assert (rows[0]["done"], rows[0]["to_call"]) == (2, 0)
    rows = plan_summary(_plan(tmp_path, fake, run_id="new", entries=_entries(3)))
    assert (rows[0]["done"], rows[0]["cached"], rows[0]["to_call"]) == (0, 2, 1)
    assert rows[0]["audio_hours_to_call"] == pytest.approx(1 / 3600, abs=1e-4)
    assert sorted(fake["calls"]) == [("route", "u0"), ("route", "u1")]


def test_cache_path_depends_on_signature(tmp_path):
    cache = Cache(tmp_path)
    a = cache.path("r", {"x": 1}, "u", "sha")
    assert a == cache.path("r", {"x": 1}, "u", "sha")
    assert a != cache.path("r", {"x": 2}, "u", "sha")
    assert a != cache.path("r", {"x": 1}, "u", "sha2")


# --- crash recovery -----------------------------------------------------------------------


def test_cache_written_but_result_lost(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    _run(plan)
    (plan.run_dir / "results.jsonl").unlink()
    _run(_plan(tmp_path, fake))
    assert len(fake["calls"]) == 3
    assert all(r["cached"] for r in _results(plan).values())


def test_truncated_last_line_is_repaired(tmp_path, fake):
    plan = _plan(tmp_path, fake, entries=_entries(2))
    _run(plan)
    results = plan.run_dir / "results.jsonl"
    with open(results, "ab") as f:
        f.write(b'{"route": "route", "utt_')
    with open(plan.run_dir / "ledger.jsonl", "ab") as f:
        f.write(b'{"op": "res')
    _run(_plan(tmp_path, fake, entries=_entries(2)), fake)
    assert any("damaged" in m or "truncat" in m for m in fake["logs"])
    assert results.read_bytes().endswith(b"\n")
    assert len(load_results(results)) == 2 and len(fake["calls"]) == 2


def test_damaged_middle_line_stops(tmp_path, fake):
    plan = _plan(tmp_path, fake, entries=_entries(2))
    _run(plan)
    results = plan.run_dir / "results.jsonl"
    lines = results.read_bytes().splitlines(keepends=True)
    results.write_bytes(lines[0] + b"{broken\n" + lines[1])
    with pytest.raises(CorruptLogError):
        _run(_plan(tmp_path, fake, entries=_entries(2)))
    assert not (plan.run_dir / "lock").exists()


def test_lock_held_by_live_process(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    info = {"pid": os.getpid(), "host": socket.gethostname(), "token": "x", "started_at": "t",
            "refreshed_ts": time.time() - 3600}
    (plan.run_dir / "lock").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(LockHeld):
        _run(plan)
    assert fake["calls"] == []


def test_stale_lock_from_other_host_is_reclaimed(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    info = {"pid": os.getpid(), "host": "other-host", "token": "x", "started_at": "t",
            "refreshed_ts": time.time() - 3600}
    (plan.run_dir / "lock").write_text(json.dumps(info), encoding="utf-8")
    _run(plan)
    assert len(fake["calls"]) == 3 and not (plan.run_dir / "lock").exists()


def test_recent_lock_from_other_host_is_kept(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    info = {"pid": 1, "host": "other-host", "token": "x", "started_at": "t", "refreshed_ts": time.time()}
    (plan.run_dir / "lock").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(LockHeld):
        _run(plan)


def test_recent_lock_of_dead_local_process_is_reclaimed(tmp_path, fake):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    info = {"pid": proc.pid, "host": socket.gethostname(), "token": "x", "started_at": "t",
            "refreshed_ts": time.time()}
    (plan.run_dir / "lock").write_text(json.dumps(info), encoding="utf-8")
    _run(plan)
    assert len(fake["calls"]) == 3 and not (plan.run_dir / "lock").exists()


def test_stop_file_at_start(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    (plan.run_dir / "STOP").write_text("", encoding="utf-8")
    out = _run(plan)
    assert out["stopped"] is True and fake["calls"] == []
    assert out["remaining"] == {"route": 3}


def test_stop_file_during_run_finishes_in_flight(tmp_path, fake):
    plan = _plan(tmp_path, fake, entries=_entries(6))
    fake["delay"] = 0.05
    fake["on_call"] = lambda k, u: (plan.run_dir / "STOP").write_text("", encoding="utf-8")
    out = _run(plan)
    assert out["stopped"] is True
    assert len(fake["calls"]) == 1 and len(_results(plan)) == 1


# --- pauses and route health --------------------------------------------------------------


def test_network_outage_pauses_and_resumes_without_attempts(tmp_path, fake):
    err = NetworkError("connect failed")
    fake["script"]["u0"] = [err, err, err, "복구"]
    plan = _plan(tmp_path, fake, entries=_entries(1), timings=_timings(network_errors=2, network_resources=1))
    _run(plan, fake)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["text"]) == ("ok", 1, "복구")
    assert any("paused (paused_network)" in m for m in fake["logs"])
    assert any("resumed (paused_network)" in m for m in fake["logs"])
    assert ledger_summary(plan.run_dir)["audio_hours"] == pytest.approx(1 / 3600)


def test_local_network_errors_fail_without_attempts(tmp_path, fake):
    fake["script"]["u0"] = lambda: NetworkError("connect failed")
    plan = _plan(tmp_path, fake, entries=_entries(1), timings=_timings(network_errors=99))
    _run(plan)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["reason"]) == ("failed", 0, "network")
    assert len(fake["calls"]) == 4


def test_auth_error_waits_for_login(tmp_path, fake):
    fake["script"]["u0"] = [AuthError("az login expired"), "로그인 후"]
    fake["token_ok"] = [False, True]
    plan = _plan(tmp_path, fake, entries=_entries(2))
    _run(plan, fake)
    assert set(_statuses(plan).values()) == {"ok"}
    assert any("needs_login" in m for m in fake["logs"])
    assert _results(plan)[("route", "u0")]["attempts"] == 1


def test_unauthorized_with_expired_token_pauses(tmp_path, fake):
    fake["script"]["u0"] = [PermanentError("HTTP 401", status=401), "재시도"]
    fake["token_ok"] = [False, True]
    plan = _plan(tmp_path, fake, entries=_entries(1))
    _run(plan, fake)
    rec = _results(plan)[("route", "u0")]
    assert (rec["status"], rec["attempts"]) == ("ok", 1)
    assert any("needs_login" in m for m in fake["logs"])


def test_route_errors_block_only_that_route(tmp_path, fake):
    fake["script"].update({("bad", f"u{i}"): (lambda: PermanentError("HTTP 403", status=403)) for i in range(3)})
    routes = [_cfg("bad", resource="res-a"), _cfg("good", resource="res-b")]
    plan = _plan(tmp_path, fake, routes=routes, timings=_timings(route_errors=2, route_cooldowns=1))
    out = _run(plan, fake)
    assert out["blocked_routes"] == ["bad"]
    assert out["remaining"] == {"bad": 3, "good": 0}
    assert set(_statuses(plan, "good").values()) == {"ok"}
    assert _statuses(plan, "bad") == {}
    assert sum(1 for k, _ in fake["calls"] if k == "bad") == 3  # 2 errors -> cooldown -> 1 probe -> blocked
    assert ledger_summary(plan.run_dir)["attempts"] == 3


def test_route_recovers_after_cooldown_probe(tmp_path, fake):
    fake["script"]["u0"] = [PermanentError("404", status=404), PermanentError("404", status=404), "복구"]
    plan = _plan(tmp_path, fake, timings=_timings(route_errors=2, route_cooldowns=1))
    out = _run(plan, fake)
    assert out["blocked_routes"] == [] and set(_statuses(plan).values()) == {"ok"}
    assert any("route reachable again" in m for m in fake["logs"])


def test_limiter_halves_and_recovers():
    async def scenario():
        ctl = Control(Timings(throttle_recover_s=0.0), Probes(), lambda m: None)
        lim = ResourceLimiter("res", 8, ctl)
        lim.throttled()
        assert lim.limit == 4
        lim.throttled()
        lim.throttled()
        lim.throttled()
        assert lim.limit == 1
        assert await lim.acquire()  # recovery runs while waiting
        assert lim.limit == 2
        await lim.release()

        ctl2 = Control(Timings(throttle_recover_s=3600), Probes(), lambda m: None)
        lim2 = ResourceLimiter("res", 4, ctl2)
        lim2.throttled()
        lim2._recover()
        assert lim2.limit == 2

    asyncio.run(scenario())


def test_resource_limit_caps_concurrency(tmp_path, fake):
    peak = {"now": 0, "max": 0}

    async def slow():
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        await asyncio.sleep(0.02)
        peak["now"] -= 1

    class Counting(FakeAdapter):
        async def transcribe(self, audio):
            await slow()
            return await super().transcribe(audio)

    runner_mod.create_adapter = lambda cfg: Counting(cfg, fake)
    routes = [_cfg("a", concurrency=3, resource="shared"), _cfg("b", concurrency=3, resource="shared")]
    plan = _plan(tmp_path, fake, routes=routes, entries=_entries(6))
    plan.resource_limits = {"shared": 2}
    _run(plan)
    assert peak["max"] == 2 and len(fake["calls"]) == 12


# --- latency validity ---------------------------------------------------------------------


def test_clock_gap_marks_latency_invalid(tmp_path, fake):
    ticks = {"n": 0}

    def wall():
        ticks["n"] += 1
        return ticks["n"] * 1000.0

    fake["delay"] = 0.05
    plan = _plan(tmp_path, fake, entries=_entries(1), probes=_probes(fake, wall_clock=wall))
    _run(plan, fake)
    rec = _results(plan)[("route", "u0")]
    assert rec["status"] == "ok" and rec["latency_invalid"] is True
    assert any("clock gap" in m for m in fake["logs"])


def test_normal_run_latency_valid(tmp_path, fake):
    plan = _plan(tmp_path, fake, entries=_entries(1))
    _run(plan)
    assert _results(plan)[("route", "u0")]["latency_invalid"] is False


# --- selection ----------------------------------------------------------------------------


def test_select_entries_seed_limit_and_per_split():
    entries = _entries(200, splits=("eval_clean", "eval_other"))
    sample = select_entries(entries, "s1", per_split=20)
    assert sum(e.split == "eval_clean" for e in sample) == 20 and len(sample) == 40
    assert [e.utt_id for e in sample] == [e.utt_id for e in entries if e in sample]  # manifest order
    assert select_entries(entries, "s1", per_split=20) == sample
    small = select_entries(entries, "s1", limit=10)
    assert len(small) == 10 and {e.utt_id for e in small} <= {e.utt_id for e in sample}
    assert select_entries(entries, "s2", per_split=20) != sample
    assert select_entries(entries, "s1") == entries
    with pytest.raises(RunError):
        select_entries(entries, "s1", limit=5, per_split=5)


def test_seed_order_mixes_splits():
    entries = _entries(200, splits=("eval_clean", "eval_other"))
    split = {e.utt_id: e.split for e in entries}
    head = seed_order("kstt-1", [e.utt_id for e in entries])[:50]
    clean = sum(split[u] == "eval_clean" for u in head)
    assert 15 <= clean <= 35


def test_pass_follows_seed_order(tmp_path, fake):
    entries = _entries(10)
    plan = _plan(tmp_path, fake, entries=entries)
    _run(plan)
    assert [u for _, u in fake["calls"]] == seed_order(plan.seed, [e.utt_id for e in entries])
