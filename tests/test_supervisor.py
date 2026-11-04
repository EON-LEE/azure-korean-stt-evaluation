import json
import socket
import threading
import time

import pytest

import kstt_eval.supervisor as sup_mod
from kstt_eval.runner import LockHeld, RunLock, load_results
from kstt_eval.supervisor import EXIT_INCOMPLETE, EXIT_OK, EXIT_STOPPED, render_status, run_status, supervise
from kstt_eval.types import PermanentError, TransientError
from test_runner import _cfg, _entries, _plan, _timings, fake  # noqa: F401  (fake is a fixture)


def _supervise(plan, state, **kw):
    kw.setdefault("cooldowns_s", (0.0,))
    return supervise(plan, log=state["logs"].append, stay_awake=False, **kw)


def _meta(plan):
    return json.loads((plan.run_dir / "run.json").read_text(encoding="utf-8"))


def test_supervise_finishes_and_calls_on_finish(tmp_path, fake):
    finished = []
    plan = _plan(tmp_path, fake)
    assert _supervise(plan, fake, on_finish=finished.append) == EXIT_OK
    assert finished == [plan]
    inv = _meta(plan)["invocations"]
    assert len(inv) == 1 and inv[0]["id"].endswith("-p1")
    assert not (plan.run_dir / "lock").exists()
    assert list((plan.run_dir / "logs").glob("supervise-*.log"))


def test_supervise_retries_failed_items_in_next_pass(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("HTTP 503", status=503, retry_after_s=0), "두 번째 패스"]
    plan = _plan(tmp_path, fake, max_retries=0)
    assert _supervise(plan, fake) == EXIT_OK
    rec = load_results(plan.run_dir / "results.jsonl")[("route", "u0")]
    assert (rec["status"], rec["attempts"], rec["text"]) == ("ok", 2, "두 번째 패스")
    ids = [i["id"] for i in _meta(plan)["invocations"]]
    assert len(ids) == 2 and ids[0].endswith("-p1") and ids[1].endswith("-p2") and len(set(ids)) == 2


def test_supervise_stop_file_at_start(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    plan.run_dir.mkdir(parents=True)
    (plan.run_dir / "STOP").touch()
    assert _supervise(plan, fake) == EXIT_STOPPED
    assert fake["calls"] == [] and not (plan.run_dir / "lock").exists()


def test_supervise_stop_during_cooldown(tmp_path, fake):
    fake["script"]["u0"] = [TransientError("HTTP 503", status=503, retry_after_s=0), "안 보냄"]
    plan = _plan(tmp_path, fake, max_retries=0)
    timer = threading.Timer(0.3, lambda: (plan.run_dir / "STOP").touch())
    timer.start()
    t0 = time.monotonic()
    try:
        assert _supervise(plan, fake, cooldowns_s=(30.0,)) == EXIT_STOPPED
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 10
    assert len(_meta(plan)["invocations"]) == 1
    assert not (plan.run_dir / "lock").exists()


def test_supervise_budget_exhausted(tmp_path, fake):
    finished = []
    plan = _plan(tmp_path, fake, max_hours=1.5 / 3600)
    assert _supervise(plan, fake, on_finish=finished.append) == EXIT_INCOMPLETE
    assert finished == [plan]
    assert any("budget exhausted" in m for m in fake["logs"])


def test_supervise_stops_when_only_blocked_routes_remain(tmp_path, fake):
    fake["script"].update({("bad", f"u{i}"): (lambda: PermanentError("HTTP 403", status=403)) for i in range(3)})
    routes = [_cfg("bad", resource="res-a"), _cfg("good", resource="res-b")]
    plan = _plan(tmp_path, fake, routes=routes, timings=_timings(route_errors=2, route_cooldowns=1))
    assert _supervise(plan, fake) == EXIT_INCOMPLETE
    assert plan.skip_routes == {"bad"}
    assert any("only blocked routes remain" in m for m in fake["logs"])


def test_supervise_stops_after_idle_passes(tmp_path, fake, monkeypatch):
    calls = []

    async def no_progress(plan, log, *, lock=None):
        calls.append(plan.invocation)
        return {"stopped": False, "blocked_routes": [], "remaining": {"route": 1}, "finalized": 0,
                "budget_used_audio_hours": 0.0, "budget_exhausted": False, "ledger_audio_hours": 0.0}

    monkeypatch.setattr(sup_mod, "execute", no_progress)
    plan = _plan(tmp_path, fake)
    assert _supervise(plan, fake, max_idle_passes=3, cooldowns_s=(0.0, 0.01)) == EXIT_INCOMPLETE
    assert len(calls) == 3 and len(set(calls)) == 3
    assert any("3 pass(es) without progress" in m for m in fake["logs"])


def test_supervise_refuses_held_lock(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    holder = RunLock(plan.run_dir).acquire()
    try:
        with pytest.raises(LockHeld):
            _supervise(plan, fake)
    finally:
        holder.release()
    assert fake["calls"] == []


def test_status_after_finished_run(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    assert _supervise(plan, fake) == EXIT_OK
    st = run_status(plan.run_dir)
    assert st["state"] == "stopped (no live lock)" and st["lock"] is None
    assert st["routes"]["route"]["final"] == 3 and st["routes"]["route"]["remaining"] == 0
    assert st["budget"]["audio_hours"] == pytest.approx(3 / 3600) and st["invocations"] == 1
    text = render_status(st)
    assert text.startswith("run r1: stopped (no live lock)") and "route" in text


def test_status_live_and_stale_lock(tmp_path, fake):
    plan = _plan(tmp_path, fake)
    assert _supervise(plan, fake) == EXIT_OK
    (plan.run_dir / "progress.json").write_text(json.dumps({"state": "running", "pauses": {"network": "probe"},
                                                            "routes": {"route": {"state": "active"}}}),
                                                encoding="utf-8")
    holder = RunLock(plan.run_dir).acquire()
    try:
        st = run_status(plan.run_dir)
        assert st["state"] == "running" and st["pauses"] == {"network": "probe"}
        assert st["routes"]["route"]["state"] == "active"
    finally:
        holder.release()

    old = {"pid": 999999, "host": "elsewhere-" + socket.gethostname(), "refreshed_ts": time.time() - 3600}
    (plan.run_dir / "lock").write_text(json.dumps(old), encoding="utf-8")
    st = run_status(plan.run_dir)
    assert st["state"] == "stopped (stale lock)" and st["pauses"] == {}
    (plan.run_dir / "STOP").touch()
    assert "[STOP file present]" in render_status(run_status(plan.run_dir))


def test_status_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_status(tmp_path / "absent")
