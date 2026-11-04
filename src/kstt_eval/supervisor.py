"""Long unattended runs: repeated passes under one lock, plus a read-only status view.

``supervise`` keeps calling :func:`runner.execute` until nothing is left, the budget is spent,
the user creates the STOP file, or passes stop making progress. Exit codes:

    0  every item is final (report written)
    2  configuration, data or log problem
    3  stopped early: budget exhausted, only blocked routes left, or no progress
    4  STOP file
    5  another process holds the run lock
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

from .runner import FINAL_STATUSES, RunLock, RunLog, RunPlan, execute, ledger_summary, load_results, lock_is_stale, \
    new_invocation_id, read_lock
from .util import utc_now

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_INCOMPLETE = 3
EXIT_STOPPED = 4
EXIT_LOCKED = 5
DEFAULT_COOLDOWNS_S = (300.0, 900.0, 1800.0)


@contextlib.contextmanager
def keep_awake(enabled: bool = True) -> Iterator[None]:
    """Ask Windows not to sleep while this thread runs. The request ends with the process."""
    if not enabled or sys.platform != "win32":
        yield
        return
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    try:
        yield
    finally:
        k32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS: clear the request


async def _cooldown(plan: RunPlan, lock: RunLock, seconds: float) -> bool:
    """Wait between passes while keeping the lock fresh. Returns False if STOP appears."""
    end = time.monotonic() + seconds
    step = max(0.01, min(plan.timings.lock_refresh_s, 1.0))
    next_refresh = time.monotonic()
    while time.monotonic() < end:
        if (plan.run_dir / "STOP").exists():
            return False
        if time.monotonic() >= next_refresh:
            lock.refresh()
            next_refresh = time.monotonic() + plan.timings.lock_refresh_s
        await asyncio.sleep(min(step, max(0.0, end - time.monotonic())))
    return not (plan.run_dir / "STOP").exists()


async def _supervise(plan: RunPlan, log: Callable[[str], None], lock: RunLock, cooldowns_s: tuple[float, ...],
                     max_idle_passes: int, on_finish: Callable[[RunPlan], None] | None) -> int:
    base = new_invocation_id()
    idle = 0
    step = 0
    n = 0
    while True:
        if (plan.run_dir / "STOP").exists():
            log("STOP file present; not starting a pass. Delete it to continue.")
            return EXIT_STOPPED
        n += 1
        plan.invocation = f"{base}-p{n}"
        log(f"pass {n} ({plan.invocation}) skip_routes={sorted(plan.skip_routes) or '-'}")
        out = await execute(plan, log, lock=lock)
        if out["stopped"]:
            return EXIT_STOPPED
        plan.skip_routes |= set(out["blocked_routes"])
        left = {k: v for k, v in out["remaining"].items() if v}
        active_left = {k: v for k, v in left.items() if k not in plan.skip_routes}
        progressed = out["finalized"] > 0 or out["budget_used_audio_hours"] > 0
        log(f"pass {n} done: finalized {out['finalized']}, sent {out['budget_used_audio_hours']:.4f} audio h, "
            f"remaining {left or 0}, blocked {sorted(plan.skip_routes) or '-'}")
        if not left:
            if on_finish:
                on_finish(plan)
            log("all items final")
            return EXIT_OK
        if out["budget_exhausted"]:
            log(f"budget exhausted ({out['ledger_audio_hours']:.4f} audio h used); raise --max-audio-hours "
                "and run again to continue")
            if on_finish:
                on_finish(plan)
            return EXIT_INCOMPLETE
        if not active_left:
            log(f"only blocked routes remain: {sorted(left)}; fix them and run again")
            if on_finish:
                on_finish(plan)
            return EXIT_INCOMPLETE
        if progressed:
            idle, step = 0, 0
        else:
            idle += 1
            if idle >= max_idle_passes:
                log(f"{idle} pass(es) without progress; stopping. Check status and logs, then run again")
                if on_finish:
                    on_finish(plan)
                return EXIT_INCOMPLETE
            step = min(step + 1, len(cooldowns_s) - 1)
        wait = cooldowns_s[min(step, len(cooldowns_s) - 1)] if cooldowns_s else 0.0
        log(f"next pass in {wait:.0f}s (create {plan.run_dir / 'STOP'} to stop)")
        if not await _cooldown(plan, lock, wait):
            return EXIT_STOPPED


def supervise(plan: RunPlan, log: Callable[[str], None] = print, *,
              cooldowns_s: tuple[float, ...] = DEFAULT_COOLDOWNS_S, max_idle_passes: int = 2,
              on_finish: Callable[[RunPlan], None] | None = None, stay_awake: bool = True) -> int:
    """Run passes until done. Raises :class:`runner.LockHeld` if the run is busy."""
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "STOP").exists():
        log(f"STOP file present: {run_dir / 'STOP'}. Delete it to run.")
        return EXIT_STOPPED
    lock = RunLock(run_dir, plan.timings.lock_stale_s).acquire()
    runlog = RunLog(run_dir / "logs" / f"supervise-{new_invocation_id()}.log", log)
    try:
        with keep_awake(stay_awake):
            return asyncio.run(_supervise(plan, runlog, lock, cooldowns_s, max_idle_passes, on_finish))
    finally:
        runlog.close()
        lock.release()


# ---------------------------------------------------------------------------------------------
# Status


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_status(run_dir: Path, stale_s: float = 120.0) -> dict[str, Any]:
    """Read-only snapshot from run.json, results, ledger, progress and lock. Never repairs files."""
    meta = _read_json(run_dir / "run.json")
    if meta is None:
        raise FileNotFoundError(f"run not found: {run_dir}")
    items = meta.get("items", [])
    warnings: list[str] = []
    results = load_results(run_dir / "results.jsonl", warn=warnings.append)
    routes: dict[str, Any] = {}
    for key in meta.get("routes", {}):
        counts = Counter(results.get((key, u), {}).get("status", "pending") for u in items)
        final = sum(counts[s] for s in FINAL_STATUSES)
        routes[key] = {"total": len(items), "final": final, "remaining": len(items) - final, **dict(counts)}
    progress = _read_json(run_dir / "progress.json") or {}
    lock_path = run_dir / "lock"
    lock = read_lock(lock_path) if lock_path.exists() else None
    live = lock_path.exists() and not lock_is_stale(lock_path, lock, stale_s)
    if live:
        state = progress.get("state", "running")
        for key, live_route in (progress.get("routes") or {}).items():
            if key in routes:
                routes[key].update({k: live_route.get(k) for k in ("state", "rate_per_min", "eta_s", "last_error")})
    elif lock_path.exists():
        state = "stopped (stale lock)"
    else:
        state = "stopped (no live lock)"
    invocations = meta.get("invocations", [])
    last = invocations[-1] if invocations else {}
    return {
        "run_id": meta.get("run_id", run_dir.name), "state": state, "stop_file": (run_dir / "STOP").exists(),
        "lock": {k: lock.get(k) for k in ("pid", "host", "started_at", "refreshed_at")} if lock else None,
        "pauses": progress.get("pauses", {}) if live else {},
        "progress_updated_at": progress.get("updated_at"),
        "progress_age_s": round(time.time() - progress["updated_ts"]) if progress.get("updated_ts") else None,
        "items": len(items), "routes": routes,
        "budget": {"limit_audio_hours": meta.get("max_audio_hours"), **ledger_summary(run_dir)},
        "invocations": len(invocations),
        "last_invocation": {k: last.get(k) for k in ("id", "started_at", "finished_at", "exit_reason")},
        "eta_s": progress.get("eta_s") if live else None,
        "warnings": warnings, "checked_at": utc_now(),
    }


def render_status(st: dict[str, Any]) -> str:
    lines = [f"run {st['run_id']}: {st['state']}" + ("  [STOP file present]" if st["stop_file"] else "")]
    if st["lock"]:
        lk = st["lock"]
        lines.append(f"lock: pid {lk['pid']} on {lk['host']}, refreshed {lk['refreshed_at']}")
    for reason, detail in st["pauses"].items():
        lines.append(f"paused ({reason}): {detail}")
    if st["progress_updated_at"]:
        lines.append(f"progress.json updated {st['progress_updated_at']} ({st['progress_age_s']}s ago)")
    b = st["budget"]
    limit = "none" if b["limit_audio_hours"] is None else f"{b['limit_audio_hours']} h"
    lines.append(f"budget: {b['audio_hours']:.4f} audio h used of {limit}; attempts {b['attempts']}; "
                 f"unsettled {b['unsettled']} ({b['unsettled_audio_hours']:.4f} h)")
    lines.append(f"invocations: {st['invocations']}; last: {st['last_invocation']}")
    if st["eta_s"]:
        lines.append(f"eta: {st['eta_s'] / 3600:.1f} h")
    lines.append("")
    lines.append(f"{'route':34} {'final':>6} {'left':>6} {'ok':>6} {'empty':>5} {'fail':>5} {'final_f':>7} "
                 f"{'skip':>5} {'state':>8} {'/min':>6}")
    for key, r in st["routes"].items():
        rate = r.get("rate_per_min")
        lines.append(f"{key:34} {r['final']:6} {r['remaining']:6} {r.get('ok', 0):6} {r.get('empty', 0):5} "
                     f"{r.get('failed', 0):5} {r.get('failed_final', 0):7} {r.get('skipped', 0):5} "
                     f"{r.get('state') or '-':>8} {rate if rate is not None else '-':>6}")
        if r.get("last_error"):
            lines.append(f"{'':4}last error: {r['last_error'][:160]}")
    for w in st["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
