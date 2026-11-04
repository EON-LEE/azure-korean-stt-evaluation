"""Run routes over manifest items with caching, retries, an audio-hour budget and resume.

Layout::

    runs/cache/<route>/<sha256>.json   shared transcript cache (raw responses, local only)
    runs/<run_id>/run.json             provenance: routes, signatures, items, invocations
    runs/<run_id>/results.jsonl        append-only; the last record per (route, utt_id) wins
    runs/<run_id>/ledger.jsonl         reserve / settle / release record per send attempt
    runs/<run_id>/progress.json        live state, rewritten every ``Timings.progress_s``
    runs/<run_id>/lock                 owner of the run directory
    runs/<run_id>/logs/<id>.log        log lines (secrets redacted)
    runs/<run_id>/STOP                 created by the user: finish in-flight requests, then stop

A resumed run skips items whose last status is final (``ok``, ``empty``, ``failed_final``).
The ledger makes the audio-hour budget and the per-item attempt count cumulative across
invocations. Errors that did not reach the model (throttling, network, login, missing
deployment) release their reservation and do not count as attempts.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .adapters import create_adapter
from .audio import sha256_file
from .auth import COGNITIVE_SCOPE, redact, refresh_token
from .config import ModelConfig
from .data import dataset_revisions
from .normalize_ko import NORMALIZER_VERSION
from .types import AdapterError, AuthError, ManifestEntry, NetworkError, PermanentError, Transcription, \
    TransientError
from .util import append_jsonl, dumps, git_info, package_versions, read_jsonl, seed_order, sha256_json, utc_now, \
    write_json

DONE_STATUSES = {"ok", "empty"}
FINAL_STATUSES = {"ok", "empty", "failed_final"}
THROTTLE_CODES = {"rate_limit_exceeded", "too_many_requests", "TooManyRequests"}
AUTH_CODES = {"AuthenticationFailure", "Unauthorized", "invalid_api_key"}
ROUTE_CODES = {"DeploymentNotFound", "Forbidden", "NotFound", "PermissionDenied", "model_not_found"}
# Outcomes that never reached the model: the reservation is released and no attempt is counted.
RELEASED_KINDS = {"throttle", "network", "auth", "unauthorized", "route"}
DEFAULT_SEED = "kstt-1"
DEFAULT_MAX_ATTEMPTS = 4


class RunError(RuntimeError):
    """The run cannot start or resume (e.g. configuration changed under an existing run id)."""


class LockHeld(RunError):
    """Another live process owns the run directory."""


class LockLost(RunError):
    """The lock file no longer belongs to this process."""


@dataclass
class Timings:
    """Intervals and thresholds. Tests shrink them; defaults follow plan §14.5."""

    heartbeat_s: float = 5.0
    sleep_gap_s: float = 30.0
    loop_lag_s: float = 0.2
    send_delay_s: float = 0.5
    progress_s: float = 30.0
    network_window_s: float = 60.0
    network_errors: int = 5
    network_resources: int = 2
    network_probe_s: float = 60.0
    network_probe_successes: int = 2
    local_network_retries: int = 3
    token_retry_s: float = 300.0
    route_errors: int = 3
    route_cooldown_s: float = 900.0
    route_cooldowns: int = 3
    throttle_recover_s: float = 300.0
    max_throttle_waits: int = 20
    disk_min_bytes: int = 2 * 1024 ** 3
    lock_refresh_s: float = 10.0
    lock_stale_s: float = 120.0


@dataclass
class Probes:
    """Checks used while paused. ``None`` selects the real implementation."""

    network: Callable[[], Awaitable[bool]] | None = None
    token: Callable[[], Awaitable[bool]] | None = None
    disk_free: Callable[[Path], int] | None = None
    wall_clock: Callable[[], float] = time.time


@dataclass
class RunPlan:
    run_id: str
    runs_dir: Path
    data_dir: Path
    subset: str
    routes: list[ModelConfig]
    entries: list[ManifestEntry]
    max_audio_hours: float | None
    experiment: str = "baseline"
    parallel_routes: int = 0
    use_cache: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    seed: str = DEFAULT_SEED
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    resource_limits: dict[str, int] = field(default_factory=dict)
    skip_routes: set[str] = field(default_factory=set)
    timings: Timings = field(default_factory=Timings)
    probes: Probes = field(default_factory=Probes)
    invocation: str | None = None

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def cache_dir(self) -> Path:
        return self.runs_dir / "cache"


# ---------------------------------------------------------------------------------------------
# Item selection and order


def select_entries(entries: list[ManifestEntry], seed: str = DEFAULT_SEED, limit: int | None = None,
                   per_split: int | None = None) -> list[ManifestEntry]:
    """Pick items in seed order and return them in manifest order.

    ``limit`` keeps the first N items of the seed order. ``per_split`` keeps the first N items
    of each split. With the same seed, a smaller selection is a subset of a larger one.
    """
    if limit is not None and per_split is not None:
        raise RunError("use either limit or per_split, not both")
    if limit is None and per_split is None:
        return list(entries)
    order = seed_order(seed, [e.utt_id for e in entries])
    by_id = {e.utt_id: e for e in entries}
    keep: set[str] = set()
    if limit is not None:
        keep = set(order[:max(0, limit)])
    else:
        taken: Counter = Counter()
        for u in order:
            split = by_id[u].split
            if taken[split] < per_split:
                taken[split] += 1
                keep.add(u)
    return [e for e in entries if e.utt_id in keep]


def items_hash(entries: list[ManifestEntry]) -> str:
    return sha256_json([e.utt_id for e in entries])


def verify_audio(entries: list[ManifestEntry], data_dir: Path) -> list[str]:
    """Return problems (missing file or sha256 mismatch). An empty list means all audio matches."""
    problems = []
    for e in entries:
        path = data_dir / e.audio_path
        if not path.exists():
            problems.append(f"{e.utt_id}: missing {e.audio_path}")
        elif sha256_file(path) != e.audio_sha256:
            problems.append(f"{e.utt_id}: sha256 mismatch for {e.audio_path}")
    return problems


def new_invocation_id() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------------------------
# Persistent state


class Cache:
    def __init__(self, root: Path):
        self.root = root

    def path(self, route: str, signature: dict, utt_id: str, audio_sha256: str) -> Path:
        key = sha256_json({"signature": signature, "utt_id": utt_id, "audio_sha256": audio_sha256})
        return self.root / route / f"{key}.json"

    @staticmethod
    def get(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def put(path: Path, record: dict) -> None:
        write_json(path, record)


def load_results(path: Path, warn: Callable[[str], None] | None = None,
                 repair: bool = False) -> dict[tuple[str, str], dict]:
    """Last record per (route, utt_id). Only the lock owner may ``repair`` a damaged last line."""
    records: dict[tuple[str, str], dict] = {}
    for rec in read_jsonl(path, repair=repair, warn=warn):
        records[(rec["route"], rec["utt_id"])] = rec
    return records


class Ledger:
    """Append-only record of send attempts.

    ``reserve`` is written before audio is sent, ``settle`` after the service answered or the
    attempt failed in a way that counts, ``release`` when the attempt did not reach the model.
    Used audio = reserved - released. A reservation that is neither settled nor released (the
    process died mid-request) stays counted as used and as an attempt.
    """

    def __init__(self, path: Path, invocation: str, warn: Callable[[str], None] | None = None,
                 repair: bool = True):
        self.path = path
        self.invocation = invocation
        self._seq = 0
        self.reserved_s = 0.0
        self.released_s = 0.0
        self.attempts: Counter = Counter()
        self.permanent: Counter = Counter()
        self.outcomes: Counter = Counter()
        self._info: dict[str, tuple[str, str, float]] = {}
        self.open: dict[str, float] = {}
        for rec in read_jsonl(path, repair=repair, warn=warn):
            self._apply(rec)
        self.unsettled_previous = len(self.open)
        self.unsettled_previous_s = sum(self.open.values())
        self.used_at_start_s = self.used_s

    @property
    def used_s(self) -> float:
        return self.reserved_s - self.released_s

    def _apply(self, rec: dict[str, Any]) -> None:
        op, rid = rec.get("op"), rec.get("id")
        if op == "reserve":
            key = (rec["route"], rec["utt_id"])
            audio_s = float(rec["audio_s"])
            self._info[rid] = (key[0], key[1], audio_s)
            self.reserved_s += audio_s
            self.attempts[key] += 1
            self.open[rid] = audio_s
        elif op in ("settle", "release") and rid in self._info:
            route, utt, audio_s = self._info[rid]
            self.open.pop(rid, None)
            if op == "settle":
                self.outcomes[rec.get("outcome")] += 1
                if rec.get("outcome") == "permanent":
                    self.permanent[(route, utt)] += 1
            else:
                self.released_s += audio_s
                self.attempts[(route, utt)] -= 1

    def _write(self, rec: dict[str, Any]) -> None:
        append_jsonl(self.path, rec)
        self._apply(rec)

    def reserve(self, route: str, utt_id: str, audio_s: float, limit_s: float | None) -> str | None:
        if limit_s is not None and self.used_s + audio_s > limit_s + 1e-9:
            return None
        self._seq += 1
        rid = f"{self.invocation}:{self._seq}"
        self._write({"op": "reserve", "id": rid, "route": route, "utt_id": utt_id, "audio_s": round(audio_s, 6),
                     "at": utc_now()})
        return rid

    def settle(self, rid: str, outcome: str) -> None:
        self._write({"op": "settle", "id": rid, "outcome": outcome, "at": utc_now()})

    def release(self, rid: str, reason: str) -> None:
        self._write({"op": "release", "id": rid, "reason": reason, "at": utc_now()})


def ledger_summary(run_dir: Path) -> dict[str, Any]:
    """Read-only ledger totals for status and reports."""
    led = Ledger(run_dir / "ledger.jsonl", "read-only", repair=False)
    return {"audio_hours": led.used_s / 3600, "unsettled": len(led.open),
            "unsettled_audio_hours": sum(led.open.values()) / 3600,
            "attempts": sum(led.attempts.values()), "outcomes": dict(led.outcomes)}


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: the process exists
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def lock_is_stale(path: Path, info: dict[str, Any] | None, stale_s: float, now: float | None = None) -> bool:
    """Stale when its owner on this host is gone, or when it was not refreshed for ``stale_s`` and its
    owner is gone or on another host (whose PID cannot be checked)."""
    now = time.time() if now is None else now
    if info is None:
        try:
            return now - path.stat().st_mtime > stale_s
        except OSError:
            return True
    same_host = info.get("host") == socket.gethostname()
    if same_host and not pid_alive(int(info.get("pid", -1))):
        return True  # a crashed run on this machine; a restart need not wait for stale_s
    if now - float(info.get("refreshed_ts", 0)) <= stale_s:
        return False
    return not same_host


class RunLock:
    def __init__(self, run_dir: Path, stale_s: float = 120.0):
        self.path = run_dir / "lock"
        self.stale_s = stale_s
        self.token = uuid.uuid4().hex
        self.started_at = utc_now()
        self.held = False

    def _info(self) -> dict[str, Any]:
        return {"pid": os.getpid(), "host": socket.gethostname(), "token": self.token,
                "started_at": self.started_at, "refreshed_at": utc_now(), "refreshed_ts": time.time()}

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = read_lock(self.path)
                if lock_is_stale(self.path, info, self.stale_s):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                who = f"pid {info.get('pid')} on {info.get('host')} since {info.get('started_at')}" if info else "?"
                raise LockHeld(f"{self.path.parent.name} is in use by {who}. If no kstt process is running, "
                               f"delete {self.path} and retry.")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(dumps(self._info()))
                f.flush()
                os.fsync(f.fileno())
            self.held = True
            return self
        raise LockHeld(f"could not acquire {self.path}")

    def refresh(self) -> None:
        info = read_lock(self.path)
        if info is None or info.get("token") != self.token:
            self.held = False
            raise LockLost(f"{self.path} was taken over by another process")
        try:
            write_json(self.path, self._info())
        except OSError:
            pass  # a reader holds the file open; the next refresh retries

    def release(self) -> None:
        if not self.held:
            return
        info = read_lock(self.path)
        if info and info.get("token") == self.token:
            try:
                self.path.unlink()
            except OSError:
                pass
        self.held = False


class RunLog:
    """Log sink that also appends redacted lines to ``logs/<invocation>.log``."""

    def __init__(self, path: Path, sink: Callable[[str], None]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.sink = sink
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        msg = redact(msg)
        if not self._f.closed:
            self._f.write(f"{utc_now()} {msg}\n")
            self._f.flush()
        self.sink(msg)

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------------------------
# Shared run control: pauses, stop, route cooldowns, per-resource limits


def backoff_s(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base * 2 ** (attempt - 1)) * (0.75 + random.random() * 0.5)


class Control:
    def __init__(self, timings: Timings, probes: Probes, log: Callable[[str], None]):
        self.t = timings
        self.probes = probes
        self.log = log
        self.cond = asyncio.Condition()
        self.stop_event = asyncio.Event()
        self.pauses: dict[str, str] = {}
        self.sleep_epoch = 0
        self.lag_epoch = 0
        self.max_lag_s = 0.0
        self.network_events: deque[tuple[float, str]] = deque()
        self.network_epoch = 0
        self._watchers: dict[str, asyncio.Task] = {}
        self._token_check: asyncio.Task | None = None

    @property
    def stopping(self) -> bool:
        return self.stop_event.is_set()

    @property
    def state(self) -> str:
        if self.stopping:
            return "stopping"
        return next(iter(self.pauses), "running")

    async def notify(self) -> None:
        async with self.cond:
            self.cond.notify_all()

    async def wait_for(self, predicate: Callable[[], bool]) -> bool:
        """Wait until ``predicate()`` or stop. Returns False when stopping."""
        async with self.cond:
            await self.cond.wait_for(lambda: self.stopping or predicate())
        return not self.stopping

    async def wait_open(self) -> bool:
        return await self.wait_for(lambda: not self.pauses)

    async def sleep(self, seconds: float) -> bool:
        """Sleep unless stopped first. Returns False when stopping."""
        if seconds > 0 and not self.stopping:
            try:
                await asyncio.wait_for(self.stop_event.wait(), seconds)
            except asyncio.TimeoutError:
                pass
        return not self.stopping

    async def stop(self, why: str) -> None:
        if not self.stopping:
            self.log(f"stopping: {why}; waiting for in-flight requests")
            self.stop_event.set()
        await self.notify()

    async def pause(self, reason: str, detail: str, watcher: Callable[[], Awaitable[None]] | None = None) -> None:
        if reason in self.pauses:
            return
        self.pauses[reason] = detail
        self.log(f"paused ({reason}): {detail}")
        if watcher is not None:
            self._watchers[reason] = asyncio.create_task(watcher())
        await self.notify()

    async def resume(self, reason: str) -> None:
        if self.pauses.pop(reason, None) is not None:
            self.log(f"resumed ({reason})")
            if reason == "paused_network":
                self.network_events.clear()
                self.network_epoch += 1
        await self.notify()

    async def network_error(self, resource: str, detail: str) -> bool:
        """Record a pre-send connection failure. Returns True when the run is paused for the network."""
        now = time.monotonic()
        self.network_events.append((now, resource))
        while self.network_events and now - self.network_events[0][0] > self.t.network_window_s:
            self.network_events.popleft()
        if "paused_network" in self.pauses:
            return True
        resources = {r for _, r in self.network_events}
        if len(self.network_events) >= self.t.network_errors and len(resources) >= self.t.network_resources:
            await self.pause("paused_network", f"{len(self.network_events)} connection failures on "
                                               f"{len(resources)} resources; last: {detail}", self._watch_network)
            return True
        return False

    async def _watch_network(self) -> None:
        ok = 0
        while await self.sleep(self.t.network_probe_s):
            try:
                reachable = await self.probes.network()
            except Exception:  # noqa: BLE001 - a failed probe means still offline
                reachable = False
            ok = ok + 1 if reachable else 0
            if ok >= self.t.network_probe_successes:
                await self.resume("paused_network")
                return

    async def need_login(self, detail: str) -> None:
        await self.pause("needs_login", f"{detail}. Run `az login`; the run resumes automatically.",
                         self._watch_token)

    async def _probe_token(self) -> bool:
        try:
            return bool(await self.probes.token())
        except Exception:  # noqa: BLE001
            return False

    async def _watch_token(self) -> None:
        while await self.sleep(self.t.token_retry_s):
            if await self._probe_token():
                await self.resume("needs_login")
                return

    async def token_ok(self, detail: str) -> bool:
        """After an HTTP 401: refresh the token once (shared by concurrent callers).

        If no token can be obtained the run pauses for login and this returns False.
        """
        if "needs_login" in self.pauses:
            return False
        if self._token_check is None or self._token_check.done():
            self._token_check = asyncio.ensure_future(self._probe_token())
        ok = await asyncio.shield(self._token_check)
        if not ok:
            await self.need_login(detail)
        return ok

    async def close(self) -> None:
        for task in list(self._watchers.values()) + ([self._token_check] if self._token_check else []):
            task.cancel()
        for task in list(self._watchers.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


class RouteGuard:
    """Cooldown for route-level errors (403, 404, missing deployment, 401 with a valid token)."""

    def __init__(self, key: str, ctl: Control):
        self.key = key
        self.ctl = ctl
        self.state = "active"  # active | cooldown | probe | blocked
        self.streak = 0
        self.cooldowns = 0
        self.probe_taken = False
        self.last_error: str | None = None
        self._task: asyncio.Task | None = None

    async def enter(self) -> str | None:
        """Wait until the route may send. Returns "active", "probe" or None (blocked or stopping)."""
        def ready() -> bool:
            return self.state in ("active", "blocked") or (self.state == "probe" and not self.probe_taken)

        if not await self.ctl.wait_for(ready) or self.state == "blocked":
            return None
        if self.state == "probe":
            self.probe_taken = True
            return "probe"
        return "active"

    async def ok(self) -> None:
        self.streak = 0
        if self.state == "probe":
            self.state = "active"
            self.cooldowns = 0
            self.ctl.log(f"[{self.key}] route reachable again")
            await self.ctl.notify()

    async def inconclusive(self, mode: str | None) -> None:
        if mode == "probe" and self.state == "probe":
            self.probe_taken = False
            await self.ctl.notify()

    async def route_error(self, detail: str) -> None:
        t = self.ctl.t
        self.last_error = detail
        self.streak += 1
        if self.state == "probe" or (self.state == "active" and self.streak >= t.route_errors):
            if self.cooldowns >= t.route_cooldowns:
                self.state = "blocked"
                self.ctl.log(f"[{self.key}] route_blocked after {self.cooldowns} cooldown(s): {detail}")
            else:
                self.cooldowns += 1
                self.state = "cooldown"
                self.probe_taken = False
                self.ctl.log(f"[{self.key}] {self.streak} route error(s); cooldown {t.route_cooldown_s:.0f}s "
                             f"({self.cooldowns}/{t.route_cooldowns}): {detail}")
                self._task = asyncio.create_task(self._cooldown())
        await self.ctl.notify()

    async def _cooldown(self) -> None:
        if await self.ctl.sleep(self.ctl.t.route_cooldown_s):
            self.state = "probe"
            self.probe_taken = False
            await self.ctl.notify()

    def close(self) -> None:
        if self._task:
            self._task.cancel()


class ResourceLimiter:
    """Concurrent requests per Azure resource. Halves on 429 and adds 1 per quiet interval."""

    def __init__(self, name: str, cap: int, ctl: Control):
        self.name = name
        self.cap = max(1, cap)
        self.limit = self.cap
        self.active = 0
        self.throttles = 0
        self.ctl = ctl
        self.changed_at = time.monotonic()

    def _recover(self) -> None:
        now = time.monotonic()
        if self.limit < self.cap and now - self.changed_at >= self.ctl.t.throttle_recover_s:
            self.limit += 1
            self.changed_at = now
            self.ctl.log(f"[{self.name}] concurrency limit restored to {self.limit}/{self.cap}")

    async def acquire(self) -> bool:
        def ready() -> bool:
            self._recover()
            return self.active < self.limit

        if not await self.ctl.wait_for(ready):
            return False
        self.active += 1
        return True

    async def release(self) -> None:
        self.active -= 1
        await self.ctl.notify()

    def throttled(self) -> None:
        self.throttles += 1
        new = max(1, self.limit // 2)
        if new < self.limit:
            self.ctl.log(f"[{self.name}] HTTP 429: concurrency limit {self.limit} -> {new}")
        self.limit = new
        self.changed_at = time.monotonic()


def classify(exc: BaseException) -> str:
    """Map an adapter exception to a handling rule (plan §14.5)."""
    if isinstance(exc, AuthError):
        return "auth"
    status = getattr(exc, "status", None)
    code = getattr(exc, "code", None)
    if isinstance(exc, PermanentError):
        if status == 401 or code in AUTH_CODES:
            return "unauthorized"
        if status in (403, 404) or code in ROUTE_CODES:
            return "route"
        return "permanent"
    if isinstance(exc, TransientError):
        if status == 429 or code in THROTTLE_CODES:
            return "throttle"
        if isinstance(exc, NetworkError):
            return "network"
        return "ambiguous" if exc.ambiguous else "server"
    if isinstance(exc, AdapterError):
        return "permanent"
    return "unexpected"


def _error_dict(exc: BaseException, kind: str) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "kind": kind,
        "status": getattr(exc, "status", None),
        "code": getattr(exc, "code", None),
        "ambiguous": getattr(exc, "ambiguous", False),
        "message": redact(str(exc))[:500],
    }


def _record(route: str, entry: ManifestEntry, status: str, *, tr: Transcription | None = None, cached: bool = False,
            attempts: int = 0, throttled: int = 0, error: dict | None = None, reason: str | None = None,
            cache_file: str | None = None, invocation: str | None = None, latency_invalid: bool = False,
            latency_flags: list[str] | None = None) -> dict[str, Any]:
    return {
        "route": route,
        "utt_id": entry.utt_id,
        "split": entry.split,
        "duration_s": entry.duration_s,
        "status": status,
        "text": tr.text if tr else None,
        "latency": tr.latency if tr else {},
        "partials": tr.partials if tr else 0,
        "usage": tr.usage if tr else None,
        "meta": tr.meta if tr else {},
        "cached": cached,
        "attempts": attempts,
        "throttled": throttled,
        "error": error,
        "reason": reason,
        "cache_file": cache_file,
        "latency_invalid": latency_invalid,
        "latency_flags": latency_flags or [],
        "invocation": invocation,
        "finished_at": utc_now(),
    }


def _sent_audio_s(adapter: Any, duration_s: float) -> float:
    fn = getattr(adapter, "sent_audio_s", None)
    return float(fn(duration_s)) if callable(fn) else duration_s


def resource_of(cfg: ModelConfig) -> str:
    return str(cfg.params.get("resource") or cfg.region or cfg.key)


# ---------------------------------------------------------------------------------------------
# Route execution


class RouteRunner:
    def __init__(self, cfg: ModelConfig, plan: RunPlan, ctx: "RunContext"):
        self.cfg = cfg
        self.plan = plan
        self.ctx = ctx
        self.ctl = ctx.ctl
        self.adapter = create_adapter(cfg)
        self.signature = self.adapter.signature()
        params = cfg.params
        self.max_retries = int(params.get("max_retries", 4))
        self.max_attempts = int(params.get("max_attempts", plan.max_attempts))
        self.retry_ambiguous = bool(params.get("retry_ambiguous", False))
        self.concurrency = max(1, int(params.get("concurrency", 4)))
        self.min_interval = float(params.get("min_interval_s", 0))
        self._next_start = 0.0
        self.resource = resource_of(cfg)
        self.guard = RouteGuard(cfg.key, self.ctl)
        self.counts: Counter = Counter()
        self.finalized = 0
        self.last_error: str | None = None

    async def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        delay = self._next_start - time.monotonic()
        self._next_start = max(time.monotonic(), self._next_start) + self.min_interval
        if delay > 0:
            await asyncio.sleep(delay)

    def _latency_marks(self, tr: Transcription, entry: ManifestEntry, epochs: tuple[int, int]) -> tuple[bool, list]:
        invalid = self.ctl.sleep_epoch != epochs[0]
        flags = []
        if self.ctl.lag_epoch != epochs[1]:
            flags.append("loop_lag")
        send = (tr.latency or {}).get("audio_send_s")
        if isinstance(send, (int, float)) and send - entry.duration_s > self.ctl.t.send_delay_s:
            flags.append("send_delay")
        return invalid, flags

    async def process(self, entry: ManifestEntry) -> dict[str, Any] | None:
        """Return the record to append, or None to leave the item for a later pass."""
        key = self.cfg.key
        led = self.ctx.ledger
        inv = self.ctx.invocation
        item = (key, entry.utt_id)
        cpath = self.ctx.cache.path(key, self.signature, entry.utt_id, entry.audio_sha256)
        crel = cpath.relative_to(self.plan.runs_dir).as_posix()
        if self.plan.use_cache:
            cached = self.ctx.cache.get(cpath)
            if cached:
                tr = Transcription.from_dict(cached["transcription"])
                status = "ok" if tr.text.strip() else "empty"
                return _record(key, entry, status, tr=tr, cached=True, cache_file=crel, invocation=inv,
                               attempts=led.attempts[item])
        if led.attempts[item] >= self.max_attempts:
            return _record(key, entry, "failed_final", attempts=led.attempts[item], reason="attempt_cap",
                           invocation=inv)

        audio = entry.to_audio_input(self.plan.data_dir)
        audio_s = _sent_audio_s(self.adapter, entry.duration_s)
        server_retries = network_errors = throttled = 0
        net_epoch = self.ctl.network_epoch
        while True:
            if not await self.ctl.wait_open():
                return None
            mode = await self.guard.enter()
            if mode is None:
                return None
            limiter = self.ctx.limiters[self.resource]
            if not await limiter.acquire():
                await self.guard.inconclusive(mode)
                return None
            exc: Exception | None = None
            tr: Transcription | None = None
            rid: str | None = None
            try:
                rid = led.reserve(key, entry.utt_id, audio_s, self.ctx.budget_limit_s)
                if rid is not None:
                    await self._pace()
                    epochs = (self.ctl.sleep_epoch, self.ctl.lag_epoch)
                    tr = await self.adapter.transcribe(audio)
            except Exception as e:  # noqa: BLE001 - classified below
                exc = e
            finally:
                await limiter.release()
            if rid is None:
                await self.guard.inconclusive(mode)
                self.ctx.budget_exhausted = True
                return _record(key, entry, "skipped", attempts=led.attempts[item], throttled=throttled,
                               reason="budget_exhausted", invocation=inv)

            if exc is None:
                led.settle(rid, "ok")
                await self.guard.ok()
                invalid, flags = self._latency_marks(tr, entry, epochs)
                self.ctx.cache.put(cpath, {"route": key, "signature": self.signature, "utt_id": entry.utt_id,
                                           "audio_sha256": entry.audio_sha256, "created_at": utc_now(),
                                           "transcription": tr.to_dict()})
                status = "ok" if tr.text.strip() else "empty"
                return _record(key, entry, status, tr=tr, attempts=led.attempts[item], throttled=throttled,
                               cache_file=crel, invocation=inv, latency_invalid=invalid, latency_flags=flags)

            kind = classify(exc)
            err = _error_dict(exc, kind)
            detail = f"{entry.utt_id} {kind}: {err['message'][:200]}"
            self.last_error = detail
            if kind in RELEASED_KINDS:
                led.release(rid, kind)
            else:
                led.settle(rid, kind)
            attempts = led.attempts[item]

            if kind == "throttle":
                throttled += 1
                limiter.throttled()
                await self.guard.ok()
                if throttled > self.ctl.t.max_throttle_waits:
                    return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=err,
                                   reason="throttled_too_often", invocation=inv)
                delay = exc.retry_after_s if getattr(exc, "retry_after_s", None) is not None else backoff_s(throttled)
                self.ctx.log(f"[{key}] {entry.utt_id} throttled; retry in {delay:.1f}s")
                if not await self.ctl.sleep(delay):
                    return None
                continue
            if kind == "network":
                await self.guard.inconclusive(mode)
                if await self.ctl.network_error(self.resource, detail):
                    continue
                if self.ctl.network_epoch != net_epoch:
                    net_epoch, network_errors = self.ctl.network_epoch, 0
                network_errors += 1
                if network_errors > self.ctl.t.local_network_retries:
                    return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=err,
                                   reason="network", invocation=inv)
                if not await self.ctl.sleep(backoff_s(network_errors)):
                    return None
                continue
            if kind == "auth":
                await self.guard.inconclusive(mode)
                await self.ctl.need_login(err["message"][:200])
                continue
            if kind == "unauthorized":
                if not await self.ctl.token_ok(err["message"][:200]):
                    await self.guard.inconclusive(mode)
                    continue
                kind = "route"
            if kind == "route":
                await self.guard.route_error(detail)
                continue

            await self.guard.ok()
            if kind == "permanent":
                final = led.permanent[item] >= 2 or attempts >= self.max_attempts
                return _record(key, entry, "failed_final" if final else "failed", attempts=attempts,
                               throttled=throttled, error=err, invocation=inv)
            if kind == "unexpected":
                err["unexpected"] = True
            if attempts >= self.max_attempts:
                return _record(key, entry, "failed_final", attempts=attempts, throttled=throttled, error=err,
                               reason="attempt_cap", invocation=inv)
            if kind == "server" or (kind == "ambiguous" and self.retry_ambiguous):
                if server_retries < self.max_retries:
                    server_retries += 1
                    delay = exc.retry_after_s if getattr(exc, "retry_after_s", None) is not None \
                        else backoff_s(server_retries)
                    self.ctx.log(f"[{key}] {entry.utt_id} {kind} ({err['status'] or err['code'] or '-'}); "
                                 f"retry in {delay:.1f}s")
                    if not await self.ctl.sleep(delay):
                        return None
                    continue
            return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=err, invocation=inv)

    def _write(self, entry: ManifestEntry, rec: dict[str, Any]) -> None:
        append_jsonl(self.ctx.results_path, rec)
        self.ctx.status[(self.cfg.key, entry.utt_id)] = rec["status"]
        self.counts[rec["status"]] += 1
        if rec["status"] in FINAL_STATUSES:
            self.finalized += 1
        lat = rec["latency"]
        main = lat.get("request_s", lat.get("final_after_audio_end_s"))
        extra = f" {main:.2f}s" if isinstance(main, (int, float)) else ""
        tag = " (cache)" if rec["cached"] else ""
        why = f" {rec['error']['message'][:120]}" if rec["error"] else (f" {rec['reason']}" if rec["reason"] else "")
        self.ctx.log(f"[{self.cfg.key}] {entry.utt_id} {rec['status']}{tag}{extra}{why}")

    async def run(self, todo: list[ManifestEntry]) -> dict[str, int]:
        queue = deque(todo)

        async def worker() -> None:
            while queue and not self.ctl.stopping:
                entry = queue.popleft()
                rec = await self.process(entry)
                if rec is not None:
                    self._write(entry, rec)

        try:
            await asyncio.gather(*(worker() for _ in range(min(self.concurrency, len(todo)))))
        finally:
            self.guard.close()
            await self.adapter.aclose()
        return dict(self.counts)


# ---------------------------------------------------------------------------------------------
# Run metadata


def run_metadata(plan: RunPlan, signatures: dict[str, dict]) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "subset": plan.subset,
        "experiment": plan.experiment,
        "items": [e.utt_id for e in plan.entries],
        "items_sha256": items_hash(plan.entries),
        "item_audio_hours": round(sum(e.duration_s for e in plan.entries) / 3600, 6),
        "routes": {
            cfg.key: {
                "adapter": cfg.adapter, "track": cfg.track, "region": cfg.region, "model_id": cfg.model_id,
                "resource": resource_of(cfg), "live_verified": cfg.live_verified,
                "price_usd_per_audio_hour": cfg.price_usd_per_audio_hour, "notes": cfg.notes,
                "signature": signatures[cfg.key],
            }
            for cfg in plan.routes
        },
        "normalizer": NORMALIZER_VERSION,
        "dataset_revisions": dataset_revisions(plan.entries),
        "git": git_info(),
        "packages": package_versions(),
        "max_audio_hours": plan.max_audio_hours,
        **plan.extra,
    }


def _merge_run_json(path: Path, new: dict[str, Any]) -> dict[str, Any]:
    """Create run.json, or check that a resumed run keeps the same items and route signatures.

    Creation-time provenance (git, packages) is kept; each invocation is appended separately.
    """
    if not path.exists():
        new["created_at"] = utc_now()
        return new
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("subset") != new["subset"] or old.get("items") != new["items"]:
        raise RunError(f"{path.parent.name}: items or subset differ from the existing run. Use a new --run-id.")
    for key, spec in new["routes"].items():
        prev = old.get("routes", {}).get(key)
        if prev and prev.get("signature") != spec["signature"]:
            raise RunError(f"{key}: configuration changed since this run started. Use a new --run-id.")
    merged = dict(old)
    merged["routes"] = {**old.get("routes", {}), **new["routes"]}
    merged.setdefault("items_sha256", new["items_sha256"])
    merged["max_audio_hours"] = new["max_audio_hours"]
    return merged


def plan_summary(plan: RunPlan) -> list[dict[str, Any]]:
    """Per-route work estimate without calling any service."""
    previous = load_results(plan.run_dir / "results.jsonl")
    cache = Cache(plan.cache_dir)
    rows = []
    for cfg in plan.routes:
        adapter = create_adapter(cfg)
        sig = adapter.signature()
        done = cached = 0
        call_s = 0.0
        for e in plan.entries:
            if previous.get((cfg.key, e.utt_id), {}).get("status") in FINAL_STATUSES:
                done += 1
            elif plan.use_cache and cache.path(cfg.key, sig, e.utt_id, e.audio_sha256).exists():
                cached += 1
            else:
                call_s += _sent_audio_s(adapter, e.duration_s)
        rows.append({"route": cfg.key, "adapter": cfg.adapter, "track": cfg.track, "resource": resource_of(cfg),
                     "items": len(plan.entries), "done": done, "cached": cached,
                     "to_call": len(plan.entries) - done - cached, "audio_hours_to_call": round(call_s / 3600, 4),
                     "concurrency": int(cfg.params.get("concurrency", 4)),
                     "min_interval_s": float(cfg.params.get("min_interval_s", 0))})
    return rows


# ---------------------------------------------------------------------------------------------
# Execute one pass


class RunContext:
    def __init__(self, plan: RunPlan, ctl: Control, ledger: Ledger, cache: Cache, invocation: str,
                 log: Callable[[str], None], status: dict[tuple[str, str], str]):
        self.plan = plan
        self.ctl = ctl
        self.ledger = ledger
        self.cache = cache
        self.invocation = invocation
        self.log = log
        self.status = status
        self.results_path = plan.run_dir / "results.jsonl"
        self.budget_limit_s = None if plan.max_audio_hours is None else plan.max_audio_hours * 3600
        self.budget_exhausted = False
        self.limiters: dict[str, ResourceLimiter] = {}


def _default_probes(plan: RunPlan) -> Probes:
    import httpx

    urls = []
    for cfg in plan.routes:
        ep = str(cfg.params.get("endpoint") or "")
        if ep.startswith("wss://"):
            ep = "https://" + ep[6:]
        if ep.startswith("https://") and ep not in urls:
            urls.append(ep)

    async def network() -> bool:
        async with httpx.AsyncClient(timeout=10) as client:
            for url in urls:
                try:
                    await client.get(url)
                    return True
                except httpx.HTTPError:
                    continue
        return False

    async def token() -> bool:
        try:
            await asyncio.to_thread(refresh_token, COGNITIVE_SCOPE)
            return True
        except AuthError:
            return False

    p = plan.probes
    return Probes(network=p.network or network, token=p.token or token,
                  disk_free=p.disk_free or (lambda path: shutil.disk_usage(path).free), wall_clock=p.wall_clock)


def _progress(plan: RunPlan, ctx: RunContext, runners: list[RouteRunner], started: float,
              meta_inv: dict[str, Any]) -> dict[str, Any]:
    elapsed = max(1e-9, time.monotonic() - started)
    routes = {}
    eta = 0.0
    for r in runners:
        counts = Counter(ctx.status.get((r.cfg.key, e.utt_id), "pending") for e in plan.entries)
        final = sum(counts[s] for s in FINAL_STATUSES)
        remaining = len(plan.entries) - final
        rate = r.finalized / elapsed
        route_eta = remaining / rate if rate > 0 else None
        if route_eta is not None:
            eta = max(eta, route_eta)
        routes[r.cfg.key] = {
            "total": len(plan.entries), "final": final, "remaining": remaining, **dict(counts),
            "state": "skipped" if r.cfg.key in plan.skip_routes else r.guard.state,
            "cooldowns": r.guard.cooldowns, "resource": r.resource,
            "rate_per_min": round(rate * 60, 3), "eta_s": None if route_eta is None else round(route_eta),
            "last_error": r.last_error or r.guard.last_error,
        }
    led = ctx.ledger
    return {
        "run_id": plan.run_id, "invocation": ctx.invocation, "pid": os.getpid(), "host": socket.gethostname(),
        "state": ctx.ctl.state, "pauses": dict(ctx.ctl.pauses),
        "updated_at": utc_now(), "updated_ts": time.time(), "started_at": meta_inv["started_at"],
        "elapsed_s": round(elapsed), "eta_s": round(eta) if eta else None, "routes": routes,
        "budget": {
            "limit_audio_hours": plan.max_audio_hours,
            "used_audio_hours": round(led.used_s / 3600, 6),
            "this_invocation_audio_hours": round((led.used_s - led.used_at_start_s) / 3600, 6),
            "unsettled_previous": led.unsettled_previous,
            "unsettled_previous_audio_hours": round(led.unsettled_previous_s / 3600, 6),
            "exhausted": ctx.budget_exhausted,
        },
        "resources": {n: {"limit": lim.limit, "cap": lim.cap, "active": lim.active, "throttles": lim.throttles}
                      for n, lim in ctx.limiters.items()},
        "clock": {"sleep_gaps": ctx.ctl.sleep_epoch, "loop_lag_events": ctx.ctl.lag_epoch,
                  "max_loop_lag_s": round(ctx.ctl.max_lag_s, 3)},
        "recent_network_errors": len(ctx.ctl.network_events),
    }


async def execute(plan: RunPlan, log: Callable[[str], None] = print, *, lock: RunLock | None = None) -> dict[str, Any]:
    """Run one pass over all non-final items. Safe to call again with the same plan to resume.

    Without ``lock`` the pass takes the run lock itself (raises :class:`LockHeld` if busy).
    """
    run_dir = plan.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    invocation = plan.invocation or new_invocation_id()
    own_lock = lock is None
    if own_lock:
        lock = RunLock(run_dir, plan.timings.lock_stale_s).acquire()
    runlog = log if isinstance(log, RunLog) else RunLog(run_dir / "logs" / f"{invocation}.log", log)
    try:
        return await _execute(plan, runlog, lock, own_lock, invocation)
    finally:
        if runlog is not log:
            runlog.close()
        if own_lock:
            lock.release()


async def _execute(plan: RunPlan, log: Callable[[str], None], lock: RunLock, own_lock: bool,
                   invocation: str) -> dict[str, Any]:
    run_dir = plan.run_dir
    t = plan.timings

    def warn(msg: str) -> None:
        log(f"warning: {msg}")

    previous = load_results(run_dir / "results.jsonl", warn=warn, repair=True)
    ledger = Ledger(run_dir / "ledger.jsonl", invocation, warn=warn)
    if ledger.unsettled_previous:
        log(f"{ledger.unsettled_previous} reservation(s) from earlier invocations were never settled; "
            f"counted as used ({ledger.unsettled_previous_s / 3600:.4f} audio h)")
    probes = _default_probes(plan)
    ctl = Control(t, probes, log)
    status = {k: v["status"] for k, v in previous.items()}
    ctx = RunContext(plan, ctl, ledger, Cache(plan.cache_dir), invocation, log, status)
    runners = [RouteRunner(cfg, plan, ctx) for cfg in plan.routes]
    caps: Counter = Counter()
    for r in runners:
        caps[r.resource] += r.concurrency
    for name, cap in caps.items():
        ctx.limiters[name] = ResourceLimiter(name, int(plan.resource_limits.get(name, cap)), ctl)

    run_json = run_dir / "run.json"
    meta = _merge_run_json(run_json, run_metadata(plan, {r.cfg.key: r.signature for r in runners}))
    inv: dict[str, Any] = {
        "id": invocation, "pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now(),
        "git": git_info(), "packages": package_versions(), "max_audio_hours": plan.max_audio_hours,
        "seed": plan.seed, "max_attempts": plan.max_attempts,
        "routes": [r.cfg.key for r in runners if r.cfg.key not in plan.skip_routes],
        "ledger_audio_hours_at_start": round(ledger.used_s / 3600, 6),
        "unsettled_previous": ledger.unsettled_previous,
    }
    meta.setdefault("invocations", []).append(inv)
    meta["last_started_at"] = inv["started_at"]
    write_json(run_json, meta)

    started = time.monotonic()
    progress_path = run_dir / "progress.json"
    exit_reason = "error"

    async def monitor() -> None:
        last_wall, last_mono = probes.wall_clock(), time.monotonic()
        next_progress = next_lock = last_mono
        while True:
            await asyncio.sleep(t.heartbeat_s)
            wall, mono = probes.wall_clock(), time.monotonic()
            if wall - last_wall - t.heartbeat_s > t.sleep_gap_s:
                ctl.sleep_epoch += 1
                log(f"clock gap of {wall - last_wall:.0f}s (sleep or suspend); "
                    "in-flight items are marked latency_invalid")
            else:
                lag = (mono - last_mono) - t.heartbeat_s
                if lag > t.loop_lag_s:
                    ctl.lag_epoch += 1
                    ctl.max_lag_s = max(ctl.max_lag_s, lag)
            last_wall, last_mono = wall, mono
            if (run_dir / "STOP").exists():
                await ctl.stop("STOP file found")
            try:
                free = probes.disk_free(run_dir)
            except OSError:
                free = None
            if free is not None and free < t.disk_min_bytes:
                await ctl.pause("disk_low", f"{free / 1024 ** 3:.2f} GB free in {run_dir}")
            elif "disk_low" in ctl.pauses:
                await ctl.resume("disk_low")
            if mono >= next_lock:
                try:
                    lock.refresh()
                except LockLost as exc:
                    log(f"error: {exc}")
                    await ctl.stop("lock lost")
                next_lock = mono + t.lock_refresh_s
            if mono >= next_progress:
                write_json(progress_path, _progress(plan, ctx, runners, started, inv))
                next_progress = mono + t.progress_s
            await ctl.notify()

    route_sem = asyncio.Semaphore(plan.parallel_routes) if plan.parallel_routes > 0 else None
    order = seed_order(plan.seed, [e.utt_id for e in plan.entries])
    by_id = {e.utt_id: e for e in plan.entries}

    async def run_route(runner: RouteRunner) -> tuple[str, dict[str, int]]:
        key = runner.cfg.key
        if key in plan.skip_routes:
            await runner.adapter.aclose()
            return key, {}
        todo = [by_id[u] for u in order if status.get((key, u)) not in FINAL_STATUSES]
        if not todo:
            await runner.adapter.aclose()
            return key, {}
        if route_sem is None:
            log(f"[{key}] start: {len(todo)} item(s)")
            return key, await runner.run(todo)
        async with route_sem:
            if ctl.stopping:
                await runner.adapter.aclose()
                return key, {}
            log(f"[{key}] start: {len(todo)} item(s)")
            return key, await runner.run(todo)

    if (run_dir / "STOP").exists():
        await ctl.stop("STOP file present at start; delete it to run")
    mon = asyncio.create_task(monitor())
    try:
        outcome = dict(await asyncio.gather(*(run_route(r) for r in runners)))
        exit_reason = "stopped" if ctl.stopping else ("budget_exhausted" if ctx.budget_exhausted else "pass_done")
    except BaseException as exc:
        exit_reason = f"error: {type(exc).__name__}"
        raise
    finally:
        mon.cancel()
        try:
            await mon
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        await ctl.close()
        remaining = {r.cfg.key: sum(status.get((r.cfg.key, e.utt_id)) not in FINAL_STATUSES for e in plan.entries)
                     for r in runners}
        blocked = sorted(r.cfg.key for r in runners if r.guard.state == "blocked")
        inv.update({
            "finished_at": utc_now(), "wall_s": round(time.monotonic() - started, 3), "exit_reason": exit_reason,
            "counts": {r.cfg.key: dict(r.counts) for r in runners if r.counts},
            "audio_hours": round((ledger.used_s - ledger.used_at_start_s) / 3600, 6),
            "blocked_routes": blocked, "remaining": remaining,
        })
        meta["last_finished_at"] = inv["finished_at"]
        meta["last_wall_s"] = inv["wall_s"]
        meta["last_budget_used_audio_hours"] = inv["audio_hours"]
        meta["ledger_audio_hours"] = round(ledger.used_s / 3600, 6)
        write_json(run_json, meta)
        write_json(progress_path, _progress(plan, ctx, runners, started, inv))

    return {
        "run_dir": str(run_dir), "invocation": invocation, "counts": outcome,
        "budget_used_audio_hours": (ledger.used_s - ledger.used_at_start_s) / 3600,
        "ledger_audio_hours": ledger.used_s / 3600,
        "stopped": ctl.stopping, "budget_exhausted": ctx.budget_exhausted,
        "blocked_routes": blocked, "remaining": remaining,
        "finalized": sum(r.finalized for r in runners),
        "unsettled_previous": ledger.unsettled_previous,
    }
