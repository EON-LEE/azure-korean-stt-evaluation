"""Run routes over manifest items with caching, retries, an audio-hour budget and resume.

Layout::

    runs/cache/<route>/<sha256>.json   shared transcript cache (raw responses, local only)
    runs/<run_id>/run.json             provenance: routes, signatures, items, versions
    runs/<run_id>/results.jsonl        append-only; the last record per (route, utt_id) wins

A resumed run skips items whose last status is ``ok`` or ``empty`` and retries the rest.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .adapters import create_adapter
from .auth import redact
from .config import ModelConfig
from .data import dataset_revisions
from .normalize_ko import NORMALIZER_VERSION
from .types import AdapterError, ManifestEntry, PermanentError, Transcription, TransientError
from .util import dumps, git_info, package_versions, sha256_json, utc_now, write_json

DONE_STATUSES = {"ok", "empty"}
THROTTLE_CODES = {"rate_limit_exceeded", "too_many_requests", "TooManyRequests"}
BREAKER_THRESHOLD = 3


class RunError(RuntimeError):
    """The run cannot start or resume (e.g. configuration changed under an existing run id)."""


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
    parallel_routes: int = 2
    use_cache: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def cache_dir(self) -> Path:
        return self.runs_dir / "cache"


class Budget:
    """Audio-hour budget shared by all routes. Every attempt, including retries, reserves its audio."""

    def __init__(self, max_hours: float | None):
        self.limit_s = None if max_hours is None else max_hours * 3600
        self.used_s = 0.0
        self._lock = asyncio.Lock()

    async def reserve(self, seconds: float) -> bool:
        async with self._lock:
            if self.limit_s is not None and self.used_s + seconds > self.limit_s:
                return False
            self.used_s += seconds
            return True


class Pacer:
    """Space request starts at least ``interval`` seconds apart (for low-RPM deployments)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self.interval


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


class ResultsLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    def load(self) -> dict[tuple[str, str], dict]:
        return load_results(self.path)

    async def append(self, record: dict) -> None:
        line = dumps(record) + "\n"
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)


def load_results(path: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records[(rec["route"], rec["utt_id"])] = rec
    return records


def backoff_s(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base * 2 ** (attempt - 1)) * (0.75 + random.random() * 0.5)


def _error_dict(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "status": getattr(exc, "status", None),
        "code": getattr(exc, "code", None),
        "ambiguous": getattr(exc, "ambiguous", False),
        "message": redact(str(exc))[:500],
    }


def _record(route: str, entry: ManifestEntry, status: str, *, tr: Transcription | None = None, cached: bool = False,
            attempts: int = 0, throttled: int = 0, error: dict | None = None, reason: str | None = None,
            cache_file: str | None = None) -> dict[str, Any]:
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
        "finished_at": utc_now(),
    }


class RouteRunner:
    def __init__(self, cfg: ModelConfig, plan: RunPlan, budget: Budget, cache: Cache, results: ResultsLog,
                 log: Callable[[str], None]):
        self.cfg = cfg
        self.plan = plan
        self.budget = budget
        self.cache = cache
        self.results = results
        self.log = log
        self.adapter = create_adapter(cfg)
        self.signature = self.adapter.signature()
        params = cfg.params
        self.max_retries = int(params.get("max_retries", 4))
        self.retry_ambiguous = bool(params.get("retry_ambiguous", False))
        self.sem = asyncio.Semaphore(max(1, int(params.get("concurrency", 4))))
        self.pacer = Pacer(float(params.get("min_interval_s", 0)))
        self.consecutive_permanent = 0
        self.disabled: str | None = None
        self.counts: dict[str, int] = {}

    async def process(self, entry: ManifestEntry) -> dict[str, Any]:
        key = self.cfg.key
        cpath = self.cache.path(key, self.signature, entry.utt_id, entry.audio_sha256)
        crel = cpath.relative_to(self.plan.runs_dir).as_posix()
        if self.plan.use_cache:
            cached = self.cache.get(cpath)
            if cached:
                tr = Transcription.from_dict(cached["transcription"])
                status = "ok" if tr.text.strip() else "empty"
                return _record(key, entry, status, tr=tr, cached=True, cache_file=crel)
        if self.disabled:
            return _record(key, entry, "skipped", reason=self.disabled)

        audio = entry.to_audio_input(self.plan.data_dir)
        attempts = throttled = 0
        while True:
            if not await self.budget.reserve(entry.duration_s):
                return _record(key, entry, "skipped", attempts=attempts, throttled=throttled,
                               reason="budget_exhausted")
            await self.pacer.wait()
            attempts += 1
            try:
                tr = await self.adapter.transcribe(audio)
            except TransientError as exc:
                if exc.status == 429 or exc.code in THROTTLE_CODES:
                    throttled += 1
                if (exc.ambiguous and not self.retry_ambiguous) or attempts > self.max_retries:
                    return _record(key, entry, "failed", attempts=attempts, throttled=throttled,
                                   error=_error_dict(exc))
                delay = exc.retry_after_s if exc.retry_after_s is not None else backoff_s(attempts)
                self.log(f"[{key}] {entry.utt_id} transient ({exc.status or exc.code}); retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            except PermanentError as exc:
                self.consecutive_permanent += 1
                if self.consecutive_permanent >= BREAKER_THRESHOLD and not self.disabled:
                    self.disabled = f"route_disabled_after_{BREAKER_THRESHOLD}_permanent_errors"
                    self.log(f"[{key}] disabled after {BREAKER_THRESHOLD} consecutive permanent errors")
                return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=_error_dict(exc))
            except AdapterError as exc:
                return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=_error_dict(exc))
            except Exception as exc:  # noqa: BLE001 - record unexpected adapter bugs per item
                err = _error_dict(exc)
                err["unexpected"] = True
                return _record(key, entry, "failed", attempts=attempts, throttled=throttled, error=err)
            self.consecutive_permanent = 0
            self.cache.put(cpath, {"route": key, "signature": self.signature, "utt_id": entry.utt_id,
                                   "audio_sha256": entry.audio_sha256, "created_at": utc_now(),
                                   "transcription": tr.to_dict()})
            status = "ok" if tr.text.strip() else "empty"
            return _record(key, entry, status, tr=tr, attempts=attempts, throttled=throttled, cache_file=crel)

    async def run(self, todo: list[ManifestEntry]) -> dict[str, int]:
        async def one(entry: ManifestEntry) -> None:
            async with self.sem:
                rec = await self.process(entry)
            await self.results.append(rec)
            self.counts[rec["status"]] = self.counts.get(rec["status"], 0) + 1
            lat = rec["latency"]
            main = lat.get("request_s", lat.get("final_after_audio_end_s"))
            extra = f" {main:.2f}s" if isinstance(main, (int, float)) else ""
            tag = " (cache)" if rec["cached"] else ""
            why = f" {rec['error']['message'][:120]}" if rec["error"] else (f" {rec['reason']}" if rec["reason"] else "")
            self.log(f"[{self.cfg.key}] {entry.utt_id} {rec['status']}{tag}{extra}{why}")

        try:
            await asyncio.gather(*(one(e) for e in todo))
        finally:
            await self.adapter.aclose()
        return self.counts


def run_metadata(plan: RunPlan, signatures: dict[str, dict]) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "subset": plan.subset,
        "experiment": plan.experiment,
        "items": [e.utt_id for e in plan.entries],
        "item_audio_hours": round(sum(e.duration_s for e in plan.entries) / 3600, 6),
        "routes": {
            cfg.key: {
                "adapter": cfg.adapter, "track": cfg.track, "region": cfg.region, "model_id": cfg.model_id,
                "live_verified": cfg.live_verified, "price_usd_per_audio_hour": cfg.price_usd_per_audio_hour,
                "notes": cfg.notes, "signature": signatures[cfg.key],
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
    """Create run.json, or check that a resumed run keeps the same items and route signatures."""
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
    merged["git"] = new["git"]
    merged["packages"] = new["packages"]
    merged["max_audio_hours"] = new["max_audio_hours"]
    return merged


def plan_summary(plan: RunPlan) -> list[dict[str, Any]]:
    """Per-route work estimate without calling any service."""
    previous = load_results(plan.run_dir / "results.jsonl")
    cache = Cache(plan.cache_dir)
    rows = []
    for cfg in plan.routes:
        sig = create_adapter(cfg).signature()
        done = cached = 0
        call_s = 0.0
        for e in plan.entries:
            if previous.get((cfg.key, e.utt_id), {}).get("status") in DONE_STATUSES:
                done += 1
            elif plan.use_cache and cache.path(cfg.key, sig, e.utt_id, e.audio_sha256).exists():
                cached += 1
            else:
                call_s += e.duration_s
        rows.append({"route": cfg.key, "adapter": cfg.adapter, "track": cfg.track, "items": len(plan.entries),
                     "done": done, "cached": cached, "to_call": len(plan.entries) - done - cached,
                     "audio_hours_to_call": round(call_s / 3600, 4),
                     "concurrency": int(cfg.params.get("concurrency", 4)),
                     "min_interval_s": float(cfg.params.get("min_interval_s", 0))})
    return rows


async def execute(plan: RunPlan, log: Callable[[str], None] = print) -> dict[str, Any]:
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    results = ResultsLog(plan.run_dir / "results.jsonl")
    previous = results.load()
    budget = Budget(plan.max_audio_hours)
    cache = Cache(plan.cache_dir)
    runners = [RouteRunner(cfg, plan, budget, cache, results, log) for cfg in plan.routes]
    run_json = plan.run_dir / "run.json"
    meta = _merge_run_json(run_json, run_metadata(plan, {r.cfg.key: r.signature for r in runners}))
    meta["last_started_at"] = utc_now()
    write_json(run_json, meta)

    route_sem = asyncio.Semaphore(max(1, plan.parallel_routes))

    async def guarded(runner: RouteRunner) -> tuple[str, dict[str, int]]:
        todo = [e for e in plan.entries
                if previous.get((runner.cfg.key, e.utt_id), {}).get("status") not in DONE_STATUSES]
        if not todo:
            return runner.cfg.key, {}
        async with route_sem:
            log(f"[{runner.cfg.key}] start: {len(todo)} item(s)")
            return runner.cfg.key, await runner.run(todo)

    started = time.monotonic()
    outcome = dict(await asyncio.gather(*(guarded(r) for r in runners)))
    meta["last_finished_at"] = utc_now()
    meta["last_wall_s"] = round(time.monotonic() - started, 3)
    meta["last_budget_used_audio_hours"] = round(budget.used_s / 3600, 6)
    write_json(run_json, meta)
    return {"run_dir": str(plan.run_dir), "counts": outcome, "budget_used_audio_hours": budget.used_s / 3600}
