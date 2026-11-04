"""``kstt`` command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from .adapters import create_adapter
from .auth import get_token, redact
from .config import load_env_file, load_experiments, load_models, select_routes
from .data import SPLITS, DataError, load_manifest, prepare
from .report import DEFAULT_BASELINE, write_report
from .runner import RunError, RunPlan, execute, plan_summary
from .types import ConfigError
from .util import new_run_id


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _load_models(args: argparse.Namespace):
    experiments = load_experiments(args.config_dir / "experiments.yaml")
    if args.experiment not in experiments:
        raise ConfigError(f"unknown experiment {args.experiment!r}; defined: {sorted(experiments)}")
    return load_models(args.config_dir / "models.yaml", experiments[args.experiment])


def cmd_models(args: argparse.Namespace) -> int:
    models = _load_models(args)
    print(f"{'route':34} {'trk':3} {'adapter':16} {'model_id':28} {'region':13} {'ready':5} verified")
    for m in models.values():
        ready = "yes" if m.ready else "no"
        print(f"{m.key:34} {m.track:3} {m.adapter:16} {m.model_id:28} {m.region:13} {ready:5} {m.live_verified or '-'}")
        if not m.ready:
            print(f"{'':34} missing env: {', '.join(sorted(m.missing_env))}")
    return 0


async def _doctor(routes, ws: bool) -> int:
    bad = 0
    for cfg in routes:
        if not cfg.ready:
            print(f"[FAIL] {cfg.key}: missing env {sorted(cfg.missing_env)}")
            bad += 1
            continue
        adapter = create_adapter(cfg)
        try:
            checks = await adapter.doctor(ws=ws)
        except Exception as exc:  # noqa: BLE001 - report every route
            checks = [{"name": "doctor", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}]
        finally:
            await adapter.aclose()
        ok = all(c["ok"] for c in checks)
        bad += not ok
        print(f"[{'OK' if ok else 'FAIL'}] {cfg.key}")
        for c in checks:
            print(f"    {'ok ' if c['ok'] else 'ERR'} {c['name']}: {redact(c['detail'])[:300]}")
    return bad


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        get_token()
        print("[OK] Entra ID token (Cognitive Services scope)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Entra ID token: {redact(exc)[:300]}\n       Run `az login` and retry.")
        return 1
    routes = select_routes(_load_models(args), args.routes)
    bad = asyncio.run(_doctor(routes, args.ws))
    print(f"\n{len(routes) - bad}/{len(routes)} route(s) ready")
    return 1 if bad else 0


def cmd_prepare(args: argparse.Namespace) -> int:
    splits = tuple(args.splits.split(",")) if args.splits else SPLITS
    if args.subset == "smoke" and (args.limit is not None or args.splits):
        raise ConfigError("--limit and --splits apply to --subset full only")
    summary = prepare(args.data_dir, subset=args.subset, splits=splits, limit=args.limit,
                      allow_revision_drift=args.allow_revision_drift, log=log)
    log(f"prepared: {summary if args.subset == 'smoke' else {s: v.get('entries') for s, v in summary.items()}}")
    return 0


def _print_plan(rows: list[dict], max_hours: float | None) -> None:
    print(f"{'route':34} {'items':>5} {'done':>5} {'cache':>5} {'call':>5} {'audio_h':>8} {'conc':>4} {'gap_s':>5}")
    for r in rows:
        print(f"{r['route']:34} {r['items']:5} {r['done']:5} {r['cached']:5} {r['to_call']:5} "
              f"{r['audio_hours_to_call']:8.4f} {r['concurrency']:4} {r['min_interval_s']:5.0f}")
    total = sum(r["audio_hours_to_call"] for r in rows)
    print(f"\ntotal audio to send (before retries): {total:.4f} h; budget: "
          f"{'none' if max_hours is None else f'{max_hours} h'}")


def cmd_run(args: argparse.Namespace) -> int:
    if args.subset == "full" and args.max_audio_hours is None and not args.dry_run:
        raise ConfigError("--subset full needs --max-audio-hours (shared budget across routes). "
                          "Use --dry-run to see the required audio hours.")
    routes = select_routes(_load_models(args), args.routes)
    not_ready = {m.key: sorted(m.missing_env) for m in routes if not m.ready}
    if not_ready and not args.dry_run:
        raise ConfigError(f"routes with missing env vars: {not_ready}. Fill {args.env_file}.")
    entries = load_manifest(args.data_dir, args.subset)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ConfigError("no manifest entries selected")
    plan = RunPlan(run_id=args.run_id or new_run_id(args.subset), runs_dir=args.runs_dir, data_dir=args.data_dir,
                   subset=args.subset, routes=routes, entries=entries, max_audio_hours=args.max_audio_hours,
                   experiment=args.experiment, parallel_routes=args.parallel_routes, use_cache=not args.no_cache)
    if args.dry_run:
        print(f"run_id={plan.run_id} subset={plan.subset} items={len(entries)} routes={len(routes)} (dry run)")
        if not_ready:
            print(f"not ready (missing env): {not_ready}")
        _print_plan(plan_summary(plan), args.max_audio_hours)
        return 0
    log(f"run_id={plan.run_id} subset={plan.subset} items={len(entries)} routes={len(routes)}")
    outcome = asyncio.run(execute(plan, log=log))
    log(f"budget used: {outcome['budget_used_audio_hours']:.4f} audio h; run dir: {outcome['run_dir']}")
    for route, counts in outcome["counts"].items():
        log(f"  {route}: {counts or 'already complete'}")
    if not args.no_report:
        out = write_report(plan.run_dir, args.data_dir, args.reports_dir / plan.run_id, baseline=args.baseline)
        log(f"report: {Path(out['out_dir']) / 'report.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = args.runs_dir / args.run
    if not (run_dir / "run.json").exists():
        raise ConfigError(f"run not found: {run_dir}")
    out = write_report(run_dir, args.data_dir, args.reports_dir / args.run, baseline=args.baseline,
                       n_boot=args.n_boot)
    log(f"report: {Path(out['out_dir']) / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kstt", description="Korean STT evaluation on Azure (KsponSpeech)")
    p.add_argument("--config-dir", type=Path, default=Path("configs"))
    p.add_argument("--env-file", type=Path, default=Path("configs/endpoints.env"),
                   help="KEY=VALUE file with endpoint variables (existing environment variables win)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--experiment", default="baseline")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("models", help="list configured routes")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("doctor", help="check auth, endpoints and deployments without sending audio")
    s.add_argument("--routes", default="all")
    s.add_argument("--ws", action="store_true", help="also open realtime WebSocket sessions (no audio sent)")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("prepare-data", help="download KsponSpeech eval audio and build manifests")
    s.add_argument("--subset", choices=("smoke", "full"), default="smoke")
    s.add_argument("--splits", help="comma list for --subset full (default: all)")
    s.add_argument("--limit", type=int, help="first N rows per split (writes the 'partial' subset)")
    s.add_argument("--allow-revision-drift", action="store_true")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("run", help="transcribe with the selected routes, then write the report")
    s.add_argument("--routes", default="all", help="keys, globs, 'all' or 'track:A|B' (comma-separated)")
    s.add_argument("--subset", choices=("smoke", "partial", "full"), default="smoke")
    s.add_argument("--limit", type=int, help="first N manifest entries")
    s.add_argument("--run-id")
    s.add_argument("--max-audio-hours", type=float, help="shared budget for all attempts on all routes")
    s.add_argument("--parallel-routes", type=int, default=2)
    s.add_argument("--no-cache", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="print the work plan without calling Azure")
    s.add_argument("--no-report", action="store_true")
    s.add_argument("--baseline", default=DEFAULT_BASELINE)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="rebuild the report for a run")
    s.add_argument("--run", required=True, help="run id under --runs-dir")
    s.add_argument("--baseline", default=DEFAULT_BASELINE)
    s.add_argument("--n-boot", type=int, default=1000)
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        loaded = load_env_file(args.env_file)
        if loaded:
            log(f"loaded {len(loaded)} variable(s) from {args.env_file}")
        return args.func(args)
    except (ConfigError, DataError, RunError, FileNotFoundError) as exc:
        print(f"error: {redact(exc)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted; re-run the same command with --run-id to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
