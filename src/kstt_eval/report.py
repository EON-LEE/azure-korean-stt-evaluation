"""Score a run and write CSV / Markdown reports.

``per_utterance.csv`` contains reference transcripts. Keep ``reports/`` local (it is git-ignored).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import load_manifest
from .metrics import METRICS, bootstrap_ci, corpus_rate, finite_or_none, hangul_stats, paired_bootstrap, percentiles, \
    score_pair
from .normalize_ko import NORMALIZER_VERSION, normalize
from .runner import load_results
from .types import ManifestEntry
from .util import write_json

SCORED = ("ok", "empty")
STATUSES = ("ok", "empty", "failed", "skipped", "missing")
DEFAULT_BASELINE = "fast-azure"
LOW_HANGUL_RATIO = 0.5
# An utterance whose output hangul ratio is this much lower than its reference likely came back in another language.
SCRIPT_GAP = 0.2


def _script_mismatch(ref_norm: str, hyp: str) -> bool:
    hyp_h, hyp_l = hangul_stats(hyp)
    if not hyp_l:
        return False
    ref_h, ref_l = hangul_stats(ref_norm)
    ref_ratio = ref_h / ref_l if ref_l else 1.0
    return hyp_h / hyp_l <= ref_ratio - SCRIPT_GAP


def _main_latency(track: str, latency: dict[str, Any]) -> float | None:
    key = "request_s" if track == "A" else "final_after_audio_end_s"
    v = latency.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _lat(latency: dict[str, Any], key: str) -> float | None:
    v = latency.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def score_run(run_dir: Path, data_dir: Path) -> tuple[dict, list[dict]]:
    """Return (run metadata, one scored row per route × item, including missing items)."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    results = load_results(run_dir / "results.jsonl")
    subset = meta["subset"]
    by_id: dict[str, ManifestEntry] = {e.utt_id: e for e in load_manifest(data_dir, subset)}
    missing_refs = [u for u in meta["items"] if u not in by_id]
    if missing_refs:
        raise FileNotFoundError(f"{len(missing_refs)} run item(s) are not in the {subset} manifest, "
                                f"e.g. {missing_refs[:3]}. Re-run `kstt prepare-data --subset {subset}`.")
    rows = []
    for route, spec in meta["routes"].items():
        track = spec.get("track", "")
        for utt_id in meta["items"]:
            entry = by_id[utt_id]
            rec = results.get((route, utt_id))
            status = rec["status"] if rec else "missing"
            row: dict[str, Any] = {
                "route": route, "track": track, "utt_id": utt_id, "split": entry.split,
                "duration_s": entry.duration_s, "status": status,
                "cached": bool(rec and rec.get("cached")),
                "attempts": rec.get("attempts", 0) if rec else 0,
                "throttled": rec.get("throttled", 0) if rec else 0,
                "forced_commit": bool(rec and (rec.get("meta") or {}).get("forced_commit")),
                "error": (rec.get("error") or {}).get("message") if rec else None,
                "reason": rec.get("reason") if rec else None,
                "ref_norm": normalize(entry.reference),
                "hyp_raw": None, "hyp_norm": None, "counts": None,
                "latency": None, "first_partial_s": None, "connect_s": None, "request_s": None,
                "hangul": 0, "letters": 0, "script_mismatch": False,
            }
            if status in SCORED:
                text = rec.get("text") or ""
                lat = rec.get("latency") or {}
                row["hyp_raw"] = text
                row["hyp_norm"] = normalize(text)
                row["counts"] = score_pair(row["ref_norm"], row["hyp_norm"])
                row["latency"] = _main_latency(track, lat)
                row["first_partial_s"] = _lat(lat, "first_partial_s")
                row["connect_s"] = _lat(lat, "connect_s")
                row["request_s"] = _lat(lat, "request_s")
                row["hangul"], row["letters"] = hangul_stats(text)
                row["script_mismatch"] = _script_mismatch(row["ref_norm"], text)
            rows.append(row)
    return meta, rows


def _aggregate(route: str, split: str, rows: list[dict], spec: dict, n_boot: int) -> dict[str, Any]:
    status_n = {s: sum(r["status"] == s for r in rows) for s in STATUSES}
    scored = [r for r in rows if r["status"] in SCORED]
    out: dict[str, Any] = {
        "route": route, "track": spec.get("track"), "model_id": spec.get("model_id"), "region": spec.get("region"),
        "split": split, "target": len(rows), **status_n, "scored": len(scored),
        "coverage": len(scored) / len(rows) if rows else None,
        "audio_hours_scored": sum(r["duration_s"] for r in scored) / 3600,
    }
    for m in METRICS:
        out[m] = finite_or_none(corpus_rate(r["counts"][m] for r in scored))
    point, lo, hi = bootstrap_ci([r["counts"]["scer"].errors for r in scored],
                                 [r["counts"]["scer"].ref_len for r in scored],
                                 [r["split"] for r in scored], n_boot=n_boot)
    out["scer_ci_low"], out["scer_ci_high"] = finite_or_none(lo), finite_or_none(hi)
    hangul = sum(r["hangul"] for r in scored)
    letters = sum(r["letters"] for r in scored)
    out["hangul_ratio"] = hangul / letters if letters else None
    out["script_mismatch"] = sum(r["script_mismatch"] for r in scored)
    lat = percentiles(r["latency"] for r in scored)
    out["latency_p50_s"], out["latency_p90_s"] = finite_or_none(lat["p50"]), finite_or_none(lat["p90"])
    out["first_partial_p50_s"] = finite_or_none(percentiles(r["first_partial_s"] for r in scored)["p50"])
    out["connect_p50_s"] = finite_or_none(percentiles(r["connect_s"] for r in scored)["p50"])
    rtf = [r["request_s"] / r["duration_s"] for r in scored if r["request_s"] is not None and r["duration_s"] > 0]
    out["rtf_p50"] = finite_or_none(percentiles(rtf)["p50"])
    out["cached"] = sum(r["cached"] for r in rows)
    out["throttled"] = sum(r["throttled"] for r in rows)
    out["attempts"] = sum(r["attempts"] for r in rows)
    out["forced_commit"] = sum(r["forced_commit"] for r in rows)
    price = spec.get("price_usd_per_audio_hour")
    target_hours = sum(r["duration_s"] for r in rows) / 3600
    out["price_usd_per_audio_hour"] = price
    out["est_cost_usd"] = price * target_hours if isinstance(price, (int, float)) else None
    return out


def summarize(meta: dict, rows: list[dict], n_boot: int = 1000) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["route"]].append(r)
    summary = []
    for route, route_rows in grouped.items():
        spec = meta["routes"][route]
        summary.append(_aggregate(route, "all", route_rows, spec, n_boot))
        for split in sorted({r["split"] for r in route_rows}):
            summary.append(_aggregate(route, split, [r for r in route_rows if r["split"] == split], spec, n_boot))
    return summary


def compare(rows: list[dict], baseline: str, n_boot: int = 1000) -> list[dict]:
    by_route: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["status"] in SCORED:
            by_route[r["route"]][r["utt_id"]] = r
    base = by_route.get(baseline)
    if not base:
        return []
    out = []
    for route, items in by_route.items():
        if route == baseline:
            continue
        common = [u for u in base if u in items]
        for m in ("scer", "cer"):
            ref_lens = [base[u]["counts"][m].ref_len for u in common]
            ea = [base[u]["counts"][m].errors for u in common]
            eb = [items[u]["counts"][m].errors for u in common]
            res = paired_bootstrap(ea, eb, ref_lens, [base[u]["split"] for u in common], n_boot=n_boot)
            denom = sum(ref_lens)
            out.append({
                "route": route, "baseline": baseline, "metric": m, "n_common": len(common),
                "baseline_rate": sum(ea) / denom if denom else None,
                "route_rate": sum(eb) / denom if denom else None,
                "delta": finite_or_none(res["delta"]), "ci_low": finite_or_none(res["ci_low"]),
                "ci_high": finite_or_none(res["ci_high"]), "p_value": finite_or_none(res["p_value"]),
            })
    return out


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def _utterance_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        c = r["counts"]
        out.append({
            "route": r["route"], "utt_id": r["utt_id"], "split": r["split"], "duration_s": r["duration_s"],
            "status": r["status"], "cached": r["cached"], "attempts": r["attempts"], "throttled": r["throttled"],
            "scer_errors": c["scer"].errors if c else None, "scer_ref_len": c["scer"].ref_len if c else None,
            "cer_errors": c["cer"].errors if c else None, "wer_errors": c["wer"].errors if c else None,
            "wer_ref_len": c["wer"].ref_len if c else None,
            "latency_s": r["latency"], "first_partial_s": r["first_partial_s"],
            "script_mismatch": r["script_mismatch"], "ref_norm": r["ref_norm"], "hyp_norm": r["hyp_norm"], "hyp_raw": r["hyp_raw"],
            "error": r["error"], "reason": r["reason"],
        })
    return out


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _sec(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}s"


def _num(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def render_markdown(meta: dict, summary: list[dict], comparisons: list[dict], rows: list[dict],
                    baseline: str) -> str:
    items = meta["items"]
    audio_h = meta.get("item_audio_hours")
    lines = [f"# 한국어 STT 평가 결과: {meta['run_id']}", ""]
    if meta["subset"] != "full":
        lines += [
            f"> **주의:** 이 실행의 데이터는 `{meta['subset']}` subset({len(items)}개 발화)이다. "
            "이 결과는 경로 연결과 파이프라인 동작 확인용이다. 이 결과로 모델 성능을 비교하지 않는다.", "",
        ]
    lines += [
        "## 실행 정보", "",
        f"- 데이터: KsponSpeech 평가 세트 {meta['subset']} subset, 발화 {len(items)}개, 오디오 {audio_h}시간",
        f"- 데이터셋 revision: " + ", ".join(f"`{k}@{v[:12]}`" for k, v in meta.get("dataset_revisions", {}).items()),
        f"- 정규화: `{meta.get('normalizer', NORMALIZER_VERSION)}`",
        f"- 코드: `{(meta.get('git') or {}).get('commit') or '-'}`"
        + (" (커밋되지 않은 변경 포함)" if (meta.get("git") or {}).get("dirty") else ""),
        f"- 기준 경로(baseline): `{baseline}`", "",
        "## 경로별 결과(전체)", "",
        "sCER은 공백을 제거한 문자 오류율이다. CI는 발화 단위 bootstrap 95% 구간이다. "
        "지연은 Track A에서 요청 시간, Track B에서 음성 끝부터 최종 결과까지의 시간이다. "
        "Track B의 음성 끝은 원본 클립의 끝이다. 클립 끝의 무음 구간에서 최종 결과가 나오면 지연이 음수일 수 있다.", "",
        "| 경로 | Track | 모델 | 성공/대상 | 실패 | 건너뜀 | sCER | sCER 95% CI | CER | WER | 한글 비율 | 지연 p50 | 지연 p90 | 첫 부분 결과 p50 | RTF p50 | 429 | 캐시 | 예상 비용 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    overall = sorted((s for s in summary if s["split"] == "all"),
                     key=lambda s: (s["track"] or "", s["scer"] if s["scer"] is not None else 9e9, s["route"]))
    for s in overall:
        ci = "-" if s["scer_ci_low"] is None else f"{_pct(s['scer_ci_low'])}–{_pct(s['scer_ci_high'])}"
        cost = "미설정" if s["est_cost_usd"] is None else f"${s['est_cost_usd']:.4f}"
        lines.append(
            f"| `{s['route']}` | {s['track']} | {s['model_id'] or '-'} | {s['scored']}/{s['target']} | {s['failed']} | "
            f"{s['skipped'] + s['missing']} | {_pct(s['scer'])} | {ci} | {_pct(s['cer'])} | {_pct(s['wer'])} | "
            f"{_num(s['hangul_ratio'])} | {_sec(s['latency_p50_s'])} | {_sec(s['latency_p90_s'])} | "
            f"{_sec(s['first_partial_p50_s'])} | {_num(s['rtf_p50'], 3)} | {s['throttled']} | {s['cached']} | {cost} |")
    lines += ["", "## split별 sCER", "", "| 경로 | split | 성공/대상 | sCER | sCER 95% CI |", "|---|---|---|---|---|"]
    for s in sorted((s for s in summary if s["split"] != "all"), key=lambda s: (s["route"], s["split"])):
        ci = "-" if s["scer_ci_low"] is None else f"{_pct(s['scer_ci_low'])}–{_pct(s['scer_ci_high'])}"
        lines.append(f"| `{s['route']}` | {s['split']} | {s['scored']}/{s['target']} | {_pct(s['scer'])} | {ci} |")

    lines += ["", f"## 기준 경로 대비 차이(`{baseline}`)", ""]
    if comparisons:
        lines += ["차이는 `경로 - 기준`이다. 음수이면 기준보다 오류가 적다. 두 경로가 모두 성공한 발화만 비교한다.", "",
                  "| 경로 | 지표 | 공통 발화 | 기준 | 경로 | 차이 | 95% CI | p |", "|---|---|---|---|---|---|---|---|"]
        metric_order = {"scer": 0, "cer": 1}
        for c in sorted(comparisons, key=lambda c: (metric_order.get(c["metric"], 9), c["route"])):
            ci = "-" if c["ci_low"] is None else f"{_pct(c['ci_low'])}–{_pct(c['ci_high'])}"
            lines.append(f"| `{c['route']}` | {c['metric']} | {c['n_common']} | {_pct(c['baseline_rate'])} | "
                         f"{_pct(c['route_rate'])} | {_pct(c['delta'])} | {ci} | {_num(c['p_value'], 3)} |")
    else:
        lines.append(f"기준 경로 `{baseline}`의 성공 결과가 없어서 비교하지 않았다.")

    warnings = []
    for s in overall:
        if s["hangul_ratio"] is not None and s["hangul_ratio"] < LOW_HANGUL_RATIO:
            warnings.append(f"`{s['route']}`: 한글 비율이 {_num(s['hangul_ratio'])}이다. 한국어가 아닌 출력을 확인한다.")
        if s["script_mismatch"]:
            warnings.append(f"`{s['route']}`: 한글 비율이 정답보다 {SCRIPT_GAP} 이상 낮은 출력이 {s['script_mismatch']}건이다. "
                            "다른 언어로 인식했는지 per_utterance.csv의 `script_mismatch` 열에서 확인한다.")
        if s["empty"]:
            warnings.append(f"`{s['route']}`: 빈 출력 {s['empty']}건을 전부 삭제 오류로 채점했다.")
        if s["forced_commit"]:
            warnings.append(f"`{s['route']}`: VAD가 끝나지 않아 강제 commit한 발화가 {s['forced_commit']}건이다.")
    failures = [r for r in rows if r["status"] in ("failed", "skipped", "missing")]
    if warnings:
        lines += ["", "## 확인 필요", ""] + [f"- {w}" for w in warnings]
    if failures:
        lines += ["", "## 실패·건너뜀", "", "| 경로 | 발화 | 상태 | 시도 | 내용 |", "|---|---|---|---|---|"]
        for r in failures[:200]:
            detail = (r["error"] or r["reason"] or "").replace("|", "\\|").replace("\n", " ")[:160]
            lines.append(f"| `{r['route']}` | {r['utt_id']} | {r['status']} | {r['attempts']} | {detail} |")
        if len(failures) > 200:
            lines.append(f"| … | | | | 외 {len(failures) - 200}건은 per_utterance.csv에서 확인한다. |")
    lines += ["", "## 파일", "",
              "- `summary.csv`: 경로 × split 집계",
              "- `comparisons.csv`: 기준 경로 대비 paired bootstrap 결과",
              "- `per_utterance.csv`: 발화별 정답·인식 결과. 정답 문장이 들어 있으므로 외부에 공유하지 않는다.", ""]
    return "\n".join(lines)


def write_report(run_dir: Path, data_dir: Path, out_dir: Path, baseline: str = DEFAULT_BASELINE,
                 n_boot: int = 1000) -> dict[str, Any]:
    meta, rows = score_run(run_dir, data_dir)
    summary = summarize(meta, rows, n_boot=n_boot)
    comparisons = compare(rows, baseline, n_boot=n_boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "summary.csv", summary)
    _write_csv(out_dir / "comparisons.csv", comparisons, [
        "route", "baseline", "metric", "n_common", "baseline_rate", "route_rate", "delta", "ci_low", "ci_high",
        "p_value"])
    _write_csv(out_dir / "per_utterance.csv", _utterance_rows(rows))
    (out_dir / "report.md").write_text(render_markdown(meta, summary, comparisons, rows, baseline), encoding="utf-8")
    write_json(out_dir / "summary.json", {"run_id": meta["run_id"], "subset": meta["subset"], "baseline": baseline,
                                          "summary": summary, "comparisons": comparisons})
    return {"out_dir": str(out_dir), "summary": summary, "comparisons": comparisons}
