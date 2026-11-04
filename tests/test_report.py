import csv
import json

import pytest

from kstt_eval.report import _script_mismatch, score_run, write_report
from kstt_eval.types import ManifestEntry

REFS = {
    "u1": ("eval_clean", "안녕하세요 반갑습니다"),
    "u2": ("eval_clean", "(오늘)/(오늘은) 날씨가 좋다"),
    "u3": ("eval_other", "회의는 내일 열린다"),
}


def _setup(tmp_path, results, routes=None):
    data = tmp_path / "data"
    (data / "manifests").mkdir(parents=True)
    with open(data / "manifests" / "smoke.jsonl", "w", encoding="utf-8") as f:
        for i, (u, (split, ref)) in enumerate(REFS.items()):
            e = ManifestEntry(utt_id=u, split=split, source_repo="repo", source_revision="rev0123456789abc",
                              source_row=i, original_audio_path=f"p/{u}.pcm", audio_path=f"audio/{u}.wav",
                              audio_sha256="s", duration_s=2.0, sample_rate=16000, channels=1,
                              source_sample_rate=16000, reference=ref)
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    run = tmp_path / "runs" / "smoke-test"
    run.mkdir(parents=True)
    routes = routes or {
        "fast-azure": {"track": "A", "model_id": "azure-speech", "price_usd_per_audio_hour": 1.0},
        "rt-b": {"track": "B", "model_id": "m-b", "price_usd_per_audio_hour": None},
        "en-out": {"track": "A", "model_id": "m-en", "price_usd_per_audio_hour": None},
    }
    meta = {"run_id": "smoke-test", "subset": "smoke", "items": list(REFS), "item_audio_hours": 6 / 3600,
            "routes": routes, "normalizer": "ko-norm-v1", "dataset_revisions": {"repo": "rev0123456789abc"},
            "git": {"commit": "abc", "dirty": True}}
    (run / "run.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with open(run / "results.jsonl", "w", encoding="utf-8") as f:
        for rec in results:
            base = {"latency": {}, "meta": {}, "cached": False, "attempts": 1, "throttled": 0, "error": None,
                    "reason": None}
            f.write(json.dumps({**base, **rec}, ensure_ascii=False) + "\n")
    return data, run


RESULTS = [
    {"route": "fast-azure", "utt_id": "u1", "status": "ok", "text": "안녕하세요, 반갑습니다.", "latency": {"request_s": 0.5}},
    {"route": "fast-azure", "utt_id": "u2", "status": "ok", "text": "오늘 날씨가 좋다", "latency": {"request_s": 0.7}},
    {"route": "fast-azure", "utt_id": "u3", "status": "ok", "text": "회의는 내일 열린다", "latency": {"request_s": 0.6}},
    {"route": "rt-b", "utt_id": "u1", "status": "ok", "text": "안녕하세요 반갑습니까",
     "latency": {"final_after_audio_end_s": 0.4, "first_partial_s": 0.2}, "meta": {"forced_commit": True}},
    {"route": "rt-b", "utt_id": "u2", "status": "empty", "text": ""},
    {"route": "en-out", "utt_id": "u1", "status": "ok", "text": "annyeong haseyo"},
    {"route": "en-out", "utt_id": "u2", "status": "ok", "text": "today weather good"},
    {"route": "en-out", "utt_id": "u3", "status": "failed", "error": {"message": "HTTP 500"}, "attempts": 3},
]


def test_score_and_report(tmp_path):
    data, run = _setup(tmp_path, RESULTS)
    out_dir = tmp_path / "reports" / "smoke-test"
    out = write_report(run, data, out_dir, baseline="fast-azure", n_boot=200)
    summ = {(s["route"], s["split"]): s for s in out["summary"]}

    fa = summ[("fast-azure", "all")]
    assert fa["scer"] == 0 and fa["scored"] == 3 and fa["coverage"] == 1
    assert fa["latency_p50_s"] == pytest.approx(0.6)
    assert fa["est_cost_usd"] == pytest.approx(6 / 3600)

    rb = summ[("rt-b", "all")]
    assert (rb["ok"], rb["empty"], rb["missing"], rb["scored"]) == (1, 1, 1, 2)
    # u1: 1 substitution of 10 chars; u2: empty output deletes all 7 chars ("오늘날씨가좋다")
    assert rb["scer"] == pytest.approx(8 / 17)
    assert rb["forced_commit"] == 1 and rb["latency_p50_s"] == pytest.approx(0.4)
    assert summ[("rt-b", "eval_other")]["scored"] == 0 and summ[("rt-b", "eval_other")]["scer"] is None

    en = summ[("en-out", "all")]
    assert en["hangul_ratio"] == 0 and en["failed"] == 1
    assert en["script_mismatch"] == 2 and fa["script_mismatch"] == 0 and rb["script_mismatch"] == 0

    comps = {(c["route"], c["metric"]): c for c in out["comparisons"]}
    assert comps[("rt-b", "scer")]["n_common"] == 2
    assert comps[("rt-b", "scer")]["delta"] == pytest.approx(8 / 17)

    md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "**주의:**" in md and "미설정" in md
    assert "`en-out`: 한글 비율" in md
    assert "`en-out`: 한글 비율이 정답보다 0.2 이상 낮은 출력이 2건" in md
    assert md.index("| scer |") < md.index("| cer |")
    assert "빈 출력 1건" in md and "강제 commit" in md
    assert "| `rt-b` | u3 | missing |" in md

    raw = (out_dir / "per_utterance.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with open(out_dir / "per_utterance.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9
    for name in ("summary.csv", "comparisons.csv", "summary.json"):
        assert (out_dir / name).exists()


def test_script_mismatch_is_relative_to_reference():
    assert _script_mismatch("그냥 샌드위치", "그냥 샌드위치. やらせんでいい。")
    assert _script_mismatch("아니 그냥 샌드위치", "Oh, but uh 그냥 샌드위치")
    assert not _script_mismatch("url 경로", "URL 경로")  # the reference itself has Latin letters
    assert not _script_mismatch("url 경로", "유알엘 경로")
    assert not _script_mismatch("회의는 내일", "")
    assert not _script_mismatch("회의는 내일", "회의는, 내일!")


def test_report_without_baseline_results(tmp_path):
    data, run = _setup(tmp_path, [r for r in RESULTS if r["route"] != "fast-azure"])
    out = write_report(run, data, tmp_path / "out", baseline="fast-azure", n_boot=50)
    assert out["comparisons"] == []
    assert "성공 결과가 없어서 비교하지 않았다" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_score_run_rejects_unknown_items(tmp_path):
    data, run = _setup(tmp_path, RESULTS)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    meta["items"].append("u9")
    (run / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        score_run(run, data)
