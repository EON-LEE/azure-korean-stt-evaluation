"""Offline tests for the LLM semantic judge. No test calls Azure or any network service."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
import pytest

from kstt_eval import judge as judge_mod
from kstt_eval.data import ManifestEntry
from kstt_eval.judge import (
    ERROR_TYPES,
    FEW_SHOT,
    MEANINGS,
    AzureOpenAIJudge,
    CallResult,
    JudgeCache,
    JudgeCacheError,
    JudgeConfig,
    add_judge_parser,
    build_messages,
    judge_checkset,
    judge_run,
    load_checkset,
    load_judge_config,
    pair_key,
    rule_judgment,
    validate_judgment,
)
from kstt_eval.types import ConfigError, PermanentError, TransientError

REPO = Path(__file__).resolve().parents[1]
CHECKSET = REPO / "configs" / "judge_checkset.yaml"
JUDGE_YAML = REPO / "configs" / "judge.yaml"
ROUTES = {"routeA": "secret-model-a", "routeB": "secret-model-b"}
N_BOOT = 200

# utt_id -> (split, reference)
REFS = {
    "u1": ("eval_clean", "오늘 3시에 만나"),
    "u2": ("eval_clean", "나 숙제 안 했어"),
    "u3": ("eval_other", "점심 먹었어"),
    "u4": ("eval_other", "내일 비 온대"),
}
# (route, utt_id) -> (STT status, text)
RESULTS = {
    ("routeA", "u1"): ("ok", "오늘 3시에 만나"),     # identical -> rule preserved/none
    ("routeA", "u2"): ("ok", "나 숙제 했어"),        # LLM -> lost (shared with routeB)
    ("routeA", "u3"): ("ok", "점심 먹었어."),        # normalized identical -> rule preserved/surface
    ("routeA", "u4"): ("error", ""),                 # STT failure -> not judged
    ("routeB", "u1"): ("ok", "오늘 4시에 만나"),     # LLM -> lost/number_value
    ("routeB", "u2"): ("ok", "나 숙제 했어"),        # same text as routeA -> deduplicated
    ("routeB", "u3"): ("empty", ""),                 # empty -> rule lost
    ("routeB", "u4"): ("ok", "내일 비 온데"),        # LLM -> partial
}
VERDICTS = {
    "나 숙제 했어": {"reason": "부정 '안'이 빠졌다.", "meaning": "lost", "error_types": ["negation", "deletion"]},
    "오늘 4시에 만나": {"reason": "시각이 바뀌었다.", "meaning": "lost", "error_types": ["number_value"]},
    "내일 비 온데": {"reason": "전달 형태가 달라졌다.", "meaning": "partial", "error_types": ["substitution"]},
}
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 8, "cached_tokens": 64}


class FakeCaller:
    """Answers from VERDICTS by hypothesis text; ``bad`` hypotheses get a schema-invalid response."""

    def __init__(self, verdicts=None, bad=(), usage=None):
        self.verdicts = VERDICTS if verdicts is None else verdicts
        self.bad = set(bad)
        self.usage = dict(USAGE if usage is None else usage)
        self.calls = 0
        self.messages: list[list[dict]] = []
        self.closed = False

    async def call(self, messages):
        self.calls += 1
        self.messages.append(messages)
        await asyncio.sleep(0)  # let other workers start, like a real in-flight request
        pair = json.loads(messages[-1]["content"])
        hyp = pair["인식 결과"]
        if hyp in self.bad:
            content = json.dumps({"meaning": "fine", "error_types": []})
        else:
            content = json.dumps(self.verdicts.get(hyp, {"reason": "같다", "meaning": "preserved",
                                                          "error_types": ["surface"]}), ensure_ascii=False)
        return CallResult(content=content, usage=dict(self.usage), model="judge-model")

    async def aclose(self):
        self.closed = True


def _cfg(**kw) -> JudgeConfig:
    base = dict(judge_id="jt", endpoint="https://judge.example.invalid", deployment="judge-dep",
                model_version="2025-01-01", api_version="2025-04-01-preview", reasoning_effort="low",
                concurrency=1, max_retries=2, schema_retries=2)
    base.update(kw)
    return JudgeConfig(**base)


def _entry(utt_id: str, split: str, reference: str) -> ManifestEntry:
    return ManifestEntry(utt_id=utt_id, split=split, source_repo="r", source_revision="rev", source_row=0,
                         original_audio_path=f"p/{utt_id}.pcm", audio_path=f"audio/{utt_id}.wav",
                         audio_sha256=f"sha-{utt_id}", duration_s=1.0, sample_rate=16000, channels=1,
                         source_sample_rate=16000, reference=reference)


def _make_run(tmp_path: Path, run_id: str = "r1", results=None) -> tuple[Path, Path, Path, Path]:
    data_dir, runs_dir, reports_dir = tmp_path / "data", tmp_path / "runs", tmp_path / "reports"
    manifest = data_dir / "manifests" / "smoke.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="\n") as f:
        for utt_id, (split, ref) in REFS.items():
            f.write(json.dumps(_entry(utt_id, split, ref).__dict__, ensure_ascii=False) + "\n")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    meta = {"run_id": run_id, "subset": "smoke", "items": list(REFS),
            "routes": {r: {"model_id": m, "track": "A"} for r, m in ROUTES.items()}}
    (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(run_dir / "results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for (route, utt_id), (status, text) in (results or RESULTS).items():
            f.write(json.dumps({"route": route, "utt_id": utt_id, "split": REFS[utt_id][0], "status": status,
                                "text": text}, ensure_ascii=False) + "\n")
    return run_dir, data_dir, reports_dir, runs_dir


def _run(tmp_path, caller, run_dir=None, **kw):
    if run_dir is None:
        run_dir, data_dir, reports_dir, _ = _make_run(tmp_path)
    else:
        data_dir, reports_dir = tmp_path / "data", tmp_path / "reports"
    kw.setdefault("max_tokens", 1_000_000)
    return judge_run(run_dir, data_dir, reports_dir, kw.pop("cfg", None) or _cfg(), caller=caller, n_boot=N_BOOT,
                     log=lambda m: None, **kw), run_dir


def _row(rows, route, utt_id):
    return next(r for r in rows if r["route"] == route and r["utt_id"] == utt_id)


def _cache_lines(run_dir: Path, judge_id: str = "jt") -> list[dict]:
    path = run_dir / "judge" / judge_id / "cache.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kstt")
    p.add_argument("--config-dir", type=Path, default=Path("configs"))
    p.add_argument("--env-file", type=Path, default=Path("configs/endpoints.env"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    sub = p.add_subparsers(dest="command", required=True)
    add_judge_parser(sub)
    return p


def _cli(tmp_path, *judge_args):
    args = _parser().parse_args(["--data-dir", str(tmp_path / "data"), "--runs-dir", str(tmp_path / "runs"),
                                 "--reports-dir", str(tmp_path / "reports"), "judge", "--judge-config",
                                 str(JUDGE_YAML), "--n-boot", str(N_BOOT), *judge_args])
    return args.func(args)


# --- rules, keys, validation --------------------------------------------------------------------


def test_rule_judgments():
    assert rule_judgment("오늘 3시에 만나", "오늘 3시에 만나") == {
        "meaning": "preserved", "error_types": ["none"], "reason": "기준 전사와 같다."}
    surface = rule_judgment("그거 진짜 맛있더라", "그거 진짜맛있더라.")
    assert surface["meaning"] == "preserved" and surface["error_types"] == ["surface"]
    for hyp in ("", "   ", None, "..."):
        lost = rule_judgment("점심 먹었어", hyp)
        assert lost["meaning"] == "lost" and lost["error_types"] == ["deletion"]
    assert rule_judgment("나 숙제 안 했어", "나 숙제 했어") is None
    assert rule_judgment("어 저는 녹차 마실게요", "저는 녹차 마실게요") is None  # fillers go to the judge


def test_pair_key_depends_on_identity_and_raw_texts():
    cfg = _cfg()
    k = pair_key(cfg, "참조", "가설")
    assert re.fullmatch(r"[0-9a-f]{64}", k) and k == pair_key(_cfg(), "참조", "가설")
    variants = [pair_key(_cfg(deployment="other"), "참조", "가설"),
                pair_key(_cfg(model_version="2026-01-01"), "참조", "가설"),
                pair_key(_cfg(reasoning_effort="high"), "참조", "가설"),
                pair_key(cfg, "참조", "가설."), pair_key(cfg, "참조2", "가설")]
    assert len({k, *variants}) == 6
    assert pair_key(_cfg(judge_id="another", concurrency=8), "참조", "가설") == k  # judge id/concurrency not in key


def test_validate_judgment():
    ok, problem = validate_judgment(json.dumps({"reason": " 같다 ", "meaning": "partial",
                                                "error_types": ["deletion", "deletion", "filler"]}))
    assert problem is None and ok["meaning"] == "partial" and ok["error_types"] == ["deletion", "filler"]
    assert ok["reason"] == "같다" and ok["reason_over_limit"] is False
    long, _ = validate_judgment(json.dumps({"reason": "가" * 120, "meaning": "lost", "error_types": ["deletion"]}))
    assert long["reason_over_limit"] is True and long["reason_chars"] == 120
    for bad in ("", "not json", "[1]", json.dumps({"meaning": "lost", "error_types": ["deletion"]}),
                json.dumps({"reason": "x", "meaning": "fine", "error_types": ["none"]}),
                json.dumps({"reason": "x", "meaning": "lost", "error_types": []}),
                json.dumps({"reason": "x", "meaning": "lost", "error_types": ["typo"]})):
        judgment, problem = validate_judgment(bad)
        assert judgment is None and problem


def test_prompt_is_blinded_and_static():
    msgs = build_messages("기준 문장", "인식 문장")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert json.loads(msgs[1]["content"]) == {"기준 전사": "기준 문장", "인식 결과": "인식 문장"}
    assert build_messages("다른", "문장")[0] == msgs[0]  # identical prefix for prompt caching
    system = msgs[0]["content"]
    assert "필드 순서는 reason, meaning, error_types" in system
    assert list(judge_mod.RESPONSE_SCHEMA["properties"]) == ["reason", "meaning", "error_types"]
    assert 4 <= len(FEW_SHOT) <= 6
    for _, _, out in FEW_SHOT:
        assert list(out) == ["reason", "meaning", "error_types"]
        assert out["meaning"] in MEANINGS and set(out["error_types"]) <= set(ERROR_TYPES)
        assert json.dumps(out, ensure_ascii=False) in system


# --- run judging --------------------------------------------------------------------------------


def test_rules_dedup_and_not_judged(tmp_path):
    caller = FakeCaller()
    out, run_dir = _run(tmp_path, caller)
    assert out.exit_code == 0
    assert caller.calls == 3 and out.calls == 3 and out.keys_judged == 3  # routeA/routeB u2 judged once
    assert caller.closed
    rows = out.rows
    assert len(rows) == 8
    a1, a3, a4, b3 = (_row(rows, "routeA", "u1"), _row(rows, "routeA", "u3"), _row(rows, "routeA", "u4"),
                      _row(rows, "routeB", "u3"))
    assert (a1["source"], a1["meaning"], a1["error_types"]) == ("rule", "preserved", ["none"])
    assert (a3["source"], a3["meaning"], a3["error_types"]) == ("rule", "preserved", ["surface"])
    assert a4["status"] == "not_judged" and a4["meaning"] is None and a4["key"] is None
    assert (b3["source"], b3["meaning"], b3["error_types"]) == ("rule", "lost", ["deletion"])
    a2, b2 = _row(rows, "routeA", "u2"), _row(rows, "routeB", "u2")
    assert a2["key"] == b2["key"] and a2["source"] == b2["source"] == "llm" and a2["meaning"] == "lost"
    lines = _cache_lines(run_dir)
    assert len(lines) == 3 and {ln["status"] for ln in lines} == {"ok"}
    assert lines[0]["usage"] == USAGE and lines[0]["model"] == "judge-model"
    assert {"key", "meaning", "error_types", "reason", "usage", "model", "created_at", "status"} <= set(lines[0])
    written = [json.loads(x) for x in out.paths["judgments"].read_text(encoding="utf-8").splitlines()]
    assert len(written) == 8 and {w["status"] for w in written} == {"ok", "not_judged"}
    assert {w["source"] for w in written} == {"rule", "llm", None}


def test_cache_resume_calls_nothing(tmp_path):
    first, run_dir = _run(tmp_path, FakeCaller())
    second_caller = FakeCaller()
    second, _ = _run(tmp_path, second_caller, run_dir=run_dir)
    assert second.exit_code == 0 and second_caller.calls == 0 and second.calls == 0
    assert len(_cache_lines(run_dir)) == 3
    assert _row(second.rows, "routeB", "u1")["source"] == "cache"
    strip = [{k: v for k, v in r.items() if k != "source"} for r in first.rows]
    assert strip == [{k: v for k, v in r.items() if k != "source"} for r in second.rows]


def test_schema_failure_retries_then_judge_error_then_retried(tmp_path):
    caller = FakeCaller(bad={"오늘 4시에 만나"})
    out, run_dir = _run(tmp_path, caller)
    assert out.exit_code == 0
    assert caller.calls == 2 + 3  # two good keys + one key tried 1 + schema_retries(2) times
    b1 = _row(out.rows, "routeB", "u1")
    assert b1["status"] == "judge_error" and b1["meaning"] is None and "schema" in b1["error"]
    err = [ln for ln in _cache_lines(run_dir) if ln["status"] == "judge_error"]
    assert len(err) == 1 and err[0]["attempts"] == 3 and err[0]["usage"]["prompt_tokens"] == 300
    b = next(s for s in out.summary if s["route"] == "routeB" and s["split"] == "all")
    assert b["judge_error"] == 1 and b["judged"] == 3 and b["lost"] == 2 and b["partial"] == 1
    assert "판정 실패" in out.paths["md"].read_text(encoding="utf-8")

    retry = FakeCaller()
    again, _ = _run(tmp_path, retry, run_dir=run_dir)
    assert retry.calls == 1 and _row(again.rows, "routeB", "u1")["status"] == "ok"
    assert _row(again.rows, "routeB", "u1")["meaning"] == "lost"


def test_token_budget_stops_and_resumes(tmp_path):
    caller = FakeCaller()
    out, run_dir = _run(tmp_path, caller, max_tokens=100)
    assert out.exit_code == judge_mod.EXIT_BUDGET == 3
    assert caller.calls == 1 and out.remaining == 2 and out.tokens_used == 120
    statuses = [r["status"] for r in out.rows]
    assert statuses.count("pending") >= 2
    assert "미완료" in out.paths["md"].read_text(encoding="utf-8")
    assert out.paths["judgments"].exists()

    still, _ = _run(tmp_path, FakeCaller(), run_dir=run_dir, max_tokens=120)  # budget already used up
    assert still.exit_code == 3 and still.calls == 0 and still.remaining == 2

    resume = FakeCaller()
    done, _ = _run(tmp_path, resume, run_dir=run_dir, max_tokens=10_000)
    assert done.exit_code == 0 and resume.calls == 2 and done.tokens_used == 360
    assert "pending" not in {r["status"] for r in done.rows}


def test_max_tokens_required_unless_dry_run(tmp_path):
    run_dir, data_dir, reports_dir, _ = _make_run(tmp_path)
    with pytest.raises(ConfigError, match="max-judge-tokens"):
        judge_run(run_dir, data_dir, reports_dir, _cfg(), caller=FakeCaller(), log=lambda m: None)


def test_aggregation_and_baseline(tmp_path):
    out, _ = _run(tmp_path, FakeCaller(), baseline="routeA")
    s = {(x["route"], x["split"]): x for x in out.summary}
    a, b = s[("routeA", "all")], s[("routeB", "all")]
    assert (a["target"], a["not_judged"], a["judged"], a["preserved"], a["partial"], a["lost"]) == (4, 1, 3, 2, 0, 1)
    assert a["mpr"] == pytest.approx(2 / 3) and a["lost_rate"] == pytest.approx(1 / 3) and a["partial_rate"] == 0
    assert a["rule_judged"] == 2 and a["rule_ratio"] == pytest.approx(2 / 3) and a["llm_judged"] == 1
    assert a["mpr_ci_low"] <= a["mpr"] <= a["mpr_ci_high"]
    assert a["type_negation"] == pytest.approx(1 / 3) and a["type_none"] == pytest.approx(1 / 3)
    assert a["digit_refs"] == 1 and a["number_value_rate"] == 0
    assert a["validation"] == "검증 전"
    assert (b["judged"], b["preserved"], b["partial"], b["lost"]) == (4, 0, 1, 3)
    assert b["mpr"] == 0 and b["mpr_ci_low"] == 0 and b["mpr_ci_high"] == 0
    assert b["lost_rate"] == pytest.approx(0.75) and b["partial_rate"] == pytest.approx(0.25)
    assert b["rule_judged"] == 1 and b["number_value_rate"] == 1.0 and b["type_deletion"] == pytest.approx(0.5)
    clean_a, other_b = s[("routeA", "eval_clean")], s[("routeB", "eval_other")]
    assert (clean_a["judged"], clean_a["mpr"]) == (2, 0.5)
    assert (other_b["judged"], other_b["lost"], other_b["partial"]) == (2, 1, 1)

    cmp = {c["metric"]: c for c in out.comparisons}
    assert set(cmp) == {"mpr", "lost_rate"} and cmp["mpr"]["route"] == "routeB"
    assert cmp["mpr"]["n_common"] == 3  # routeA u4 failed STT, so it is not paired
    assert cmp["mpr"]["delta"] == pytest.approx(-2 / 3) and cmp["lost_rate"]["delta"] == pytest.approx(2 / 3)
    assert b["mpr_delta"] == pytest.approx(-2 / 3) and b["baseline"] == "routeA"

    md = out.paths["md"].read_text(encoding="utf-8")
    assert "검증 전" in md and "routeA" in md and "기준 경로 대비 차이" in md
    csv_text = out.paths["csv"].read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0].startswith("route,split,validation,")
    assert len(csv_text.splitlines()) == 1 + 2 * 3  # (all + 2 splits) per route


def test_routes_filter(tmp_path):
    caller = FakeCaller()
    out, _ = _run(tmp_path, caller, routes="routeA")
    assert {r["route"] for r in out.rows} == {"routeA"} and caller.calls == 1
    with pytest.raises(ConfigError, match="no route"):
        _run(tmp_path / "x", FakeCaller(), routes="nope")


# --- cache file robustness ----------------------------------------------------------------------


def _good_line(key: str) -> str:
    return json.dumps({"key": key, "status": "ok", "meaning": "lost", "error_types": ["deletion"],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def test_truncated_last_cache_line_is_dropped_and_repaired(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_bytes((_good_line("k1") + "\n" + '{"key": "k2", "sta').encode("utf-8"))
    logs: list[str] = []
    cache = JudgeCache(path, log=logs.append)
    state = cache.load(repair=False)
    assert set(state.records) == {"k1"} and state.dropped_tail and state.tokens_used == 15
    assert path.read_bytes().endswith(b'"sta')  # read-only load leaves the file alone
    state = cache.load(repair=True)
    assert set(state.records) == {"k1"} and any("truncated" in m for m in logs)
    assert path.read_text(encoding="utf-8") == _good_line("k1") + "\n"
    cache.append(json.loads(_good_line("k2")))
    assert set(cache.load().records) == {"k1", "k2"}


def test_cache_line_without_newline_is_terminated(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(_good_line("k1"), encoding="utf-8")
    cache = JudgeCache(path, log=lambda m: None)
    assert set(cache.load(repair=True).records) == {"k1"}
    cache.append(json.loads(_good_line("k2")))
    assert set(cache.load().records) == {"k1", "k2"}


def test_corrupted_middle_cache_line_raises(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text(_good_line("k1") + "\n{broken\n" + _good_line("k2") + "\n", encoding="utf-8")
    with pytest.raises(JudgeCacheError, match="corrupted"):
        JudgeCache(path, log=lambda m: None).load(repair=True)
    assert issubclass(JudgeCacheError, ConfigError)


def test_truncated_cache_in_run_is_recovered(tmp_path):
    _, run_dir = _run(tmp_path, FakeCaller())
    path = run_dir / "judge" / "jt" / "cache.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][: len(lines[2]) // 2], encoding="utf-8")
    caller = FakeCaller()
    out, _ = _run(tmp_path, caller, run_dir=run_dir)
    assert caller.calls == 1 and out.exit_code == 0
    assert [json.loads(x)["status"] for x in path.read_text(encoding="utf-8").splitlines()] == ["ok"] * 3


# --- HTTP layer (MockTransport) -----------------------------------------------------------------


def _completion(content: dict, usage: dict | None = None) -> dict:
    return {"model": "judge-model-2025", "choices": [{"index": 0, "finish_reason": "stop",
                                                      "message": {"role": "assistant",
                                                                  "content": json.dumps(content, ensure_ascii=False)}}],
            "usage": usage or {"prompt_tokens": 900, "completion_tokens": 60,
                               "prompt_tokens_details": {"cached_tokens": 768},
                               "completion_tokens_details": {"reasoning_tokens": 40}}}


@pytest.fixture
def fake_auth(monkeypatch):
    async def headers(params, key_header):
        return {"Authorization": "Bearer test-token"}

    monkeypatch.setattr(judge_mod, "auth_headers", headers)


def test_request_is_blinded_and_well_formed(tmp_path, fake_auth):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        pair = json.loads(json.loads(request.content)["messages"][-1]["content"])
        verdict = VERDICTS.get(pair["인식 결과"]) or {"reason": "같다", "meaning": "preserved", "error_types": ["none"]}
        return httpx.Response(200, json=_completion(verdict))

    # judge_run drives its own event loop, so the injected client is created and closed outside it.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    out, run_dir = _run(tmp_path, AzureOpenAIJudge(_cfg(), client=client, log=lambda m: None))
    asyncio.run(client.aclose())
    assert out.exit_code == 0 and len(requests) == 3
    for req in requests:
        assert req.method == "POST"
        assert str(req.url) == ("https://judge.example.invalid/openai/deployments/judge-dep/chat/completions"
                                "?api-version=2025-04-01-preview")
        assert req.headers["Authorization"] == "Bearer test-token"
        raw = req.content.decode("utf-8")
        for name in [*ROUTES, *ROUTES.values(), "routeA", "track"]:
            assert name not in raw
        body = json.loads(raw)
        assert set(body) == {"messages", "response_format", "max_completion_tokens", "reasoning_effort"}
        assert "temperature" not in body and body["reasoning_effort"] == "low"
        rf = body["response_format"]
        assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "stt_meaning_judgment"
        assert rf["json_schema"]["schema"]["required"] == ["reason", "meaning", "error_types"]
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert set(json.loads(body["messages"][1]["content"])) == {"기준 전사", "인식 결과"}
    usage = _cache_lines(run_dir)[0]["usage"]
    assert usage == {"prompt_tokens": 900, "completion_tokens": 60, "reasoning_tokens": 40, "cached_tokens": 768}
    assert _cache_lines(run_dir)[0]["model"] == "judge-model-2025"


def test_reasoning_effort_omitted_when_null():
    body = AzureOpenAIJudge(_cfg(reasoning_effort=None)).request_body(build_messages("a", "b"))
    assert "reasoning_effort" not in body and "temperature" not in body


def test_retry_after_429_then_success(fake_auth):
    statuses = iter([429, 503, 200])
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        seen.append(status)
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"code": "429"}})
        if status == 503:
            return httpx.Response(503, headers={"retry-after-ms": "1500"}, text="busy")
        return httpx.Response(200, json=_completion({"reason": "같다", "meaning": "preserved", "error_types": ["none"]}))

    delays: list[float] = []

    async def fake_sleep(s):
        delays.append(s)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            j = AzureOpenAIJudge(_cfg(max_retries=4), client=client, log=lambda m: None)
            j.sleep = fake_sleep
            return await j.call(build_messages("a", "b"))

    res = asyncio.run(scenario())
    assert seen == [429, 503, 200] and delays == [7.0, 1.5]
    assert json.loads(res.content)["meaning"] == "preserved" and res.usage["cached_tokens"] == 768


def test_transient_errors_exhaust_retries(fake_auth):
    calls: list[int] = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="err")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            j = AzureOpenAIJudge(_cfg(max_retries=2), client=client, log=lambda m: None)

            async def no_sleep(s):
                return None

            j.sleep = no_sleep
            await j.call(build_messages("a", "b"))

    with pytest.raises(TransientError):
        asyncio.run(scenario())
    assert len(calls) == 3


def test_auth_rejection_stops_run_without_cache_lines(tmp_path, fake_auth):
    def handler(request):
        return httpx.Response(401, json={"error": {"code": "PermissionDenied", "message": "no role"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(ConfigError, match="rejected"):
        _run(tmp_path, AzureOpenAIJudge(_cfg(), client=client, log=lambda m: None))
    asyncio.run(client.aclose())
    assert _cache_lines(tmp_path / "runs" / "r1") == []


def test_bad_request_is_judge_error(tmp_path):
    class Rejecting(FakeCaller):
        async def call(self, messages):
            self.calls += 1
            raise PermanentError("HTTP 400: content filter", status=400)

    caller = Rejecting()
    out, run_dir = _run(tmp_path, caller)
    assert caller.calls == 3  # one attempt per key; permanent errors are not retried
    assert {ln["status"] for ln in _cache_lines(run_dir)} == {"judge_error"}
    assert out.exit_code == 0 and sum(r["status"] == "judge_error" for r in out.rows) == 4


# --- CLI, configs, checkset ---------------------------------------------------------------------


def test_cli_dry_run_counts(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KSTT_JUDGE_AOAI_ENDPOINT", raising=False)

    def no_caller(cfg):
        raise AssertionError("dry run must not create a caller")

    monkeypatch.setattr(judge_mod, "create_caller", no_caller)
    run_dir, *_ = _make_run(tmp_path)
    assert _cli(tmp_path, "--run", "r1", "--dry-run") == 0
    out = capsys.readouterr().out
    table = {ln.split()[0]: [int(x) for x in ln.split()[1:]] for ln in out.splitlines()
             if ln.split() and ln.split()[0] in ROUTES}
    # items scored not_j rule llm keys cached call
    assert table["routeA"] == [4, 3, 1, 2, 1, 1, 0, 1]
    assert table["routeB"] == [4, 4, 0, 1, 3, 3, 0, 3]
    assert "unique keys after dedup: 3" in out and "to call: 3" in out
    assert "not ready for live calls" in out
    assert not (run_dir / "judge").exists() and not (tmp_path / "reports").exists()

    _run(tmp_path, FakeCaller(), run_dir=run_dir, cfg=load_judge_config(JUDGE_YAML, "gpt-5-mini"))
    capsys.readouterr()
    assert _cli(tmp_path, "--run", "r1", "--dry-run", "--routes", "routeB") == 0
    out = capsys.readouterr().out
    assert "cached ok: 3; to call: 0" in out and "routeA" not in out.split("unique keys")[0].split("\n", 1)[1]


def test_cli_requires_budget_and_one_target(tmp_path):
    _make_run(tmp_path)
    with pytest.raises(ConfigError, match="max-judge-tokens"):
        _cli(tmp_path, "--run", "r1")
    with pytest.raises(ConfigError, match="exactly one"):
        _cli(tmp_path, "--dry-run")
    with pytest.raises(ConfigError, match="unknown judge"):
        _cli(tmp_path, "--run", "r1", "--dry-run", "--judge", "nope")


def test_cli_live_path_with_injected_caller(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KSTT_JUDGE_AOAI_ENDPOINT", "https://judge.example.invalid")
    callers: list[FakeCaller] = []

    def factory(cfg):
        callers.append(FakeCaller())
        return callers[-1]

    monkeypatch.setattr(judge_mod, "create_caller", factory)
    _make_run(tmp_path)
    assert _cli(tmp_path, "--run", "r1", "--max-judge-tokens", "100", "--concurrency", "2") == 3
    assert callers[-1].calls == 2  # both workers start one call before the budget is reached; both finish
    assert len(_cache_lines(tmp_path / "runs" / "r1", "gpt-5-mini")) == 2
    assert "token budget reached" in capsys.readouterr().out
    assert _cli(tmp_path, "--run", "r1", "--max-judge-tokens", "5000", "--baseline", "routeA") == 0
    assert callers[-1].calls == 1
    assert (tmp_path / "reports" / "r1" / "judge_gpt-5-mini.md").exists()
    assert (tmp_path / "runs" / "r1" / "judge" / "gpt-5-mini" / "judgments.jsonl").exists()


def test_cli_refuses_live_calls_without_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("KSTT_JUDGE_AOAI_ENDPOINT", raising=False)
    monkeypatch.setattr(judge_mod, "create_caller", lambda cfg: pytest.fail("must not create a caller"))
    _make_run(tmp_path)
    with pytest.raises(ConfigError, match="endpoint is not configured"):
        _cli(tmp_path, "--run", "r1", "--max-judge-tokens", "1000")


def test_judge_yaml_uses_env_placeholders(monkeypatch):
    monkeypatch.delenv("KSTT_JUDGE_AOAI_ENDPOINT", raising=False)
    monkeypatch.delenv("KSTT_JUDGE_MINI_DEPLOYMENT", raising=False)
    mini = load_judge_config(JUDGE_YAML, "gpt-5-mini")
    assert not mini.ready and "KSTT_JUDGE_AOAI_ENDPOINT" in mini.missing_env
    assert (mini.deployment, mini.model_version, mini.reasoning_effort) == ("gpt-5-mini", "2025-08-07", "low")
    assert mini.max_completion_tokens == 2000 and mini.concurrency == 4
    large = load_judge_config(JUDGE_YAML, "gpt-5.5", concurrency=2)
    assert (large.deployment, large.model_version, large.concurrency) == ("gpt-5.5", "2026-04-24", 2)
    monkeypatch.setenv("KSTT_JUDGE_AOAI_ENDPOINT", "https://judge.example.invalid/")
    monkeypatch.setenv("KSTT_JUDGE_MINI_DEPLOYMENT", "my-mini")
    ready = load_judge_config(JUDGE_YAML, "gpt-5-mini")
    assert ready.ready and ready.url.startswith("https://judge.example.invalid/openai/deployments/my-mini/")
    text = JUDGE_YAML.read_text(encoding="utf-8")
    assert "https://" not in text and ".azure.com" not in text
    example = (REPO / "configs" / "endpoints.env.example").read_text(encoding="utf-8")
    assert "KSTT_JUDGE_AOAI_ENDPOINT=https://<judge-ai-services-resource>.openai.azure.com" in example


def test_checkset_file_is_valid():
    items = load_checkset(CHECKSET)
    assert len(items) == 20 and len({it.utt_id for it in items}) == 20
    labels = {it.expected for it in items}
    assert labels == set(MEANINGS)
    few_shot_refs = {ref for ref, _, _ in FEW_SHOT}
    assert not few_shot_refs & {it.reference for it in items}
    pairs = {(it.reference, it.hypothesis): it.expected for it in items}
    assert pairs[("월세가 30만 원이야", "월세가 삼십만 원이야")] == "preserved"
    assert pairs[("나 숙제 안 했어", "나 숙제 했어")] == "lost"
    assert pairs[("시험은 25일에 봐", "시험은 26일에 봐")] == "lost"
    assert {it.category for it in items} >= {"filler", "spacing", "punctuation", "negation", "unrelated",
                                             "wrong_language", "side_info_deletion", "named_entity"}


def test_checkset_cli_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(judge_mod, "create_caller", lambda cfg: pytest.fail("must not create a caller"))
    assert _cli(tmp_path, "--checkset", str(CHECKSET), "--dry-run") == 0
    out = capsys.readouterr().out
    assert "pairs=20" in out and "unique keys after dedup: 20" in out and "to call: 20" in out
    assert not (tmp_path / "runs").exists()


def test_checkset_run_reports_accuracy(tmp_path):
    items = load_checkset(CHECKSET)
    verdicts = {it.hypothesis: {"reason": "점검", "meaning": it.expected,
                                "error_types": ["none"] if it.expected == "preserved" else ["substitution"]}
                for it in items}
    verdicts[items[0].hypothesis] = {"reason": "틀림", "meaning": "lost", "error_types": ["deletion"]}
    caller = FakeCaller(verdicts=verdicts)
    out = judge_checkset(CHECKSET, tmp_path / "runs", tmp_path / "reports", _cfg(), max_tokens=10**6, caller=caller,
                         log=lambda m: None)
    assert out.exit_code == 0 and caller.calls == 20  # rules are not applied to the check set
    assert sum(r["match"] for r in out.rows) == 19 and out.rows[0]["match"] is False
    md = out.paths["md"].read_text(encoding="utf-8")
    assert "FAIL" in md and "19/20" in md
    assert (tmp_path / "runs" / "judge-checkset" / "jt" / "cache.jsonl").exists()
