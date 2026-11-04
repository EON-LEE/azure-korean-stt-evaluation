import math

import pytest

from kstt_eval.metrics import (
    align_counts, bootstrap_ci, corpus_rate, finite_or_none, hangul_stats, paired_bootstrap, percentiles, score_pair,
)


def test_align_counts_basic_operations():
    c = align_counts("가나다", "가나라")
    assert (c.sub, c.dele, c.ins, c.ref_len) == (1, 0, 0, 3)
    c = align_counts("가나다", "")
    assert (c.dele, c.errors) == (3, 3)
    c = align_counts("", "가나")
    assert (c.ins, c.ref_len) == (2, 0)
    c = align_counts("가나다라", "가다라마")
    assert c.errors == 2


def test_corpus_rate_is_micro_average():
    # 1-char reference with 1 error and 9-char reference with 0 errors: 1/10, not mean(100%, 0%).
    counts = [score_pair("가", "나")["scer"], score_pair("가나다라마바사아자", "가나다라마바사아자")["scer"]]
    assert corpus_rate(counts) == pytest.approx(0.1)


def test_scer_ignores_spacing_but_cer_and_wer_do_not():
    s = score_pair("안녕 하세요", "안녕하세요")
    assert s["scer"].errors == 0
    assert s["cer"].errors == 1
    assert s["wer"].errors == 2
    assert s["wer"].ref_len == 2


def test_corpus_rate_empty_is_nan_and_finite_or_none():
    assert math.isnan(corpus_rate([]))
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(0.25) == 0.25


def test_bootstrap_ci_contains_point_and_is_deterministic():
    errors = [0, 1, 2, 0, 3, 1, 0, 2]
    lens = [10, 12, 9, 11, 10, 8, 12, 10]
    strata = ["a"] * 4 + ["b"] * 4
    p1 = bootstrap_ci(errors, lens, strata, n_boot=300)
    p2 = bootstrap_ci(errors, lens, strata, n_boot=300)
    assert p1 == p2
    point, lo, hi = p1
    assert point == pytest.approx(sum(errors) / sum(lens))
    assert lo <= point <= hi


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], [3], None)
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 2], [3, 4])


def test_paired_bootstrap_identical_and_better():
    same = paired_bootstrap([1, 2, 3], [1, 2, 3], [10, 10, 10], n_boot=200)
    assert same["delta"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0 and same["n"] == 3
    better = paired_bootstrap([3, 3, 3, 3], [0, 0, 0, 0], [10, 10, 10, 10], n_boot=200)
    assert better["delta"] == pytest.approx(-0.3)
    assert better["ci_high"] < 0


def test_percentiles_skip_missing():
    out = percentiles([1.0, None, float("nan"), 3.0])
    assert out["p50"] == pytest.approx(2.0)
    assert math.isnan(percentiles([])["p50"])


def test_hangul_stats():
    assert hangul_stats("Hello 안녕!") == (2, 7)
    assert hangul_stats("") == (0, 0)
