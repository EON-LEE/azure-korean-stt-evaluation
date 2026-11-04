"""Edit-distance metrics, corpus aggregation and bootstrap statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .normalize_ko import remove_spaces


@dataclass(frozen=True)
class EditCounts:
    sub: int
    dele: int
    ins: int
    ref_len: int
    hyp_len: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.sub + other.sub,
            self.dele + other.dele,
            self.ins + other.ins,
            self.ref_len + other.ref_len,
            self.hyp_len + other.hyp_len,
        )


ZERO = EditCounts(0, 0, 0, 0, 0)


def align_counts(ref: Sequence, hyp: Sequence) -> EditCounts:
    """Levenshtein alignment counts.

    The total error count is the minimum edit distance. When several minimal alignments
    exist, the backtrace prefers match/substitution, then deletion, then insertion.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return EditCounts(0, 0, m, 0, m)
    if m == 0:
        return EditCounts(0, n, 0, n, 0)
    dist = [list(range(m + 1))]
    for i in range(1, n + 1):
        r = ref[i - 1]
        prev = dist[i - 1]
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if r == hyp[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        dist.append(cur)
    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dist[i][j] == dist[i - 1][j - 1] + cost:
                sub += cost
                i -= 1
                j -= 1
                continue
        if i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            dele += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return EditCounts(sub, dele, ins, n, m)


def units(normalized: str, metric: str) -> Sequence:
    """Split normalized text into scoring units for a metric."""
    if metric == "scer":
        return remove_spaces(normalized)
    if metric == "cer":
        return normalized
    if metric == "wer":
        return normalized.split()
    raise ValueError(f"unknown metric: {metric}")


METRICS = ("scer", "cer", "wer")


def score_pair(ref_norm: str, hyp_norm: str) -> dict[str, EditCounts]:
    return {m: align_counts(units(ref_norm, m), units(hyp_norm, m)) for m in METRICS}


def corpus_rate(counts: Iterable[EditCounts]) -> float:
    total = sum(counts, ZERO)
    return total.errors / total.ref_len if total.ref_len else float("nan")


def finite_or_none(value: float | None) -> float | None:
    """JSON-safe number: NaN/inf become ``None`` (callers record the reason separately)."""
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None


def hangul_stats(text: str) -> tuple[int, int]:
    """Return (Hangul syllables/jamo, all letters) in ``text`` for a Korean-output check."""
    hangul = letters = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            cp = ord(ch)
            if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
                hangul += 1
    return hangul, letters


def _check_lengths(*arrays: Sequence) -> None:
    lengths = {len(a) for a in arrays}
    if len(lengths) > 1:
        raise ValueError(f"inputs must have the same length, got {sorted(lengths)}")


def _strata_indices(strata: Sequence[str]) -> list[np.ndarray]:
    labels = np.asarray(strata)
    return [np.flatnonzero(labels == s) for s in sorted(set(strata))]


def _resample(rng: np.random.Generator, groups: list[np.ndarray], n_boot: int) -> np.ndarray:
    """Return (n_boot, N) index matrix; resampling is stratified by group."""
    parts = [g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups if len(g)]
    return np.concatenate(parts, axis=1)


def bootstrap_ci(
    errors: Sequence[int],
    ref_lens: Sequence[int],
    strata: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 20261103,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Corpus error rate with a percentile bootstrap CI over utterances.

    Resampling is stratified by ``strata`` (e.g. eval split) so that a pooled rate keeps
    the split proportions fixed.
    """
    e = np.asarray(errors, dtype=np.float64)
    r = np.asarray(ref_lens, dtype=np.float64)
    _check_lengths(e, r, *([strata] if strata is not None else []))
    if len(e) == 0 or r.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    point = e.sum() / r.sum()
    groups = _strata_indices(strata) if strata is not None else [np.arange(len(e))]
    idx = _resample(np.random.default_rng(seed), groups, n_boot)
    rates = e[idx].sum(axis=1) / np.maximum(r[idx].sum(axis=1), 1e-12)
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap(
    errors_a: Sequence[int],
    errors_b: Sequence[int],
    ref_lens: Sequence[int],
    strata: Sequence[str] | None = None,
    n_boot: int = 1000,
    seed: int = 20261103,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap of the corpus-rate difference (B - A) on the same utterances."""
    a = np.asarray(errors_a, dtype=np.float64)
    b = np.asarray(errors_b, dtype=np.float64)
    r = np.asarray(ref_lens, dtype=np.float64)
    _check_lengths(a, b, r, *([strata] if strata is not None else []))
    if len(a) == 0 or r.sum() == 0:
        nan = float("nan")
        return {"delta": nan, "ci_low": nan, "ci_high": nan, "p_value": nan, "n": 0}
    delta = (b.sum() - a.sum()) / r.sum()
    groups = _strata_indices(strata) if strata is not None else [np.arange(len(a))]
    idx = _resample(np.random.default_rng(seed), groups, n_boot)
    denom = np.maximum(r[idx].sum(axis=1), 1e-12)
    deltas = (b[idx].sum(axis=1) - a[idx].sum(axis=1)) / denom
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(1.0, p)),
        "n": int(len(a)),
    }


def percentiles(values: Iterable[float], qs: Sequence[float] = (50, 90)) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}
