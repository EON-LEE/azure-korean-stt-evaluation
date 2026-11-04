"""KsponSpeech evaluation data from pinned Hugging Face copies.

The copies expose a single physical split named ``train``. It is mapped to the logical
eval split here and is never used for training.

Two preparation paths exist:

* ``full``: download the pinned parquet files, verify sha256, decode every row.
  Output: ``data/manifests/full/<split>.jsonl``. With ``--limit`` the output goes to
  ``data/manifests/partial/`` so a partial run never overwrites the full manifest.
* ``smoke``: read row metadata through the Hugging Face datasets-server rows API, pick a fixed
  hash-selected subset (3 eval_clean + 2 eval_other) and download only those audio files.
  Output: ``data/manifests/smoke.jsonl``. The rows API serves the latest dataset revision, so
  the revision is checked against the pinned one.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Sequence, TypeVar

from .audio import TARGET_SR, decode_audio, encode_wav_pcm16, resample, sha256_bytes, sha256_file, to_mono
from .normalize_ko import has_kspon_markup
from .types import ManifestEntry


@dataclass(frozen=True)
class SourceSpec:
    split: str
    repo_id: str
    revision: str
    hf_split: str
    expected_rows: int
    path_prefix: str


SOURCES: dict[str, SourceSpec] = {
    "eval_clean": SourceSpec(
        split="eval_clean",
        repo_id="Kyudan/KsponSpeech-eval-clean",
        revision="65ac2ed264f5f4020927736e165551a95168e77a",
        hf_split="train",
        expected_rows=3000,
        path_prefix="KsponSpeech_eval/eval_clean/",
    ),
    "eval_other": SourceSpec(
        split="eval_other",
        repo_id="Kyudan/KsponSpeech_eval_other",
        revision="6e6ec517f71bbc59fe7556e131ac253bf699d145",
        hf_split="train",
        expected_rows=3000,
        path_prefix="KsponSpeech_eval/eval_other/",
    ),
}

SPLITS = tuple(SOURCES)
SUBSETS = ("smoke", "full", "partial")
SMOKE_SALT = "kstt-smoke-v1"
SMOKE_PER_SPLIT = {"eval_clean": 3, "eval_other": 2}
ROWS_API = "https://datasets-server.huggingface.co/rows"
ROWS_PAGE = 100


class DataError(RuntimeError):
    """Dataset source, revision or manifest problem."""


def manifest_path(data_dir: Path, subset: str, split: str | None = None) -> Path:
    if subset == "smoke":
        return data_dir / "manifests" / "smoke.jsonl"
    if subset not in ("full", "partial") or split is None:
        raise ValueError(f"invalid subset/split: {subset}/{split}")
    return data_dir / "manifests" / subset / f"{split}.jsonl"


def summary_path(data_dir: Path, subset: str) -> Path:
    if subset == "smoke":
        return data_dir / "manifests" / "smoke.summary.json"
    return data_dir / "manifests" / subset / "summary.json"


def utt_id_from_path(original_audio_path: str) -> str:
    return PurePosixPath(original_audio_path.replace("\\", "/")).stem


def smoke_key(utt_id: str) -> str:
    return hashlib.sha256(f"{SMOKE_SALT}:{utt_id}".encode()).hexdigest()


T = TypeVar("T")


def pick_smoke(items: Sequence[T], split_of: Callable[[T], str], id_of: Callable[[T], str],
               per_split: dict[str, int] = SMOKE_PER_SPLIT) -> list[T]:
    """Lowest salted-hash IDs per split. Raises if a split has fewer valid items than needed."""
    chosen: list[T] = []
    for split, n in per_split.items():
        pool = sorted((x for x in items if split_of(x) == split), key=lambda x: smoke_key(id_of(x)))
        if len(pool) < n:
            raise DataError(f"smoke selection needs {n} items from {split}, found {len(pool)}")
        chosen.extend(pool[:n])
    return chosen


def select_smoke(entries: list[ManifestEntry], per_split: dict[str, int] = SMOKE_PER_SPLIT) -> list[ManifestEntry]:
    return pick_smoke(entries, lambda e: e.split, lambda e: e.utt_id, per_split)


@dataclass
class RowCheck:
    """Validation state shared by both preparation paths."""

    spec: SourceSpec
    seen: Counter
    issues: list[dict]

    def validate(self, row_index: int, orig: str, text: str) -> str | None:
        """Return the utt_id when the row is usable, else record an issue and return None."""
        utt_id = utt_id_from_path(orig) if orig else f"{self.spec.split}-row{row_index:05d}"
        self.seen[utt_id] += 1
        if not orig.replace("\\", "/").startswith(self.spec.path_prefix):
            self.issues.append({"row": row_index, "utt_id": utt_id, "issue": "unexpected_path", "value": orig})
            return None
        if self.seen[utt_id] > 1:
            self.issues.append({"row": row_index, "utt_id": utt_id, "issue": "duplicate_id"})
            return None
        if not text:
            self.issues.append({"row": row_index, "utt_id": utt_id, "issue": "empty_reference"})
            return None
        return utt_id


def _write_wav_entry(spec: SourceSpec, data_dir: Path, utt_id: str, row_index: int, orig: str,
                     text: str, encoded: bytes) -> tuple[ManifestEntry, int, int]:
    samples, sr, ch = decode_audio(encoded)
    if samples.shape[0] == 0:
        raise ValueError("empty audio")
    mono = resample(to_mono(samples), sr, TARGET_SR)
    wav = encode_wav_pcm16(mono, TARGET_SR)
    audio_dir = data_dir / "audio" / spec.split
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = audio_dir / f"{utt_id}.wav"
    out.write_bytes(wav)
    entry = ManifestEntry(
        utt_id=utt_id,
        split=spec.split,
        source_repo=spec.repo_id,
        source_revision=spec.revision,
        source_row=row_index,
        original_audio_path=orig,
        audio_path=out.relative_to(data_dir).as_posix(),
        audio_sha256=sha256_bytes(wav),
        duration_s=round(len(mono) / TARGET_SR, 4),
        sample_rate=TARGET_SR,
        channels=1,
        source_sample_rate=sr,
        reference=text,
    )
    return entry, sr, ch


def _write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in sorted(entries, key=lambda e: (e.split, e.utt_id)):
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


# --- full / partial path: pinned parquet files -------------------------------------------------


def list_source_files(spec: SourceSpec) -> list[tuple[str, str | None]]:
    """Return [(path, lfs_sha256)] of parquet files for the pinned revision and HF split."""
    from huggingface_hub import HfApi

    files = []
    for item in HfApi().list_repo_tree(spec.repo_id, repo_type="dataset", revision=spec.revision, recursive=True):
        path = getattr(item, "path", "")
        lfs = getattr(item, "lfs", None)
        if path.endswith(".parquet") and PurePosixPath(path).name.startswith(f"{spec.hf_split}-"):
            files.append((path, lfs.sha256 if lfs is not None else None))
    if not files:
        raise DataError(f"no parquet files for {spec.repo_id}@{spec.revision}")
    return sorted(files)


def download_verified(spec: SourceSpec, filename: str, expected_sha: str | None, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local = Path(hf_hub_download(spec.repo_id, filename, repo_type="dataset", revision=spec.revision,
                                 cache_dir=str(cache_dir)))
    if expected_sha and sha256_file(local) != expected_sha:
        raise DataError(f"sha256 mismatch for {spec.repo_id}/{filename}")
    return local


def iter_parquet_rows(path: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=64, columns=["audio", "text", "original_audio_path"]):
        yield from batch.to_pylist()


def _audio_bytes(audio_field) -> bytes:
    if isinstance(audio_field, dict) and audio_field.get("bytes"):
        return audio_field["bytes"]
    raise ValueError("audio field has no embedded bytes")


def prepare_split(spec: SourceSpec, data_dir: Path, limit: int | None = None, log=print) -> dict:
    """Download, verify, decode and write 16 kHz mono PCM16 WAVs plus a JSONL manifest."""
    subset = "full" if limit is None else "partial"
    cache_dir = data_dir / "hf_cache"
    files = list_source_files(spec)
    check = RowCheck(spec, Counter(), [])
    entries: list[ManifestEntry] = []
    source_srs: Counter = Counter()
    channels_seen: Counter = Counter()
    markup = 0
    row_index = 0
    used_files = []

    for filename, sha in files:
        if limit is not None and row_index >= limit:
            break
        log(f"[{spec.split}] downloading {filename}")
        local = download_verified(spec, filename, sha, cache_dir)
        used_files.append({"path": filename, "sha256": sha})
        for row in iter_parquet_rows(local):
            if limit is not None and row_index >= limit:
                break
            current = row_index
            row_index += 1
            orig = row.get("original_audio_path") or ""
            text = (row.get("text") or "").strip()
            utt_id = check.validate(current, orig, text)
            if utt_id is None:
                continue
            try:
                entry, sr, ch = _write_wav_entry(spec, data_dir, utt_id, current, orig, text,
                                                 _audio_bytes(row.get("audio")))
            except Exception as exc:  # noqa: BLE001 - record and continue
                check.issues.append({"row": current, "utt_id": utt_id, "issue": "decode_error", "value": str(exc)})
                continue
            source_srs[sr] += 1
            channels_seen[ch] += 1
            markup += has_kspon_markup(text)
            entries.append(entry)

    mpath = manifest_path(data_dir, subset, spec.split)
    _write_manifest(mpath, entries)
    complete = limit is None
    return {
        "subset": subset,
        "split": spec.split,
        "source_repo": spec.repo_id,
        "source_revision": spec.revision,
        "hf_split": spec.hf_split,
        "files": used_files,
        "rows_read": row_index,
        "entries": len(entries),
        "expected_rows": spec.expected_rows,
        "row_count_ok": (row_index == spec.expected_rows and len(entries) == spec.expected_rows) if complete else None,
        "partial": not complete,
        "total_audio_hours": round(sum(e.duration_s for e in entries) / 3600, 4),
        "source_sample_rates": dict(source_srs),
        "source_channels": dict(channels_seen),
        "references_with_markup_chars": markup,
        "issues": check.issues,
        "manifest": mpath.relative_to(data_dir).as_posix(),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# --- smoke path: datasets-server rows API -------------------------------------------------------


def _http_get(client, url: str, params: dict | None = None, attempts: int = 5):
    delay = 2.0
    for attempt in range(attempts):
        try:
            r = client.get(url, params=params)
        except Exception:  # noqa: BLE001 - network error, retry
            if attempt == attempts - 1:
                raise
        else:
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise DataError(f"GET {url} returned HTTP {r.status_code}")
        time.sleep(delay)
        delay = min(delay * 2, 30)
    raise DataError(f"GET {url} failed")


def check_revision(spec: SourceSpec, allow_drift: bool) -> str:
    """Return the current Hub revision; fail when it differs from the pinned one."""
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(spec.repo_id).sha
    if sha != spec.revision and not allow_drift:
        raise DataError(
            f"{spec.repo_id} is now at {sha}, pinned {spec.revision}. The rows API serves the latest revision. "
            "Use `prepare-data --subset full` (pinned download) or pass --allow-revision-drift."
        )
    return sha


def _audio_src(row: dict) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and audio:
        audio = audio[0]
    if isinstance(audio, dict) and audio.get("src"):
        return audio["src"]
    raise ValueError("row has no audio src")


def fetch_row_index(spec: SourceSpec, client, log=print) -> list[dict]:
    """Read all row metadata (path, text, audio URL) of a split through the rows API."""
    rows: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        r = _http_get(client, ROWS_API, {"dataset": spec.repo_id, "config": "default", "split": spec.hf_split,
                                         "offset": offset, "length": ROWS_PAGE})
        page = r.json()
        total = int(page["num_rows_total"])
        if not page.get("rows"):
            break
        for item in page["rows"]:
            rows.append({"row_idx": int(item["row_idx"]), **item["row"]})
        offset += ROWS_PAGE
        if offset % 1000 == 0:
            log(f"[{spec.split}] indexed {offset}/{total} rows")
    if total != spec.expected_rows or len(rows) != total:
        raise DataError(f"{spec.repo_id}: rows API returned {len(rows)}/{total} rows, expected {spec.expected_rows}")
    return rows


def prepare_smoke(data_dir: Path, allow_revision_drift: bool = False, log=print) -> dict:
    import httpx

    summary: dict = {"subset": "smoke", "salt": SMOKE_SALT, "per_split": SMOKE_PER_SPLIT, "splits": {}}
    entries: list[ManifestEntry] = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for split, n in SMOKE_PER_SPLIT.items():
            spec = SOURCES[split]
            hub_rev = check_revision(spec, allow_revision_drift)
            rows = fetch_row_index(spec, client, log=log)
            check = RowCheck(spec, Counter(), [])
            valid = []
            for row in rows:
                orig = row.get("original_audio_path") or ""
                text = (row.get("text") or "").strip()
                utt_id = check.validate(row["row_idx"], orig, text)
                if utt_id is not None:
                    valid.append((utt_id, row, text))
            picked = pick_smoke(valid, lambda _: split, lambda x: x[0], {split: n})
            for utt_id, row, text in picked:
                src = _audio_src(row)
                if f"/--/{spec.revision}/--/" not in src and not allow_revision_drift:
                    raise DataError(f"{spec.repo_id}: audio URL is not from pinned revision {spec.revision}")
                audio = _http_get(client, src).content
                entry, _, _ = _write_wav_entry(spec, data_dir, utt_id, row["row_idx"], row["original_audio_path"],
                                               text, audio)
                entries.append(entry)
                log(f"[{split}] {utt_id} {entry.duration_s:.2f}s")
            summary["splits"][split] = {
                "source_repo": spec.repo_id,
                "pinned_revision": spec.revision,
                "hub_revision": hub_rev,
                "revision_drift": hub_rev != spec.revision,
                "rows_indexed": len(rows),
                "valid_rows": len(valid),
                "issues": check.issues,
                "utt_ids": [u for u, _, _ in picked],
            }
    mpath = manifest_path(data_dir, "smoke")
    _write_manifest(mpath, entries)
    summary.update({
        "entries": len(entries),
        "total_audio_seconds": round(sum(e.duration_s for e in entries), 3),
        "manifest": mpath.relative_to(data_dir).as_posix(),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    spath = summary_path(data_dir, "smoke")
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# --- entry points ---------------------------------------------------------------------------


def prepare(data_dir: Path, subset: str = "full", splits=SPLITS, limit: int | None = None,
            allow_revision_drift: bool = False, log=print) -> dict:
    if subset == "smoke":
        return prepare_smoke(data_dir, allow_revision_drift=allow_revision_drift, log=log)
    if subset != "full":
        raise ValueError("subset must be 'smoke' or 'full' (use limit for a partial full download)")
    summaries = {s: prepare_split(SOURCES[s], data_dir, limit=limit, log=log) for s in splits}
    out = summary_path(data_dir, "full" if limit is None else "partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing.update(summaries)
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def _read_jsonl(path: Path) -> list[ManifestEntry]:
    with open(path, encoding="utf-8") as f:
        return [ManifestEntry.from_dict(json.loads(line)) for line in f if line.strip()]


def load_manifest(data_dir: Path, subset: str = "full", splits=SPLITS) -> list[ManifestEntry]:
    """Load manifest entries. A full manifest must have passed its row-count check."""
    if subset == "smoke":
        p = manifest_path(data_dir, "smoke")
        if not p.exists():
            raise FileNotFoundError(f"manifest missing: {p} (run `kstt prepare-data --subset smoke`)")
        return [e for e in _read_jsonl(p) if e.split in splits]
    if subset not in ("full", "partial"):
        raise ValueError(f"unknown subset {subset!r}")
    summary_file = summary_path(data_dir, subset)
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    entries = []
    for s in splits:
        p = manifest_path(data_dir, subset, s)
        if not p.exists():
            raise FileNotFoundError(f"manifest missing: {p} (run `kstt prepare-data --subset full`)")
        if subset == "full" and summary.get(s, {}).get("row_count_ok") is not True:
            raise DataError(f"{s}: full manifest failed the row-count check; see {summary_file}")
        entries.extend(_read_jsonl(p))
    return entries


def dataset_revisions(entries: Sequence[ManifestEntry]) -> dict[str, str]:
    return {e.source_repo: e.source_revision for e in entries}
