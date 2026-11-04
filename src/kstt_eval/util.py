"""Small shared helpers: JSON output, timestamps, provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id(prefix: str = "") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}" if prefix else stamp


def json_safe(value: Any) -> Any:
    """Replace NaN/inf with ``None`` recursively so output is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False, **kwargs)


def write_json(path: Path, value: Any) -> None:
    """Write via a temporary file, fsync, then replace, so a crash leaves the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(dumps(value, indent=2))
        f.flush()
        os.fsync(f.fileno())
    # On Windows, replace fails while another process (status, report) has the file open.
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


class CorruptLogError(RuntimeError):
    """A JSONL log has a damaged line before its last line."""


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write((dumps(record) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path, *, repair: bool = False,
               warn: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Read a JSONL log written by :func:`append_jsonl`.

    A damaged last line (power loss during a write) is skipped with a warning. With ``repair`` the
    file is also cut back to the end of the last good line, so the next append starts on a clean
    line. Only the process that writes the log may repair it. A damaged earlier line raises
    :class:`CorruptLogError`.
    """
    if not path.exists():
        return []
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    missing_newline = False
    pos = 0
    bad: tuple[int, int] | None = None
    lineno = 0
    while pos < len(data):
        nl = data.find(b"\n", pos)
        end = len(data) if nl < 0 else nl + 1
        # Power loss can leave NUL padding; treat it like blank space.
        line = data[pos:end].strip(b" \t\r\n\x00")
        lineno += 1
        if line:
            if bad is not None:
                raise CorruptLogError(f"{path}: line {bad[0]} is damaged and is not the last line")
            try:
                records.append(json.loads(line.decode("utf-8")))
                missing_newline = nl < 0
            except (UnicodeDecodeError, ValueError):
                bad = (lineno, pos)
        pos = end
    if bad is not None:
        if warn:
            warn(f"{path.name}: skipped a damaged last line (line {bad[0]})")
        if repair:
            with open(path, "r+b") as f:
                f.truncate(bad[1])
                f.flush()
                os.fsync(f.fileno())
    elif missing_newline and repair:
        with open(path, "ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
    return records


def seed_order(seed: str, ids: Iterable[str]) -> list[str]:
    """Deterministic pseudo-random order: sort by ``sha256(seed:id)``."""
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def package_versions(names: tuple[str, ...] = (
    "kstt-eval", "httpx", "websockets", "azure-identity", "azure-cognitiveservices-speech", "numpy", "scipy",
    "soundfile", "huggingface-hub",
)) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions
