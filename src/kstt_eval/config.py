"""Model and experiment configuration loading."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Keys that would inject vocabulary or context into a request. Baseline runs must not use them.
PROMPT_KEYS = {"prompt", "phraseList", "phrase_list", "phrases", "instructions"}


def expand_env(value: Any, missing: set[str]) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            v = os.environ.get(name)
            if v:
                return v
            if default is not None:
                return default
            missing.add(name)
            return m.group(0)

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, missing) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def find_prompt_keys(obj: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in PROMPT_KEYS and v not in (None, "", [], {}):
                found.append(p)
            found.extend(find_prompt_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(find_prompt_keys(v, f"{path}[{i}]"))
    return found


@dataclass
class ModelConfig:
    key: str
    adapter: str
    track: str
    params: dict[str, Any]
    missing_env: set[str] = field(default_factory=set)
    notes: str = ""
    price_usd_per_audio_hour: float | None = None
    region: str = ""
    model_id: str = ""
    live_verified: str | None = None

    @property
    def ready(self) -> bool:
        return not self.missing_env

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "adapter": self.adapter,
            "track": self.track,
            "region": self.region,
            "model_id": self.model_id,
            "params": self.params,
        }


def load_env_file(path: Path, override: bool = False) -> list[str]:
    """Load ``KEY=VALUE`` lines into ``os.environ``. Existing variables win unless ``override``.

    Returns the names that were set. Blank lines and ``#`` comments are ignored.
    """
    if not path.exists():
        return []
    loaded = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{lineno}: expected KEY=VALUE")
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigError(f"{path}:{lineno}: invalid variable name {name!r}")
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


def select_routes(models: dict[str, ModelConfig], spec: str) -> list[ModelConfig]:
    """Select routes by comma-separated keys or glob patterns (``all``, ``fast-*``, ``track:B``)."""
    import fnmatch

    chosen: list[str] = []
    for token in [t.strip() for t in spec.split(",") if t.strip()]:
        if token == "all":
            matched = list(models)
        elif token.startswith("track:"):
            matched = [k for k, m in models.items() if m.track == token.split(":", 1)[1]]
        else:
            matched = [k for k in models if fnmatch.fnmatchcase(k, token)]
        if not matched:
            raise ConfigError(f"no route matches {token!r}. Run `kstt models` to list routes.")
        chosen.extend(k for k in matched if k not in chosen)
    return [models[k] for k in chosen]


@dataclass
class Experiment:
    name: str
    description: str
    allow_prompts: bool
    overrides: dict[str, dict]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiments(path: Path) -> dict[str, Experiment]:
    raw = load_yaml(path).get("experiments", {})
    return {
        name: Experiment(
            name=name,
            description=spec.get("description", ""),
            allow_prompts=bool(spec.get("allow_prompts", False)),
            overrides=spec.get("overrides") or {},
        )
        for name, spec in raw.items()
    }


def load_models(path: Path, experiment: Experiment | None = None) -> dict[str, ModelConfig]:
    raw = load_yaml(path)
    defaults = raw.get("defaults") or {}
    models: dict[str, ModelConfig] = {}
    for key, spec in (raw.get("models") or {}).items():
        spec = deep_merge(defaults, spec)
        if experiment is not None:
            for pattern, override in experiment.overrides.items():
                if re.fullmatch(pattern, key):
                    spec = deep_merge(spec, override)
        missing: set[str] = set()
        params = expand_env(spec.get("params") or {}, missing)
        adapter = spec.get("adapter")
        track = spec.get("track")
        if not adapter or track not in ("A", "B"):
            raise ConfigError(f"{key}: 'adapter' and 'track' (A|B) are required")
        if experiment is not None and not experiment.allow_prompts:
            bad = find_prompt_keys(params)
            if bad:
                raise ConfigError(f"{key}: experiment '{experiment.name}' forbids prompt/phrase keys: {bad}")
        models[key] = ModelConfig(
            key=key,
            adapter=adapter,
            track=track,
            params=params,
            missing_env=missing,
            notes=spec.get("notes", ""),
            price_usd_per_audio_hour=spec.get("price_usd_per_audio_hour"),
            region=spec.get("region", ""),
            model_id=spec.get("model_id", ""),
            live_verified=spec.get("live_verified"),
        )
    return models
