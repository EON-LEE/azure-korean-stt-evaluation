import os
import re
from pathlib import Path

import pytest

from kstt_eval.adapters import REGISTRY
from kstt_eval.config import load_env_file, load_experiments, load_models, select_routes
from kstt_eval.types import ConfigError

ROOT = Path(__file__).resolve().parents[1]

MODELS_YAML = """
defaults:
  params: {timeout_s: 10, concurrency: 4}
models:
  r-a:
    adapter: fast
    track: A
    params:
      endpoint: ${KSTT_TEST_EP}
      definition: {locales: [ko-KR]}
  r-b:
    adapter: openai_file
    track: B
    params:
      endpoint: ${KSTT_TEST_MISSING}
      concurrency: 1
      prompt: "고유명사 목록"
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_env_expansion_defaults_and_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSTT_TEST_EP", "https://example.invalid")
    monkeypatch.delenv("KSTT_TEST_MISSING", raising=False)
    models = load_models(_write(tmp_path, MODELS_YAML))
    assert models["r-a"].params["endpoint"] == "https://example.invalid"
    assert models["r-a"].params["timeout_s"] == 10 and models["r-a"].ready
    assert models["r-b"].params["concurrency"] == 1
    assert models["r-b"].missing_env == {"KSTT_TEST_MISSING"} and not models["r-b"].ready


def test_baseline_experiment_rejects_prompt_keys(tmp_path):
    exp = tmp_path / "experiments.yaml"
    exp.write_text("experiments:\n  baseline: {allow_prompts: false}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="prompt"):
        load_models(_write(tmp_path, MODELS_YAML), load_experiments(exp)["baseline"])


def test_select_routes(tmp_path):
    models = load_models(_write(tmp_path, MODELS_YAML))
    assert [m.key for m in select_routes(models, "all")] == ["r-a", "r-b"]
    assert [m.key for m in select_routes(models, "track:B")] == ["r-b"]
    assert [m.key for m in select_routes(models, "r-*,r-a")] == ["r-a", "r-b"]
    with pytest.raises(ConfigError):
        select_routes(models, "nope-*")


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / "x.env"
    env.write_text("# comment\nKSTT_T1=one\nKSTT_T2='two'\n\n", encoding="utf-8")
    monkeypatch.setenv("KSTT_T1", "keep")
    monkeypatch.setenv("KSTT_T2", "")  # empty counts as unset; monkeypatch restores it afterwards
    assert load_env_file(env) == ["KSTT_T2"]
    assert os.environ["KSTT_T1"] == "keep" and os.environ["KSTT_T2"] == "two"
    assert load_env_file(tmp_path / "absent.env") == []
    bad = tmp_path / "bad.env"
    bad.write_text("NOEQUALS\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_env_file(bad)


def test_repository_configs_load_and_have_no_endpoints():
    text = (ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
    for marker in (".cognitiveservices.azure.com", ".openai.azure.com", ".services.ai.azure.com"):
        assert marker not in text, "real endpoints belong in configs/endpoints.env, not models.yaml"
    experiments = load_experiments(ROOT / "configs" / "experiments.yaml")
    models = load_models(ROOT / "configs" / "models.yaml", experiments["baseline"])
    assert len(models) == 29
    for m in models.values():
        assert m.adapter in REGISTRY, m.key
        assert m.model_id and m.region, m.key
    assert models["aoai-whisper"].params["concurrency"] == 1
    example = (ROOT / "configs" / "endpoints.env.example").read_text(encoding="utf-8")
    for name in set(re.findall(r"\$\{(KSTT_[A-Z0-9_]+)", text)):
        assert f"{name}=" in example, name
