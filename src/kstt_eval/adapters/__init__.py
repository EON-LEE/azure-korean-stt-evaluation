"""Adapter registry. Adapter modules are imported lazily so optional SDKs stay optional."""

from __future__ import annotations

import importlib

from ..config import ModelConfig
from ..types import ConfigError
from .base import Adapter

REGISTRY: dict[str, tuple[str, str]] = {
    "fast": ("fast", "FastAdapter"),
    "openai_file": ("openai_file", "OpenAIFileAdapter"),
    "openai_realtime": ("realtime", "OpenAIRealtimeAdapter"),
    "mai_realtime": ("realtime", "MAIRealtimeAdapter"),
    "voicelive": ("realtime", "VoiceLiveAdapter"),
    "speech_sdk": ("speech_sdk", "SpeechSDKAdapter"),
}


def create_adapter(cfg: ModelConfig) -> Adapter:
    if cfg.adapter not in REGISTRY:
        raise ConfigError(f"{cfg.key}: unknown adapter {cfg.adapter!r}; known: {sorted(REGISTRY)}")
    module, cls = REGISTRY[cfg.adapter]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(cfg)


__all__ = ["Adapter", "REGISTRY", "create_adapter"]
