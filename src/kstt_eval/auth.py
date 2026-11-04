"""Entra ID authentication and secret redaction.

The Azure CLI login (``az login``) is used by default because the evaluated resources disable
key authentication. Set ``KSTT_CREDENTIAL=default`` to use ``DefaultAzureCredential`` instead.
An API key env var (``api_key_env`` in model params) is an optional fallback.

Tokens are cached in memory only and are never written to run outputs.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

_credential = None
_lock = threading.Lock()
_fetch_lock = threading.Lock()
_tokens: dict[str, tuple[str, float]] = {}


def get_credential():
    global _credential
    with _lock:
        if _credential is None:
            mode = os.environ.get("KSTT_CREDENTIAL", "cli").strip().lower()
            if mode == "cli":
                from azure.identity import AzureCliCredential

                _credential = AzureCliCredential(process_timeout=30)
            elif mode == "default":
                from azure.identity import DefaultAzureCredential

                _credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            else:
                raise ValueError(f"KSTT_CREDENTIAL must be 'cli' or 'default', got {mode!r}")
        return _credential


def _get_access(scope: str) -> tuple[str, float]:
    def cached() -> tuple[str, float] | None:
        with _lock:
            hit = _tokens.get(scope)
            return hit if hit and hit[1] - time.time() > 300 else None

    hit = cached()
    if hit:
        return hit
    with _fetch_lock:  # concurrent cache misses wait for one fetch instead of each starting `az`
        hit = cached()
        if hit:
            return hit
        tok = get_credential().get_token(scope)
        with _lock:
            _tokens[scope] = (tok.token, float(tok.expires_on))
            return _tokens[scope]


def get_token(scope: str = COGNITIVE_SCOPE) -> str:
    return _get_access(scope)[0]


class CachedTokenCredential:
    """TokenCredential backed by the in-memory token cache.

    ``AzureCliCredential`` starts an ``az`` process on every call. SDKs that take a credential
    (the Speech SDK) would otherwise pay that cost on each connection.
    """

    def get_token(self, *scopes: str, **_kwargs: Any):
        from azure.core.credentials import AccessToken

        token, expires_on = _get_access(scopes[0] if scopes else COGNITIVE_SCOPE)
        return AccessToken(token, int(expires_on))

    def get_token_info(self, *scopes: str, options: Any = None):
        from azure.core.credentials import AccessTokenInfo

        token, expires_on = _get_access(scopes[0] if scopes else COGNITIVE_SCOPE)
        return AccessTokenInfo(token, int(expires_on))


def cached_credential() -> CachedTokenCredential:
    return CachedTokenCredential()


async def aget_token(scope: str = COGNITIVE_SCOPE) -> str:
    return await asyncio.to_thread(get_token, scope)


async def auth_headers(params: dict[str, Any], key_header: str) -> dict[str, str]:
    """Build auth headers from model params.

    ``api_key_env`` names an env var holding a key (sent in ``key_header``).
    Otherwise an Entra ID bearer token for ``auth_scope`` (default Cognitive Services) is used.
    """
    key_env = params.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"api_key_env {key_env} is set in config but the variable is empty")
        return {key_header: key}
    token = await aget_token(params.get("auth_scope", COGNITIVE_SCOPE))
    return {"Authorization": "Bearer " + token}


_REDACTIONS = [
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-_.~+/=]+"), r"\1[REDACTED]"),
    (re.compile(r"eyJ[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{4,}"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)\b(sig|access_token|api-key|subscription-key|ocp-apim-subscription-key|key)=([^&\s\"']+)"),
     r"\1=[REDACTED]"),
    (re.compile(r"(?i)(\"?(?:api-key|ocp-apim-subscription-key|authorization)\"?\s*[:=]\s*\"?)[^\"\s,}]+"),
     r"\1[REDACTED]"),
]


def redact(text: Any) -> str:
    """Remove bearer tokens, JWTs, keys and SAS signatures from text destined for outputs."""
    s = str(text)
    for pattern, repl in _REDACTIONS:
        s = pattern.sub(repl, s)
    return s
