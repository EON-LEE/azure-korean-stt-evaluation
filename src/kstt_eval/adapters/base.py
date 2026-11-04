"""Adapter base classes, cache signature and HTTP error classification."""

from __future__ import annotations

import email.utils
import time
from typing import Any

import httpx

from ..auth import auth_headers, redact
from ..config import ModelConfig
from ..types import AudioInput, NetworkError, PermanentError, Transcription, TransientError

# Bump when adapter request/parse behaviour changes so cached results are not reused.
ADAPTER_VERSION = "1"

# Params that change how a request is executed but not what is sent or returned.
NON_SIGNATURE_PARAMS = frozenset({
    "timeout_s", "max_retries", "concurrency", "retry_ambiguous", "api_key_env", "auth_scope",
    "final_timeout_s", "connect_timeout_s", "min_interval_s", "max_attempts", "resource",
})


class Adapter:
    """One evaluated route. Subclasses implement :meth:`transcribe`."""

    kind = "base"
    key_header = "api-key"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.params: dict[str, Any] = cfg.params

    @property
    def key(self) -> str:
        return self.cfg.key

    def signature(self) -> dict[str, Any]:
        """Everything that can change the transcript. Used as the cache key."""
        return {
            "adapter": self.cfg.adapter,
            "adapter_version": ADAPTER_VERSION,
            "model_id": self.cfg.model_id,
            "params": {k: v for k, v in sorted(self.params.items()) if k not in NON_SIGNATURE_PARAMS},
        }

    async def headers(self) -> dict[str, str]:
        return await auth_headers(self.params, self.key_header)

    async def transcribe(self, audio: AudioInput) -> Transcription:  # pragma: no cover - abstract
        raise NotImplementedError

    def sent_audio_s(self, duration_s: float) -> float:
        """Seconds of audio one attempt sends. Used for the audio-hour budget."""
        return duration_s

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        """Non-billable readiness checks. Each check is ``{"name", "ok", "detail"}``."""
        await self.headers()
        return [{"name": "token", "ok": True, "detail": "Entra ID token acquired"}]

    async def aclose(self) -> None:
        return None


def parse_retry_after(headers: Any) -> float | None:
    """Return seconds from ``retry-after-ms`` or ``Retry-After`` (seconds or HTTP date)."""
    if headers is None:
        return None
    ms = headers.get("retry-after-ms") or headers.get("x-ms-retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return max(0.0, float(ra))
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(ra)
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _error_code(resp: httpx.Response) -> str | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return err.get("code") or err.get("type")
    if isinstance(body, dict):
        return body.get("code")
    return None


def raise_for_response(resp: httpx.Response) -> None:
    """429, 408 and 5xx are transient; other 4xx are permanent."""
    status = resp.status_code
    if status < 400:
        return
    message = redact(f"HTTP {status}: {resp.text[:500]}")
    code = _error_code(resp)
    if status in (408, 429) or status >= 500:
        raise TransientError(message, status=status, code=code, retry_after_s=parse_retry_after(resp.headers))
    raise PermanentError(message, status=status, code=code)


class HttpAdapter(Adapter):
    """Adapter that sends one HTTP request per utterance."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(float(self.params.get("timeout_s", 120)),
                                    connect=float(self.params.get("connect_timeout_s", 30)))
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = await self.client.request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            raise NetworkError(redact(f"{type(exc).__name__}: {exc}")) from exc
        except httpx.HTTPError as exc:
            # Read/write timeouts and dropped connections happen after the audio was sent.
            raise TransientError(redact(f"{type(exc).__name__}: {exc}"), ambiguous=True) from exc
        raise_for_response(resp)
        return resp

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def require(params: dict[str, Any], *names: str) -> None:
    missing = [n for n in names if not params.get(n)]
    if missing:
        raise PermanentError(f"missing params: {missing}")


def speech_models_check_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/speechtotext/models/base?api-version=2024-11-15&top=1"


def aoai_deployment_url(endpoint: str, deployment: str) -> str:
    return f"{endpoint.rstrip('/')}/openai/deployments/{deployment}?api-version=2022-12-01"


async def check_get(adapter: HttpAdapter, name: str, url: str) -> tuple[dict[str, Any], httpx.Response | None]:
    try:
        resp = await adapter.request("GET", url, headers=await adapter.headers())
    except (TransientError, PermanentError) as exc:
        return {"name": name, "ok": False, "detail": str(exc)}, None
    return {"name": name, "ok": True, "detail": f"HTTP {resp.status_code}"}, resp


async def check_aoai_deployment(adapter: HttpAdapter, endpoint: str, deployment: str,
                                model_id: str) -> dict[str, Any]:
    check, resp = await check_get(adapter, "aoai_deployment", aoai_deployment_url(endpoint, deployment))
    if resp is None:
        return check
    body = resp.json()
    model, status = body.get("model"), body.get("status")
    ok = status == "succeeded" and (not model_id or model == model_id)
    return {"name": "aoai_deployment", "ok": ok,
            "detail": f"deployment={deployment} model={model} status={status} expected_model={model_id or '-'}"}


def now() -> float:
    return time.perf_counter()
