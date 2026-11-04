import asyncio
import email.utils
import time

import httpx
import pytest

from kstt_eval import auth
from kstt_eval.adapters import create_adapter
from kstt_eval.adapters.base import parse_retry_after, raise_for_response
from kstt_eval.auth import auth_headers, redact
from kstt_eval.config import ModelConfig
from kstt_eval.types import ConfigError, PermanentError, TransientError


def test_parse_retry_after_variants():
    assert parse_retry_after(httpx.Headers({"retry-after-ms": "1500"})) == 1.5
    assert parse_retry_after(httpx.Headers({"Retry-After": "3"})) == 3.0
    past = email.utils.formatdate(time.time() - 60, usegmt=True)
    assert parse_retry_after(httpx.Headers({"Retry-After": past})) == 0.0
    assert parse_retry_after(httpx.Headers({"Retry-After": "soon"})) is None
    assert parse_retry_after(httpx.Headers({})) is None
    assert parse_retry_after(None) is None


def test_raise_for_response_classification():
    raise_for_response(httpx.Response(200))
    with pytest.raises(TransientError) as e:
        raise_for_response(httpx.Response(429, headers={"retry-after": "2"},
                                          json={"error": {"code": "rate_limit_exceeded"}}))
    assert (e.value.status, e.value.code, e.value.retry_after_s, e.value.ambiguous) == (
        429, "rate_limit_exceeded", 2.0, False)
    with pytest.raises(TransientError):
        raise_for_response(httpx.Response(503, text="busy"))
    with pytest.raises(PermanentError) as e:
        raise_for_response(httpx.Response(401, json={"error": {"code": "Unauthorized",
                                                               "message": "token Bearer abc.def.ghi"}}))
    assert e.value.code == "Unauthorized"
    assert "abc.def.ghi" not in str(e.value)


def _cfg(**params):
    return ModelConfig(key="t", adapter="fast", track="A", params=params, model_id="m")


def test_signature_ignores_execution_params():
    base = create_adapter(_cfg(endpoint="https://x.invalid", locale="ko-KR", timeout_s=10, concurrency=4))
    tuned = create_adapter(_cfg(endpoint="https://x.invalid", locale="ko-KR", timeout_s=99, concurrency=1,
                                min_interval_s=5, max_retries=9))
    other = create_adapter(_cfg(endpoint="https://x.invalid", locale="en-US", timeout_s=10))
    assert base.signature() == tuned.signature()
    assert base.signature() != other.signature()


def test_unknown_adapter():
    with pytest.raises(ConfigError):
        create_adapter(ModelConfig(key="t", adapter="nope", track="A", params={}))


def test_redact():
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlX3ZhbHVl"
    text = (f"Authorization: Bearer abc.DEF-123 jwt={jwt} url=https://x/y?sig=SECRET1&a=1 "
            '{"api-key": "SECRET2"} key=SECRET3')
    out = redact(text)
    for secret in ("abc.DEF-123", jwt, "SECRET1", "SECRET2", "SECRET3"):
        assert secret not in out
    assert "a=1" in out


def test_auth_headers_api_key_fallback(monkeypatch):
    monkeypatch.setenv("KSTT_TEST_KEY", "k1")
    assert asyncio.run(auth_headers({"api_key_env": "KSTT_TEST_KEY"}, "Ocp-Apim-Subscription-Key")) == {
        "Ocp-Apim-Subscription-Key": "k1"}
    monkeypatch.setenv("KSTT_TEST_KEY", "")
    with pytest.raises(RuntimeError):
        asyncio.run(auth_headers({"api_key_env": "KSTT_TEST_KEY"}, "api-key"))


def test_cached_credential_reuses_token(monkeypatch):
    calls = []

    class FakeCred:
        def get_token(self, scope):
            calls.append(scope)
            return type("T", (), {"token": "tok-" + str(len(calls)), "expires_on": time.time() + 3600})()

    monkeypatch.setattr(auth, "_credential", FakeCred())
    monkeypatch.setattr(auth, "_tokens", {})
    cred = auth.cached_credential()
    first = cred.get_token("scope-a")
    assert first.token == "tok-1" and first.expires_on > time.time()
    assert cred.get_token("scope-a").token == "tok-1"
    assert cred.get_token_info("scope-a").token == "tok-1"
    assert auth.get_token("scope-a") == "tok-1"
    assert calls == ["scope-a"]
    assert cred.get_token("scope-b").token == "tok-2"

    # a token within 5 minutes of expiry is refreshed
    monkeypatch.setattr(auth, "_tokens", {"scope-a": ("old", time.time() + 60)})
    assert cred.get_token("scope-a").token == "tok-3"


def test_concurrent_token_misses_fetch_once(monkeypatch):
    import threading

    calls = []

    class SlowCred:
        def get_token(self, scope):
            calls.append(scope)
            time.sleep(0.2)
            return type("T", (), {"token": "tok", "expires_on": time.time() + 3600})()

    monkeypatch.setattr(auth, "_credential", SlowCred())
    monkeypatch.setattr(auth, "_tokens", {})
    got = []
    threads = [threading.Thread(target=lambda: got.append(auth.get_token("s"))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == ["tok"] * 4 and calls == ["s"]
