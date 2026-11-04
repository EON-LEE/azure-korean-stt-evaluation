"""Realtime WebSocket transcription: Azure OpenAI Realtime, MAI-Transcribe streaming, Voice Live.

Audio is streamed at real-time pace in fixed chunks. A reader task feeds every server event to a
:class:`TranscriptAccumulator`, which decides when the final transcript is complete.

Modes:

* ``commit``: no server VAD. The client commits once after the audio; the result is the item(s)
  committed after that commit.
* ``vad``: server VAD segments the audio. Trailing silence lets VAD close the last turn. If VAD
  has not closed the turn ``vad_flush_after_s`` after the audio ends, the client commits once and
  records ``meta.forced_commit``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections import Counter
from typing import Any

from ..audio import pcm_chunks, read_pcm16, silence_pcm16
from ..auth import redact
from ..types import AudioInput, PermanentError, Transcription, TransientError
from .base import HttpAdapter, check_aoai_deployment, now, parse_retry_after, require

PREFIX = "conversation.item.input_audio_transcription."
TRANSIENT_ERROR_CODES = {"rate_limit_exceeded", "server_error", "too_many_requests", "service_unavailable",
                         "internal_error", "timeout"}


class RealtimeServiceError(Exception):
    """An ``error`` event from the service."""

    def __init__(self, error: dict[str, Any]):
        self.error = error or {}
        super().__init__(redact(json.dumps(self.error, ensure_ascii=False)[:500]))

    @property
    def code(self) -> str | None:
        return self.error.get("code")

    @property
    def transient(self) -> bool:
        return self.code in TRANSIENT_ERROR_CODES or self.error.get("type") == "server_error"


class TranscriptAccumulator:
    """Collect transcription events and decide when the transcript is final."""

    def __init__(self) -> None:
        self.committed: list[str] = []
        self.completed: dict[str, str] = {}
        self.completed_order: list[str] = []
        self.failed: dict[str, Any] = {}
        self.speaking = False
        self.speech_stopped = 0
        self.commit_empty = 0
        self.response_events = 0
        self.partials = 0
        self.events: Counter = Counter()
        self.first_partial_t: float | None = None
        self.first_final_t: float | None = None
        self.last_final_t: float | None = None
        self.commit_base: int | None = None
        self.commit_empty_base = 0
        self.completed_after_commit = 0
        self._anon = 0

    def mark_commit(self) -> None:
        """Call immediately before the client sends ``input_audio_buffer.commit``."""
        self.commit_base = len(self.committed)
        self.commit_empty_base = self.commit_empty
        self.completed_after_commit = 0

    def _item(self, ev: dict[str, Any]) -> str:
        item = ev.get("item_id")
        if item:
            return item
        self._anon += 1
        return f"_anon{self._anon}"

    def feed(self, ev: dict[str, Any], t: float) -> None:
        typ = ev.get("type", "")
        self.events[typ] += 1
        if typ == "error":
            err = ev.get("error") or {}
            if err.get("code") == "input_audio_buffer_commit_empty":
                self.commit_empty += 1
                return
            raise RealtimeServiceError(err)
        if typ == "input_audio_buffer.committed":
            item = self._item(ev)
            if item not in self.committed:
                self.committed.append(item)
        elif typ == "input_audio_buffer.speech_started":
            self.speaking = True
        elif typ == "input_audio_buffer.speech_stopped":
            self.speaking = False
            self.speech_stopped += 1
        elif typ in (PREFIX + "delta", PREFIX + "intermediate"):
            self.partials += 1
            if self.first_partial_t is None:
                self.first_partial_t = t
        elif typ == PREFIX + "completed":
            item = self._item(ev)
            if item not in self.completed:
                self.completed_order.append(item)
            self.completed[item] = ev.get("transcript") or ""
            if self.first_final_t is None:
                self.first_final_t = t
            self.last_final_t = t
            if self.commit_base is not None:
                self.completed_after_commit += 1
        elif typ == PREFIX + "failed":
            self.failed[self._item(ev)] = ev.get("error")
        elif typ.startswith("response."):
            self.response_events += 1

    def pending(self) -> list[str]:
        return [i for i in self.committed if i not in self.completed and i not in self.failed]

    def done(self) -> bool:
        if self.pending():
            return False
        if self.commit_base is not None:
            if len(self.committed) > self.commit_base:
                return True
            if not self.committed and self.completed_after_commit:
                return True  # service that sends no committed events (completed only)
            return self.commit_empty > self.commit_empty_base
        return (not self.speaking) and len(self.committed) >= 1 and len(self.committed) >= self.speech_stopped

    def segments(self) -> list[str]:
        seen = set(self.committed)
        order = [i for i in self.committed if i in self.completed]
        order += [i for i in self.completed_order if i not in seen]
        return [self.completed[i] for i in order]

    def text(self) -> str:
        return " ".join(s.strip() for s in self.segments() if s and s.strip())


def _ws_url(endpoint_or_host: str) -> str:
    host = endpoint_or_host.strip().rstrip("/")
    for scheme in ("https://", "http://", "wss://"):
        if host.startswith(scheme):
            host = host[len(scheme):]
    return "wss://" + host


class RealtimeAdapter(HttpAdapter):
    """Common WebSocket streaming flow. Subclasses define the URL and the session.update payload."""

    kind = "realtime"
    input_rate = 16000
    modes = ("commit", "vad")

    @property
    def mode(self) -> str:
        mode = self.params.get("mode", "commit")
        if mode not in self.modes:
            raise PermanentError(f"{self.cfg.key}: mode must be one of {self.modes}, got {mode!r}")
        return mode

    def url(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def session_update(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    async def _connect(self):
        from websockets.asyncio.client import connect

        return connect(self.url(), additional_headers=await self.headers(), max_size=None,
                       open_timeout=float(self.params.get("connect_timeout_s", 30)))

    @staticmethod
    async def _wait_for(ws, kind: str, timeout: float) -> dict[str, Any]:
        async def loop() -> dict[str, Any]:
            while True:
                ev = json.loads(await ws.recv())
                if ev.get("type") == "error":
                    raise RealtimeServiceError(ev.get("error") or {})
                if ev.get("type") == kind:
                    return ev

        return await asyncio.wait_for(loop(), timeout)

    @staticmethod
    async def _read(ws, acc: TranscriptAccumulator, changed: asyncio.Event) -> None:
        from websockets.exceptions import ConnectionClosedOK

        try:
            async for msg in ws:
                acc.feed(json.loads(msg), now())
                changed.set()
        except ConnectionClosedOK:
            return
        finally:
            changed.set()

    @staticmethod
    async def _send_paced(ws, chunks: list[bytes], chunk_ms: int, reader: asyncio.Task) -> None:
        t0 = now()
        for i, chunk in enumerate(chunks):
            delay = t0 + i * chunk_ms / 1000 - now()
            if delay > 0:
                await asyncio.sleep(delay)
            if reader.done():
                reader.result()  # raise a service error now instead of streaming the rest
                raise TransientError("connection closed while sending audio", ambiguous=True)
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(chunk).decode()}))

    async def transcribe(self, audio: AudioInput) -> Transcription:
        mode = self.mode
        chunk_ms = int(self.params.get("chunk_ms", 20))
        pcm, sr = await asyncio.to_thread(read_pcm16, audio.path, self.input_rate)
        speech_chunks = pcm_chunks(pcm, sr, chunk_ms)
        trailing_s = float(self.params.get("trailing_silence_s", 1.0 if mode == "vad" else 0.0))
        silence_chunks = pcm_chunks(silence_pcm16(sr, trailing_s), sr, chunk_ms) if trailing_s > 0 else []
        connect_timeout = float(self.params.get("connect_timeout_s", 30))
        final_timeout = float(self.params.get("final_timeout_s", 60))
        flush_after = float(self.params.get("vad_flush_after_s", 3.0))
        commit = json.dumps({"type": "input_audio_buffer.commit"})

        acc = TranscriptAccumulator()
        meta: dict[str, Any] = {"mode": mode, "forced_commit": False}
        audio_started = False
        try:
            t_connect = now()
            async with await self._connect() as ws:
                await self._wait_for(ws, "session.created", connect_timeout)
                connect_s = now() - t_connect
                await ws.send(json.dumps(self.session_update()))
                await self._wait_for(ws, "session.updated", connect_timeout)
                changed = asyncio.Event()
                reader = asyncio.create_task(self._read(ws, acc, changed))
                try:
                    audio_started = True
                    t0 = now()
                    await self._send_paced(ws, speech_chunks, chunk_ms, reader)
                    speech_end = now()
                    await self._send_paced(ws, silence_chunks, chunk_ms, reader)
                    send_end = now()
                    if mode == "commit":
                        acc.mark_commit()
                        await ws.send(commit)
                    deadline = send_end + final_timeout
                    flush_at = send_end + flush_after if mode == "vad" else None
                    while True:
                        changed.clear()
                        if reader.done() and not reader.cancelled() and reader.exception():
                            reader.result()
                        if acc.done():
                            break
                        if reader.done():
                            raise TransientError("connection closed before the final transcript", ambiguous=True)
                        if (flush_at is not None and acc.commit_base is None and now() >= flush_at
                                and (acc.speaking or not acc.committed)):
                            meta["forced_commit"] = True
                            acc.mark_commit()
                            await ws.send(commit)
                        remaining = deadline - now()
                        if remaining <= 0:
                            raise TransientError(
                                f"timed out after {final_timeout}s waiting for the final transcript "
                                f"(committed={len(acc.committed)} completed={len(acc.completed)})",
                                code="final_timeout", ambiguous=True)
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(changed.wait(), timeout=min(remaining, 0.25))
                finally:
                    reader.cancel()
                    with contextlib.suppress(BaseException):
                        await reader
        except RealtimeServiceError as exc:
            if exc.transient:
                raise TransientError(str(exc), code=exc.code, ambiguous=audio_started) from exc
            raise PermanentError(str(exc), code=exc.code) from exc
        except (TransientError, PermanentError):
            raise
        except Exception as exc:  # noqa: BLE001 - classify websocket/network failures
            raise self._classify(exc, audio_started) from exc

        if acc.failed:
            raise TransientError(redact(f"transcription failed for items: {acc.failed}")[:500],
                                 code="transcription_failed", ambiguous=True)
        latency = {"connect_s": connect_s, "audio_send_s": speech_end - t0}
        if acc.first_partial_t is not None:
            latency["first_partial_s"] = acc.first_partial_t - t0
        if acc.first_final_t is not None:
            latency["first_final_s"] = acc.first_final_t - t0
        if acc.last_final_t is not None:
            latency["final_after_audio_end_s"] = acc.last_final_t - speech_end
        segments = acc.segments()
        meta.update({"segments": len(segments), "response_events": acc.response_events,
                     "commit_empty": acc.commit_empty, "trailing_silence_s": trailing_s})
        return Transcription(
            text=acc.text(),
            raw={"segments": segments, "events": dict(acc.events)},
            latency=latency,
            partials=acc.partials,
            meta=meta,
        )

    @staticmethod
    def _classify(exc: Exception, audio_started: bool) -> Exception:
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        if isinstance(exc, InvalidStatus):
            status = exc.response.status_code
            body = exc.response.body or b""
            message = redact(f"WebSocket handshake HTTP {status}: {body[:300].decode('utf-8', 'replace')}")
            if status in (408, 429) or status >= 500:
                return TransientError(message, status=status, retry_after_s=parse_retry_after(exc.response.headers))
            return PermanentError(message, status=status)
        if isinstance(exc, ConnectionClosed):
            return TransientError(redact(f"connection closed: {exc}"), ambiguous=audio_started)
        if isinstance(exc, (OSError, asyncio.TimeoutError)):
            return TransientError(redact(f"{type(exc).__name__}: {exc}"), ambiguous=audio_started)
        return TransientError(redact(f"{type(exc).__name__}: {exc}"), ambiguous=audio_started)

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        checks = await super().doctor(ws)
        if not ws:
            return checks + [{"name": "websocket", "ok": True, "detail": "skipped (use --ws)"}]
        try:
            t = now()
            async with await self._connect() as conn:
                await self._wait_for(conn, "session.created", float(self.params.get("connect_timeout_s", 30)))
                await conn.send(json.dumps(self.session_update()))
                await self._wait_for(conn, "session.updated", float(self.params.get("connect_timeout_s", 30)))
            checks.append({"name": "websocket", "ok": True, "detail": f"session.updated in {now() - t:.2f}s"})
        except Exception as exc:  # noqa: BLE001
            err = exc if isinstance(exc, (TransientError, PermanentError)) else self._classify(exc, False)
            if isinstance(exc, RealtimeServiceError):
                err = exc
            checks.append({"name": "websocket", "ok": False, "detail": str(err)})
        return checks


class OpenAIRealtimeAdapter(RealtimeAdapter):
    """Azure OpenAI GA Realtime transcription session (``/openai/v1/realtime?intent=transcription``)."""

    kind = "openai_realtime"
    input_rate = 24000

    def url(self) -> str:
        require(self.params, "endpoint", "deployment")
        return _ws_url(self.params["endpoint"]) + "/openai/v1/realtime?intent=transcription"

    def session_update(self) -> dict[str, Any]:
        transcription = {"model": self.params["deployment"], "language": self.params.get("language", "ko")}
        turn = {"type": "server_vad"} if self.mode == "vad" else None
        return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": self.input_rate},
            "transcription": transcription,
            "turn_detection": turn,
        }}}}

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        checks = await super().doctor(ws)
        return checks + [await check_aoai_deployment(self, self.params["endpoint"], self.params["deployment"],
                                                     self.cfg.model_id)]


class MAIRealtimeAdapter(RealtimeAdapter):
    """MAI-Transcribe streaming (``/mai/v1/realtime?intent=transcription``). Manual commit only."""

    kind = "mai_realtime"
    input_rate = 16000
    modes = ("commit",)

    def url(self) -> str:
        require(self.params, "ai_host", "deployment")
        return _ws_url(self.params["ai_host"]) + "/mai/v1/realtime?intent=transcription"

    def session_update(self) -> dict[str, Any]:
        return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": self.input_rate},
            "transcription": {"model": self.params["deployment"], "language": self.params.get("language", "ko")},
            "turn_detection": None,
            "noise_reduction": None,
        }}}}


class VoiceLiveAdapter(RealtimeAdapter):
    """Voice Live input-audio transcription. ``response.create`` is never sent."""

    kind = "voicelive"
    input_rate = 16000

    def url(self) -> str:
        require(self.params, "ai_host", "chat_model", "transcription_model")
        api_version = self.params.get("api_version", "2026-04-10")
        return (_ws_url(self.params["ai_host"])
                + f"/voice-live/realtime?api-version={api_version}&model={self.params['chat_model']}")

    def session_update(self) -> dict[str, Any]:
        turn = {"type": "server_vad", "create_response": False} if self.mode == "vad" else None
        return {"type": "session.update", "session": {
            "modalities": ["text"],
            "input_audio_sampling_rate": self.input_rate,
            "input_audio_transcription": {"model": self.params["transcription_model"],
                                          "language": self.params.get("language", "ko")},
            "turn_detection": turn,
        }}
