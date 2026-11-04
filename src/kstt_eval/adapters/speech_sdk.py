"""Azure AI Speech SDK continuous recognition with audio pushed at real-time pace.

Requires the optional extra: ``uv sync --extra speech-sdk``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from ..audio import pcm_chunks, read_pcm16
from ..auth import cached_credential, redact
from ..types import AudioInput, PermanentError, Transcription, TransientError
from .base import HttpAdapter, check_get, now, require, speech_models_check_url

_TRANSIENT_CODES = {"TooManyRequests", "ConnectionFailure", "ServiceTimeout", "ServiceError", "ServiceUnavailable"}


def _sdk():
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise PermanentError("Speech SDK is not installed. Run `uv sync --extra speech-sdk`.") from exc
    return speechsdk


class SpeechSDKAdapter(HttpAdapter):
    kind = "speech_sdk"

    async def transcribe(self, audio: AudioInput) -> Transcription:
        require(self.params, "endpoint")
        return await asyncio.to_thread(self._run, audio)

    def _run(self, audio: AudioInput) -> Transcription:
        speechsdk = _sdk()
        chunk_ms = int(self.params.get("chunk_ms", 20))
        connect_timeout = float(self.params.get("connect_timeout_s", 30))
        final_timeout = float(self.params.get("final_timeout_s", 60))
        pcm, sr = read_pcm16(audio.path, 16000)
        chunks = pcm_chunks(pcm, sr, chunk_ms)

        cfg = speechsdk.SpeechConfig(token_credential=cached_credential(), endpoint=self.params["endpoint"])
        cfg.speech_recognition_language = self.params.get("language", "ko-KR")
        fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=sr, bits_per_sample=16, channels=1)
        stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=speechsdk.audio.AudioConfig(stream=stream))
        conn = speechsdk.Connection.from_recognizer(rec)

        lock = threading.Lock()
        connected, done = threading.Event(), threading.Event()
        marks: dict[str, float] = {}
        texts: list[str] = []
        errors: list[tuple[str, str]] = []
        partials = 0

        def on_partial(_evt) -> None:
            nonlocal partials
            with lock:
                partials += 1
                marks.setdefault("first_partial", now())

        def on_final(evt) -> None:
            if evt.result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return
            t = now()
            with lock:
                if evt.result.text:
                    texts.append(evt.result.text)
                marks.setdefault("first_final", t)
                marks["last_final"] = t

        def on_canceled(evt) -> None:
            if evt.reason == speechsdk.CancellationReason.Error:
                with lock:
                    errors.append((str(evt.error_code).rsplit(".", 1)[-1], redact(evt.error_details or "")))
            done.set()

        conn.connected.connect(lambda _evt: connected.set())
        rec.recognizing.connect(on_partial)
        rec.recognized.connect(on_final)
        rec.canceled.connect(on_canceled)
        rec.session_stopped.connect(lambda _evt: done.set())

        audio_started = False
        try:
            t_connect = now()
            conn.open(True)
            if not connected.wait(connect_timeout):
                raise TransientError(f"Speech SDK did not connect within {connect_timeout}s")
            connect_s = now() - t_connect
            rec.start_continuous_recognition_async().get()
            audio_started = True
            t0 = now()
            for i, chunk in enumerate(chunks):
                delay = t0 + i * chunk_ms / 1000 - now()
                if delay > 0:
                    time.sleep(delay)
                if errors:
                    break
                stream.write(chunk)
            speech_end = now()
            stream.close()
            finished = done.wait(final_timeout)
        finally:
            try:
                rec.stop_continuous_recognition_async().get()
            finally:
                conn.close()

        if errors:
            code, details = errors[0]
            message = f"Speech SDK canceled: {code} {details}"[:500]
            if code in _TRANSIENT_CODES:
                raise TransientError(message, code=code, ambiguous=audio_started)
            raise PermanentError(message, code=code)
        if not finished:
            raise TransientError(f"timed out after {final_timeout}s waiting for session end",
                                 code="final_timeout", ambiguous=True)

        latency: dict[str, Any] = {"connect_s": connect_s, "audio_send_s": speech_end - t0}
        if "first_partial" in marks:
            latency["first_partial_s"] = marks["first_partial"] - t0
        if "first_final" in marks:
            latency["first_final_s"] = marks["first_final"] - t0
            latency["final_after_audio_end_s"] = marks["last_final"] - speech_end
        return Transcription(
            text=" ".join(t.strip() for t in texts if t.strip()),
            raw={"segments": texts, "sdk_version": speechsdk.__version__},
            latency=latency,
            partials=partials,
            meta={"segments": len(texts)},
        )

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        checks = await super().doctor(ws)
        try:
            sdk = _sdk()
            checks.append({"name": "speech_sdk", "ok": True, "detail": f"version {sdk.__version__}"})
        except PermanentError as exc:
            checks.append({"name": "speech_sdk", "ok": False, "detail": str(exc)})
        check, _ = await check_get(self, "speech_models", speech_models_check_url(self.params["endpoint"]))
        return checks + [check]
