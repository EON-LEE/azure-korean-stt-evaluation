import asyncio
import json

import numpy as np
import pytest

from kstt_eval.adapters import create_adapter
from kstt_eval.adapters.realtime import PREFIX, RealtimeServiceError, TranscriptAccumulator, _ws_url
from kstt_eval.audio import encode_wav_pcm16
from kstt_eval.config import ModelConfig
from kstt_eval.types import AudioInput, PermanentError, TransientError


# --- accumulator --------------------------------------------------------------------------------

def _completed(item, text):
    return {"type": PREFIX + "completed", "item_id": item, "transcript": text}


def test_commit_mode_waits_for_item_after_commit():
    acc = TranscriptAccumulator()
    acc.mark_commit()
    assert not acc.done()
    acc.feed({"type": "input_audio_buffer.committed", "item_id": "i1"}, 1.0)
    assert not acc.done()  # committed but not transcribed yet
    acc.feed({"type": PREFIX + "delta", "item_id": "i1", "delta": "안"}, 1.1)
    acc.feed(_completed("i1", "안녕하세요"), 1.2)
    assert acc.done() and acc.text() == "안녕하세요"
    assert acc.partials == 1 and acc.first_partial_t == 1.1 and acc.first_final_t == 1.2


def test_commit_empty_finishes_with_empty_text():
    acc = TranscriptAccumulator()
    acc.mark_commit()
    acc.feed({"type": "error", "error": {"code": "input_audio_buffer_commit_empty"}}, 1.0)
    assert acc.done() and acc.text() == ""


def test_completed_only_service():
    acc = TranscriptAccumulator()
    acc.mark_commit()
    acc.feed(_completed("x", "완료 이벤트만"), 1.0)
    assert acc.done() and acc.text() == "완료 이벤트만"


def test_vad_multi_segment_order_and_speaking():
    acc = TranscriptAccumulator()
    acc.feed({"type": "input_audio_buffer.speech_started"}, 0.1)
    assert not acc.done()
    acc.feed({"type": "input_audio_buffer.speech_stopped"}, 0.5)
    acc.feed({"type": "input_audio_buffer.committed", "item_id": "a"}, 0.5)
    acc.feed({"type": "input_audio_buffer.speech_started"}, 0.6)
    acc.feed({"type": "input_audio_buffer.speech_stopped"}, 0.9)
    acc.feed({"type": "input_audio_buffer.committed", "item_id": "b"}, 0.9)
    acc.feed(_completed("b", "둘째"), 1.0)
    assert not acc.done()  # "a" still pending
    acc.feed(_completed("a", "첫째"), 1.1)
    assert acc.done() and acc.text() == "첫째 둘째"


def test_failed_item_is_not_pending_and_other_errors_raise():
    acc = TranscriptAccumulator()
    acc.mark_commit()
    acc.feed({"type": "input_audio_buffer.committed", "item_id": "i1"}, 1.0)
    acc.feed({"type": PREFIX + "failed", "item_id": "i1", "error": {"message": "x"}}, 1.1)
    assert acc.done() and acc.failed
    with pytest.raises(RealtimeServiceError):
        acc.feed({"type": "error", "error": {"code": "invalid_value", "message": "bad"}}, 1.2)


def test_ws_url():
    assert _ws_url("https://host.example/") == "wss://host.example"
    assert _ws_url("host.example") == "wss://host.example"


# --- streaming flow with a fake WebSocket -------------------------------------------------------

class FakeWS:
    """Scripted server. ``react(ws, msg)`` pushes server events in response to client messages."""

    def __init__(self, react, created=True):
        self.q: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.appends = 0
        self.react = react
        if created:
            self.push({"type": "session.created"})

    def push(self, ev):
        self.q.put_nowait(json.dumps(ev))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def recv(self):
        return await self.q.get()

    async def send(self, msg):
        d = json.loads(msg)
        self.sent.append(d)
        if d["type"] == "input_audio_buffer.append":
            self.appends += 1
        self.react(self, d)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.q.get()


def _audio(tmp_path, seconds=0.1):
    sr = 16000
    t = np.arange(int(sr * seconds)) / sr
    path = tmp_path / "u.wav"
    path.write_bytes(encode_wav_pcm16((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr))
    return AudioInput(utt_id="u", split="eval_clean", path=path, duration_s=seconds, sample_rate=sr, sha256="x")


def _adapter(ws, adapter="openai_realtime", **params):
    base = {"endpoint": "https://aoai.invalid", "ai_host": "ai.invalid", "deployment": "dep",
            "chat_model": "gpt-4.1", "transcription_model": "azure-speech", "chunk_ms": 20,
            "final_timeout_s": 5, "connect_timeout_s": 5}
    base.update(params)
    a = create_adapter(ModelConfig(key="rt", adapter=adapter, track="B", params=base, model_id="m"))

    async def fake_connect():
        return ws

    a._connect = fake_connect
    return a


def _ack_session(ws, d):
    if d["type"] == "session.update":
        ws.push({"type": "session.updated"})
        return True
    return False


def test_commit_flow(tmp_path):
    def react(ws, d):
        if _ack_session(ws, d):
            return
        if d["type"] == "input_audio_buffer.commit":
            ws.push({"type": "input_audio_buffer.committed", "item_id": "i1"})
            ws.push({"type": PREFIX + "delta", "item_id": "i1", "delta": "안녕"})
            ws.push(_completed("i1", "안녕하세요"))

    ws = FakeWS(react)
    tr = asyncio.run(_adapter(ws, mode="commit").transcribe(_audio(tmp_path)))
    assert tr.text == "안녕하세요"
    assert tr.partials == 1 and tr.meta["forced_commit"] is False
    assert {"connect_s", "audio_send_s", "first_partial_s", "final_after_audio_end_s"} <= set(tr.latency)
    update = ws.sent[0]
    assert update["session"]["audio"]["input"]["turn_detection"] is None
    assert update["session"]["audio"]["input"]["format"]["rate"] == 24000
    assert ws.appends == 5  # 0.1 s in 20 ms chunks, no trailing silence in commit mode
    assert [d["type"] for d in ws.sent].count("input_audio_buffer.commit") == 1


def test_vad_flow_two_segments(tmp_path):
    def react(ws, d):
        if _ack_session(ws, d) or d["type"] != "input_audio_buffer.append":
            return
        n = ws.appends
        if n in (1, 4):
            ws.push({"type": "input_audio_buffer.speech_started"})
        if n in (3, 6):
            item = "a" if n == 3 else "b"
            ws.push({"type": "input_audio_buffer.speech_stopped"})
            ws.push({"type": "input_audio_buffer.committed", "item_id": item})
            ws.push(_completed(item, "첫 문장" if item == "a" else "둘째 문장"))

    ws = FakeWS(react)
    tr = asyncio.run(_adapter(ws, adapter="voicelive", mode="vad", trailing_silence_s=0.1).transcribe(
        _audio(tmp_path)))
    assert tr.text == "첫 문장 둘째 문장"
    assert tr.meta["forced_commit"] is False and tr.meta["segments"] == 2
    assert ws.sent[0]["session"]["turn_detection"] == {"type": "server_vad", "create_response": False}
    assert "input_audio_buffer.commit" not in [d["type"] for d in ws.sent]
    assert not any(d["type"].startswith("response.") for d in ws.sent)


def test_vad_forced_commit(tmp_path):
    def react(ws, d):
        if _ack_session(ws, d):
            return
        if d["type"] == "input_audio_buffer.append" and ws.appends == 1:
            ws.push({"type": "input_audio_buffer.speech_started"})
        if d["type"] == "input_audio_buffer.commit":
            ws.push({"type": "input_audio_buffer.committed", "item_id": "c"})
            ws.push(_completed("c", "강제 확정"))

    ws = FakeWS(react)
    tr = asyncio.run(_adapter(ws, mode="vad", trailing_silence_s=0.04, vad_flush_after_s=0.1).transcribe(
        _audio(tmp_path)))
    assert tr.text == "강제 확정" and tr.meta["forced_commit"] is True


def test_session_errors_are_classified(tmp_path):
    def throttled(ws, d):
        if d["type"] == "session.update":
            ws.push({"type": "error", "error": {"code": "rate_limit_exceeded", "message": "slow down"}})

    with pytest.raises(TransientError) as e:
        asyncio.run(_adapter(FakeWS(throttled)).transcribe(_audio(tmp_path)))
    assert e.value.code == "rate_limit_exceeded" and e.value.ambiguous is False

    def invalid(ws, d):
        if d["type"] == "session.update":
            ws.push({"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_value"}})

    with pytest.raises(PermanentError):
        asyncio.run(_adapter(FakeWS(invalid)).transcribe(_audio(tmp_path)))


def test_error_after_audio_is_ambiguous(tmp_path):
    def react(ws, d):
        if _ack_session(ws, d):
            return
        if d["type"] == "input_audio_buffer.commit":
            ws.push({"type": "error", "error": {"type": "server_error", "code": "server_error"}})

    with pytest.raises(TransientError) as e:
        asyncio.run(_adapter(FakeWS(react)).transcribe(_audio(tmp_path)))
    assert e.value.ambiguous is True


def test_final_timeout(tmp_path):
    def react(ws, d):
        if _ack_session(ws, d):
            return
        if d["type"] == "input_audio_buffer.commit":
            ws.push({"type": "input_audio_buffer.committed", "item_id": "i1"})

    with pytest.raises(TransientError) as e:
        asyncio.run(_adapter(FakeWS(react), final_timeout_s=0.3).transcribe(_audio(tmp_path)))
    assert e.value.code == "final_timeout" and e.value.ambiguous is True


def test_mai_rejects_vad_mode(tmp_path):
    with pytest.raises(PermanentError):
        asyncio.run(_adapter(FakeWS(lambda ws, d: None), adapter="mai_realtime", mode="vad").transcribe(
            _audio(tmp_path)))
