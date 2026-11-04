"""Azure OpenAI file transcription (``/audio/transcriptions``) for whisper and gpt-*-transcribe."""

from __future__ import annotations

from typing import Any

from ..types import AudioInput, Transcription
from .base import HttpAdapter, check_aoai_deployment, now, require


class OpenAIFileAdapter(HttpAdapter):
    kind = "openai_file"
    key_header = "api-key"

    async def transcribe(self, audio: AudioInput) -> Transcription:
        require(self.params, "endpoint", "deployment")
        api_version = self.params.get("api_version", "2025-04-01-preview")
        url = (f"{self.params['endpoint'].rstrip('/')}/openai/deployments/{self.params['deployment']}"
               f"/audio/transcriptions?api-version={api_version}")
        form = {k: str(v) for k, v in (self.params.get("form") or {}).items()}
        form.setdefault("response_format", "json")
        headers = await self.headers()
        t0 = now()
        resp = await self.request("POST", url, data=form, headers=headers,
                                  files={"file": (audio.path.name, audio.path.read_bytes(), "audio/wav")})
        request_s = now() - t0
        body: dict[str, Any] = resp.json()
        return Transcription(
            text=(body.get("text") or "").strip(),
            raw=body,
            latency={"request_s": request_s},
            usage=body.get("usage"),
        )

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        checks = await super().doctor(ws)
        return checks + [await check_aoai_deployment(self, self.params["endpoint"], self.params["deployment"],
                                                     self.cfg.model_id)]
