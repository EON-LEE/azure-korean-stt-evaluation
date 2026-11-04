"""Azure AI Speech Fast transcription (``transcriptions:transcribe``).

The same endpoint serves the standard Azure model, LLM Speech and MAI-Transcribe; the
``definition`` param selects the model (``enhancedMode``).
"""

from __future__ import annotations

import json
from typing import Any

from ..types import AudioInput, Transcription
from .base import HttpAdapter, check_get, now, require, speech_models_check_url


class FastAdapter(HttpAdapter):
    kind = "fast"
    key_header = "Ocp-Apim-Subscription-Key"

    async def transcribe(self, audio: AudioInput) -> Transcription:
        require(self.params, "endpoint", "definition")
        api_version = self.params.get("api_version", "2025-10-15")
        url = f"{self.params['endpoint'].rstrip('/')}/speechtotext/transcriptions:transcribe?api-version={api_version}"
        files = {
            "audio": (audio.path.name, audio.path.read_bytes(), "audio/wav"),
            "definition": (None, json.dumps(self.params["definition"]), "application/json"),
        }
        headers = await self.headers()
        t0 = now()
        resp = await self.request("POST", url, files=files, headers=headers)
        request_s = now() - t0
        body: dict[str, Any] = resp.json()
        text = " ".join(p.get("text", "") for p in body.get("combinedPhrases") or []).strip()
        return Transcription(
            text=text,
            raw=body,
            latency={"request_s": request_s},
            usage={"durationMilliseconds": body.get("durationMilliseconds")},
            meta={"phrases": len(body.get("phrases") or [])},
        )

    async def doctor(self, ws: bool = False) -> list[dict[str, Any]]:
        checks = await super().doctor(ws)
        check, _ = await check_get(self, "speech_models", speech_models_check_url(self.params["endpoint"]))
        return checks + [check]
