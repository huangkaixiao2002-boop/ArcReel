"""MiMo v2.5 TTS backend.

MiMo exposes TTS through ``/chat/completions`` and returns WAV bytes as a
base64 encoded ``message.audio.data`` field, so it cannot use the usual
OpenAI ``/audio/speech`` backend.
"""

from __future__ import annotations

import base64

from lib.backends.audio_backends.base import (
    AudioCapability,
    AudioSynthesisRequest,
    AudioSynthesisResult,
    VoiceOption,
)
from lib.backends.openai_shared import OPENAI_RETRYABLE_ERRORS, create_openai_client
from lib.backends.transport_layer import TransportLayer
from lib.infra.async_thread import run_noninterruptible_sync
from lib.infra.retry import with_retry_async

_VOICE_CATALOG = ("Chloe",)


class MimoAudioBackend:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport_layer: TransportLayer | None = None,
        model: str,
        provider_name: str,
    ) -> None:
        headers = {"api-key": api_key} if api_key else None
        self._client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            transport_layer=transport_layer,
            default_headers=headers,
        )
        self._model = model
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> set[AudioCapability]:
        return {AudioCapability.TEXT_TO_SPEECH}

    def list_voices(self) -> list[VoiceOption]:
        return [VoiceOption(id=v, label=v) for v in _VOICE_CATALOG]

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        audio_bytes = await self._request_speech(request)
        await run_noninterruptible_sync(request.output_path.write_bytes, audio_bytes)
        return AudioSynthesisResult(
            provider=self._provider_name,
            model=self._model,
            text=request.text,
            output_path=request.output_path,
        )

    @with_retry_async(retryable_errors=OPENAI_RETRYABLE_ERRORS)
    async def _request_speech(self, request: AudioSynthesisRequest) -> bytes:
        # MiMo places the text to speak in an assistant message.  A user
        # message is optional and is intended for prosody instructions.
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "assistant", "content": request.text}],
            audio={"format": "wav", "voice": request.voice},
        )
        try:
            audio = completion.choices[0].message.audio
            encoded = getattr(audio, "data", None)
            if not isinstance(encoded, str):
                raise ValueError("missing audio data")
            content = base64.b64decode(encoded, validate=True)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("MiMo TTS response has no valid message.audio.data") from exc
        if not content:
            raise RuntimeError("MiMo TTS returned empty audio")
        return content
