from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lib.backends.audio_backends.base import AudioCapability, AudioSynthesisRequest
from tests.fakes import captured_openai_clients


def _response(data: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(audio=SimpleNamespace(data=data)))])


class TestMimoAudioBackend:
    async def test_synthesize_posts_chat_request_and_decodes_wav(self, tmp_path: Path):
        wav = b"RIFFmimo-wav"
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
        client.chat.completions.create.return_value = _response(base64.b64encode(wav).decode())

        with captured_openai_clients(client) as created:
            from lib.backends.audio_backends.mimo import MimoAudioBackend

            backend = MimoAudioBackend(
                api_key="secret",
                base_url="https://api.xiaomimimo.com/v1",
                model="mimo-v2.5-tts",
                provider_name="custom",
            )
            out = tmp_path / "speech.wav"
            result = await backend.synthesize(AudioSynthesisRequest(text="你好", output_path=out, voice="Chloe"))

        assert created[0]["default_headers"] == {"api-key": "secret"}
        client.chat.completions.create.assert_awaited_once_with(
            model="mimo-v2.5-tts",
            messages=[{"role": "assistant", "content": "你好"}],
            audio={"format": "wav", "voice": "Chloe"},
        )
        assert out.read_bytes() == wav
        assert result.provider == "custom"
        assert backend.capabilities == {AudioCapability.TEXT_TO_SPEECH}
        assert [voice.id for voice in backend.list_voices()] == ["Chloe"]

    async def test_missing_audio_data_raises(self, tmp_path: Path):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())))
        client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        with captured_openai_clients(client):
            from lib.backends.audio_backends.mimo import MimoAudioBackend

            backend = MimoAudioBackend(api_key="secret", model="mimo-v2.5-tts", provider_name="custom")
            with pytest.raises(RuntimeError, match=r"message\.audio\.data"):
                await backend.synthesize(
                    AudioSynthesisRequest(text="x", output_path=tmp_path / "bad.wav", voice="Chloe")
                )
