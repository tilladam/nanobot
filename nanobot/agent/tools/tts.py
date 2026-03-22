"""Text-to-speech tool using ElevenLabs."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.config.paths import get_media_dir

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # George


class TTSTool(Tool):
    """Convert text to speech via ElevenLabs and return the audio file path."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")

    @property
    def name(self) -> str:
        return "text_to_speech"

    @property
    def description(self) -> str:
        return (
            "Convert text to a spoken audio file using ElevenLabs. "
            "Returns the file path of the generated audio. "
            "Use the message tool with the path in the media list to send it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to speech.",
                },
                "voice_id": {
                    "type": "string",
                    "description": (
                        "ElevenLabs voice ID. Defaults to George."
                    ),
                },
                "model_id": {
                    "type": "string",
                    "description": (
                        "ElevenLabs model ID. Defaults to eleven_multilingual_v2."
                    ),
                },
            },
            "required": ["text"],
        }

    async def execute(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        model_id: str = DEFAULT_MODEL,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            return "Error: ElevenLabs API key not configured."

        if not text.strip():
            return "Error: text must not be empty."

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ELEVENLABS_API_URL}/{voice_id}",
                    headers={
                        "xi-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": model_id,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()

                out_dir = get_media_dir("tts")
                out_path = out_dir / f"{uuid.uuid4().hex}.mp3"
                out_path.write_bytes(response.content)
                logger.info("TTS audio saved to {}", out_path)
                return str(out_path)

        except httpx.HTTPStatusError as e:
            logger.error("ElevenLabs API error {}: {}", e.response.status_code, e.response.text)
            return f"Error: ElevenLabs API returned {e.response.status_code}."
        except Exception as e:
            logger.error("TTS error: {}", e)
            return f"Error: {e}"
