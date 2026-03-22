"""Voice transcription providers (Groq, OpenAI Whisper, and Gemini)."""

import base64
import os
from pathlib import Path

import httpx
import litellm
from loguru import logger


class OpenAITranscriptionProvider:
    """Voice transcription provider using OpenAI's Whisper API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("OpenAI API key not configured for transcription")
            return ""
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {"file": (path.name, f), "model": (None, "whisper-1")}
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = await client.post(
                        self.api_url, headers=headers, files=files, timeout=60.0,
                    )
                    response.raise_for_status()
                    return response.json().get("text", "")
        except Exception as e:
            logger.error("OpenAI transcription error: {}", e)
            return ""


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


_MIME_TYPES = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".amr": "audio/amr",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}


class GeminiTranscriptionProvider:
    """
    Voice transcription provider using Google's Gemini Flash model via LiteLLM.

    Sends audio as a multimodal content part to Gemini's chat completion API.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini/gemini-2.0-flash"):
        self.api_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self.model = model

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Gemini via LiteLLM multimodal completion.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Google API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        mime = _MIME_TYPES.get(path.suffix.lower(), "audio/ogg")
        audio_b64 = base64.b64encode(path.read_bytes()).decode()

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe this audio verbatim. "
                                    "Output only the transcription, nothing else."
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_b64, "format": mime},
                            },
                        ],
                    }
                ],
                api_key=self.api_key,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("Gemini transcription error: {}", e)
            return ""
