from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud import texttospeech  # type: ignore[attr-defined]


def _import_google_cloud_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech  # type: ignore[attr-defined]
    except ImportError as e:
        raise ImportError(
            "Could not import google-cloud-texttospeech python package. "
            "Please, install texttospeech dependency group: "
            "`pip install langchain-google-community[texttospeech]`"
        ) from e
    return texttospeech


def _encoding_file_extension_map(encoding: texttospeech.AudioEncoding) -> Optional[str]:
    texttospeech = _import_google_cloud_texttospeech()

    ENCODING_FILE_EXTENSION_MAP = {
        texttospeech.AudioEncoding.LINEAR16: ".wav",
        texttospeech.AudioEncoding.MP3: ".mp3",
        texttospeech.AudioEncoding.OGG_OPUS: ".ogg",
        texttospeech.AudioEncoding.MULAW: ".wav",
        texttospeech.AudioEncoding.ALAW: ".wav",
    }
    return ENCODING_FILE_EXTENSION_MAP.get(encoding)


class TextToSpeechTool(BaseTool):
    """Tool that queries the Google Cloud Text-to-Speech API.

    Inherits from [`BaseTool`][langchain_core.tools.BaseTool].

    Synthesizes audio from text with support for multiple languages and voice options.

    !!! note "Installation"

        Requires additional dependencies:

        ```bash
        pip install langchain-google-community[texttospeech]
        ```

    !!! note "Setup Required"

        Follow [setup instructions](https://cloud.google.com/text-to-speech/docs/before-you-begin)
        to configure Google Cloud Text-to-Speech API.
    """

    name: str = "google_cloud_texttospeech"
    description: str = (
        "A wrapper around Google Cloud Text-to-Speech. "
        "Useful for when you need to synthesize audio from text. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    _client: Any

    def __init__(self, **kwargs: Any) -> None:
        """Initializes private fields."""
        texttospeech = _import_google_cloud_texttospeech()

        super().__init__(**kwargs)

        self._client = texttospeech.TextToSpeechClient(
            client_info=get_client_info(module="text-to-speech")
        )

    def _run(
        self,
        input_text: str,
        language_code: str = "en-US",
        ssml_gender: Optional[texttospeech.SsmlVoiceGender] = None,
        audio_encoding: Optional[texttospeech.AudioEncoding] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synthesize speech from text.

        Args:
            language_code: Language code (e.g., `'en-US'`, `'de-DE'`).
            ssml_gender: Voice gender.
            audio_encoding: Audio format.
            run_manager: Optional callback manager.

        Returns:
            Path to temporary file containing synthesized audio.
        """
        texttospeech = _import_google_cloud_texttospeech()
        ssml_gender = ssml_gender or texttospeech.SsmlVoiceGender.NEUTRAL
        audio_encoding = audio_encoding or texttospeech.AudioEncoding.MP3

        response = self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=input_text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=language_code, ssml_gender=ssml_gender
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=audio_encoding),
        )

        suffix = _encoding_file_extension_map(audio_encoding)

        with tempfile.NamedTemporaryFile(mode="bx", suffix=suffix, delete=False) as f:
            f.write(response.audio_content)
        return f.name
