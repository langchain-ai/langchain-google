from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud.speech_v2 import RecognitionConfig  # type: ignore[import]
    from google.protobuf.field_mask_pb2 import FieldMask


class SpeechToTextLoader(BaseLoader):
    """Loader for Google Cloud Speech-to-Text audio transcripts.

    Inherits from [`BaseLoader`][langchain_core.document_loaders.BaseLoader].

    Transcribes audio files using Google Cloud Speech-to-Text API and loads
    transcribed text into documents. Supports both GCS URIs and local file paths.

    See [Speech-to-Text documentation](https://cloud.google.com/speech-to-text)
    for detailed information.

    !!! note "Installation"

        Requires additional dependencies:

        ```bash
        pip install langchain-google-community[speech]
        ```
    """

    def __init__(
        self,
        project_id: str,
        file_path: str,
        location: str = "us-central1",
        recognizer_id: str = "_",
        config: Optional[RecognitionConfig] = None,
        config_mask: Optional[FieldMask] = None,
        is_long: bool = False,
    ):
        """Initialize the Speech-to-Text loader.

        Args:
            project_id: Google Cloud Project ID.
            file_path: Google Cloud Storage URI or local file path.
            location: Speech-to-Text recognizer location.
            recognizer_id: Speech-to-Text recognizer ID.
            config: Recognition options and features. See
                [`RecognitionConfig`](https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognitionConfig).
            config_mask: Fields in config that override `default_recognition_config`
                of the recognizer. See [`RecognizeRequest`](https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognizeRequest).
            is_long: Use async Cloud Speech recognition for long audio files. See
                [batch recognize](https://cloud.google.com/speech-to-text/v2/docs/batch-recognize).
        """
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud.speech_v2 import (
                AutoDetectDecodingConfig,
                RecognitionConfig,
                RecognitionFeatures,
                SpeechClient,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-speech python package. "
                "Please, install speech dependency group: "
                "`pip install langchain-google-community[speech]`"
            ) from exc

        self.project_id = project_id
        self.file_path = file_path
        self.location = location
        self.recognizer_id = recognizer_id
        # Config must be set in speech recognition request.
        self.config = config or RecognitionConfig(
            auto_decoding_config=AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp",
            features=RecognitionFeatures(
                # Automatic punctuation could be useful for language applications
                enable_automatic_punctuation=True,
            ),
        )
        self.config_mask = config_mask

        self._client = SpeechClient(
            client_info=get_client_info(module="speech-to-text"),
            client_options=(
                ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
                if location != "global"
                else None
            ),
        )
        self._recognizer_path = self._client.recognizer_path(
            project_id, location, recognizer_id
        )
        self._is_long = is_long

    def load(self) -> List[Document]:
        """Transcribes the audio file and loads the transcript into documents.

        It uses the Google Cloud Speech-to-Text API to transcribe the audio file
        and blocks until the transcription is finished.
        """
        if self._is_long:
            return [Document(page_content=self._load_long())]
        try:
            from google.cloud.speech_v2 import RecognizeRequest
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-speech python package. "
                "Please, install speech dependency group: "
                "`pip install langchain-google-community[speech]`"
            ) from exc

        request = RecognizeRequest(
            recognizer=self._recognizer_path,
            config=self.config,
            config_mask=self.config_mask,
        )

        if "gs://" in self.file_path:
            request.uri = self.file_path
        else:
            with open(self.file_path, "rb") as f:
                request.content = f.read()

        response = self._client.recognize(request=request)

        return [
            Document(
                page_content=result.alternatives[0].transcript,
                metadata={
                    "language_code": result.language_code,
                    "result_end_offset": result.result_end_offset,
                },
            )
            for result in response.results
        ]

    def _load_long(self) -> str:
        from google.cloud.speech_v2 import (
            BatchRecognizeFileMetadata,
            BatchRecognizeRequest,
            InlineOutputConfig,
            RecognitionOutputConfig,
        )

        request = BatchRecognizeRequest(
            recognizer=self._recognizer_path,
            config=self.config,
            config_mask=self.config_mask,
            files=[BatchRecognizeFileMetadata(uri=self.file_path)],
            recognition_output_config=RecognitionOutputConfig(
                inline_response_config=InlineOutputConfig(),
            ),
        )
        operation = self._client.batch_recognize(request=request)
        response = operation.result(timeout=120)
        return "".join(
            [
                r.alternatives[0].transcript
                for r in response.results[self.file_path].transcript.results
                if r.alternatives
            ]
        )
