from typing import Any, Optional, Sequence

from langchain_core.documents import BaseDocumentTransformer, Document

from langchain_google_community._utils import get_client_info


class GoogleTranslateTransformer(BaseDocumentTransformer):
    """Translate text documents using Google Cloud Translation."""

    def __init__(
        self,
        project_id: str,
        *,
        location: str = "global",
        model_id: Optional[str] = None,
        glossary_id: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            project_id: Google Cloud Project ID.
            location: (Optional) Translate model location.
            model_id: (Optional) Translate model ID to use.
            glossary_id: (Optional) Translate glossary ID to use.
            api_endpoint: (Optional) Regional endpoint to use.
        """
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import translate  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-translate python package. "
                "Please, install translate dependency group: "
                "`pip install langchain-google-community[translate]`"
            ) from exc

        self.project_id = project_id
        self.location = location
        self.model_id = model_id
        self.glossary_id = glossary_id

        self._client = translate.TranslationServiceClient(
            client_info=get_client_info("translate"),
            client_options=(
                ClientOptions(api_endpoint=api_endpoint) if api_endpoint else None
            ),
        )
        self._parent_path = self._client.common_location_path(project_id, location)
        # For some reason, there's no `model_path()` method for the client.
        self._model_path = (
            f"{self._parent_path}/models/{model_id}" if model_id else None
        )
        self._glossary_path = (
            self._client.glossary_path(project_id, location, glossary_id)
            if glossary_id
            else None
        )

    def transform_documents(
        self,
        documents: Sequence[Document],
        *,
        source_language_code: Optional[str] = None,
        target_language_code: Optional[str] = None,
        mime_type: str = "text/plain",
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Translate text documents using Google Translate.

        Args:
            documents: Sequence of documents to translate.
            source_language_code: ISO 639 language code of the input document.
                If not provided, language will be auto-detected.
            target_language_code: ISO 639 language code of the output document.
                Required for translation. For supported languages, refer to:
                https://cloud.google.com/translate/docs/languages
            mime_type: Media Type of input text.
                Options: ``'text/plain'``, ``'text/html'``.

        Returns:
            Sequence of translated documents with updated metadata.

        Raises:
            ValueError: If ``target_language_code`` is not provided.
        """
        if target_language_code is None:
            msg = (
                "target_language_code is required for translation. "
                "Please provide an ISO 639 language code."
            )
            raise ValueError(msg)

        try:
            from google.cloud import translate  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-translate python package. "
                "Please, install translate dependency group: "
                "`pip install langchain-google-community[translate]`"
            ) from exc

        response = self._client.translate_text(
            request=translate.TranslateTextRequest(
                contents=[doc.page_content for doc in documents],
                parent=self._parent_path,
                model=self._model_path,
                glossary_config=translate.TranslateTextGlossaryConfig(
                    glossary=self._glossary_path
                ),
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                mime_type=mime_type,
            )
        )

        # If using a glossary, the translations will be in `glossary_translations`.
        translations = response.glossary_translations or response.translations

        return [
            Document(
                page_content=translation.translated_text,
                metadata={
                    **doc.metadata,
                    "model": translation.model,
                    "detected_language_code": translation.detected_language_code,
                },
            )
            for doc, translation in zip(documents, translations)
        ]
