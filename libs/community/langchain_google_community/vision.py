from typing import Iterator, List, Optional

from langchain_core.document_loaders import BaseBlobParser, BaseLoader
from langchain_core.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document

from langchain_google_community._utils import get_client_info


class CloudVisionParser(BaseBlobParser):
    def __init__(self, project: Optional[str] = None):
        try:
            from google.cloud import vision  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "Could not import google-cloud-vision python package. "
                "Please, install vision dependency group: "
                "uv sync --extra vision"
            ) from e
        client_options = None
        if project:
            client_options = {"quota_project_id": project}
        self._client = vision.ImageAnnotatorClient(
            client_options=client_options,
            client_info=get_client_info(module="cloud-vision"),
        )

    def load(self, gcs_uri: str) -> Document:
        """Loads an image from GCS path to a Document, only the text."""
        from google.cloud import vision  # type: ignore[attr-defined]

        image = vision.Image(source=vision.ImageSource(image_uri=gcs_uri))
        text_detection_response = self._client.text_detection(image=image)
        annotations = text_detection_response.text_annotations

        if annotations:
            text = annotations[0].description
        else:
            text = ""
        return Document(page_content=text, metadata={"source": gcs_uri})

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        yield self.load(blob.path)  # type: ignore[arg-type]


class CloudVisionLoader(BaseLoader):
    def __init__(self, file_path: str, project: Optional[str] = None):
        try:
            from google.cloud import vision  # type: ignore[attr-defined]
        except ImportError as e:
            raise ImportError(
                "Could not import google-cloud-vision python package. "
                "Please, install vision dependency group: "
                "`pip install langchain-google-community[vision]`"
            ) from e
        client_options = None
        if project:
            client_options = {"quota_project_id": project}
        self._client = vision.ImageAnnotatorClient(
            client_options=client_options,
            client_info=get_client_info(module="cloud-vision"),
        )
        self._file_path = file_path

    def load(self) -> List[Document]:
        """Loads an image from GCS path to a Document, only the text."""
        from google.cloud import vision  # type: ignore[attr-defined]

        image = vision.Image(source=vision.ImageSource(image_uri=self._file_path))
        text_detection_response = self._client.text_detection(image=image)
        annotations = text_detection_response.text_annotations

        if annotations:
            text = annotations[0].description
        else:
            text = ""
        return [Document(page_content=text, metadata={"source": self._file_path})]
