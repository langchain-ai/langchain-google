from typing import Callable, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_google_community._utils import get_client_info
from langchain_google_community.gcs_file import GCSFileLoader


class GCSDirectoryLoader(BaseLoader):
    """Load from GCS directory."""

    def __init__(
        self,
        project_name: str,
        bucket: str,
        prefix: str = "",
        loader_func: Optional[Callable[[str], BaseLoader]] = None,
    ):
        """Initialize with bucket and key name.

        Args:
            project_name: The ID of the project for the GCS bucket.
            bucket: The name of the GCS bucket.
            prefix: The prefix of the GCS bucket.
            loader_func: A loader function that instantiates a loader based on a
                file_path argument. If nothing is provided, the  GCSFileLoader
                would use its default loader.
        """
        self.project_name = project_name
        self.bucket = bucket
        self.prefix = prefix
        self._loader_func = loader_func

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from google.cloud import storage  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-storage python package. "
                "Please, install gcs dependency group: "
                "poetry install --with gcs"
            )
        client = storage.Client(
            project=self.project_name,
            client_info=get_client_info(module="google-cloud-storage"),
        )
        docs = []
        for blob in client.list_blobs(self.bucket, prefix=self.prefix):
            # we shall just skip directories since GCSFileLoader creates
            # intermediate directories on the fly
            if blob.name.endswith("/"):
                continue
            loader = GCSFileLoader(
                self.project_name, self.bucket, blob.name, loader_func=self._loader_func
            )
            docs.extend(loader.load())
        return docs
