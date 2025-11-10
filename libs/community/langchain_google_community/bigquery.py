from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.utils import guard_import

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.auth.credentials import Credentials  # type: ignore[import]


def import_bigquery() -> Any:
    """Import the google-cloud-bigquery library and return the client."""
    return guard_import(
        "google.cloud.bigquery", pip_name="langchain-google-community[bigquery]"
    )


class BigQueryLoader(BaseLoader):
    """Load documents from Google Cloud BigQuery.

    Inherits from [`BaseLoader`][langchain_core.document_loaders.BaseLoader].

    Each row becomes a document. Columns can be mapped to `page_content` or
    `metadata`. By default, all columns map to `page_content`.

    !!! note "Installation"

        Requires additional dependencies:

        ```bash
        pip install langchain-google-community[bigquery]
        ```
    """

    def __init__(
        self,
        query: str,
        project: Optional[str] = None,
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        credentials: Optional[Credentials] = None,
    ):
        """Initialize BigQuery document loader.

        Args:
            query: The query to run in BigQuery.
            project: Optional. The project to run the query in.
            page_content_columns: Optional. The columns to write into the `page_content`
                of the document.
            metadata_columns: Optional. The columns to write into the `metadata` of the
                document.
            credentials: Optional. Credentials for accessing Google APIs. Use this
                parameter to override default credentials, such as to use Compute Engine
                (`google.auth.compute_engine.Credentials`) or Service Account
                (`google.oauth2.service_account.Credentials`) credentials directly.
        """
        import_bigquery()
        self.query = query
        self.project = project
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns
        self.credentials = credentials

    def load(self) -> List[Document]:
        try:
            from google.cloud import bigquery  # type: ignore[attr-defined]
        except ImportError as ex:
            raise ImportError(
                "Could not import google-cloud-bigquery python package. "
                "Please, install bigquery dependency group: "
                "`pip install langchain-google-community[bigquery]`"
            ) from ex

        bq_client = bigquery.Client(
            credentials=self.credentials,
            project=self.project,
            client_info=get_client_info(module="bigquery"),
        )
        if not bq_client.project:
            error_desc = (
                "GCP project for Big Query is not set! Either provide a "
                "`project` argument during BigQueryLoader instantiation, "
                "or set a default project with `gcloud config set project` "
                "command."
            )
            raise ValueError(error_desc)
        query_result = bq_client.query(self.query).result()
        docs: List[Document] = []

        page_content_columns = self.page_content_columns
        metadata_columns = self.metadata_columns

        if page_content_columns is None:
            page_content_columns = [column.name for column in query_result.schema]
        if metadata_columns is None:
            metadata_columns = []

        for row in query_result:
            page_content = "\n".join(
                f"{k}: {v}" for k, v in row.items() if k in page_content_columns
            )
            metadata = {k: v for k, v in row.items() if k in metadata_columns}
            doc = Document(page_content=page_content, metadata=metadata)
            docs.append(doc)

        return docs
