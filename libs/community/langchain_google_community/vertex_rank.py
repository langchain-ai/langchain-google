import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.auth.credentials import Credentials  # type: ignore
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, Field

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud import discoveryengine_v1alpha  # type: ignore

if TYPE_CHECKING:
    from google.cloud import discoveryengine_v1alpha  # type: ignore


class VertexAIRank(BaseDocumentCompressor):
    """Document compressor using Vertex AI Ranking API.

    Inherits from
    [`BaseDocumentCompressor`][langchain_core.documents.compressor.BaseDocumentCompressor].

    Reranks documents based on relevance to a query using Google's semantic ranking
    model.
    """

    project_id: str = Field(default=None)  # type: ignore
    """Google Cloud project ID."""

    location_id: str = Field(default="global")
    """Location ID for the ranking service."""

    ranking_config: str = Field(default="default_config")
    """Name of the rank service config."""

    model: str = Field(default="semantic-ranker-512@latest")
    """Model identifier."""

    top_n: int = Field(default=10)
    """Number of results to return."""

    ignore_record_details_in_response: bool = Field(default=False)
    """If `True`, response contains only record ID and score."""

    id_field: Optional[str] = Field(default=None)
    """Unique document metadata field to use as an ID."""

    title_field: Optional[str] = Field(default=None)
    """Document metadata field to use as title."""

    credentials: Optional[Credentials] = Field(default=None)
    """Google Cloud credentials object."""

    credentials_path: Optional[str] = Field(default=None)
    """Path to the Google Cloud service account credentials file."""

    timeout: Optional[int] = Field(default=None)
    """Timeout for API calls in seconds."""

    client: Any = None

    def __init__(self, **kwargs: Any):
        """Initialize the Vertex AI Ranker.

        Configures ranking parameters and initializes Google Cloud services.
        """
        super().__init__(**kwargs)
        self.client = kwargs.get("client")  # type: ignore
        if not self.client:
            self.client = self._get_rank_service_client()

    def _get_rank_service_client(self) -> "discoveryengine_v1alpha.RankServiceClient":
        """Get `RankServiceClient` for Vertex AI Ranking API calls.

        Returns:
            Client instance for ranking API.
        """
        try:
            from google.cloud import discoveryengine_v1alpha  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please, install vertexaisearch dependency group: "
                "`pip install langchain-google-community[vertexaisearch]`"
            ) from exc
        return discoveryengine_v1alpha.RankServiceClient(
            credentials=self.credentials
            or (
                Credentials.from_service_account_file(self.credentials_path)  # type: ignore[attr-defined]
                if self.credentials_path
                else None
            ),
            client_info=get_client_info(module="vertex-ai-search"),
        )

    def _rerank_documents(
        self, query: str, documents: Sequence[Document]
    ) -> Sequence[Document]:
        """Rerank documents based on query relevance.

        Args:
            query: Query to use for reranking.
            documents: Documents to rerank.

        Returns:
            Reranked documents with relevance scores.
        """
        from google.cloud import discoveryengine_v1alpha  # type: ignore

        try:
            records = [
                discoveryengine_v1alpha.RankingRecord(
                    id=(doc.metadata.get(self.id_field) if self.id_field else str(idx)),
                    content=doc.page_content,
                    **(
                        {"title": doc.metadata.get(self.title_field)}
                        if self.title_field
                        else {}
                    ),
                )
                for idx, doc in enumerate(documents)
                if doc.page_content
                or (self.title_field and doc.metadata.get(self.title_field))
            ]
        except KeyError:
            warnings.warn(f"id_field '{self.id_field}' not found in document metadata.")

        ranking_config_path = (
            f"projects/{self.project_id}/locations/{self.location_id}"
            f"/rankingConfigs/{self.ranking_config}"
        )

        request = discoveryengine_v1alpha.RankRequest(
            ranking_config=ranking_config_path,
            model=self.model,
            query=query,
            records=records,
            top_n=self.top_n,
            ignore_record_details_in_response=self.ignore_record_details_in_response,
        )

        try:
            response = self.client.rank(request=request, timeout=self.timeout)
        except core_exceptions.GoogleAPICallError as e:
            print(f"Error in Vertex AI Ranking API call: {str(e)}")
            raise RuntimeError(f"Error in Vertex AI Ranking API call: {str(e)}") from e

        return [
            Document(
                page_content=record.content
                if not self.ignore_record_details_in_response
                else "",
                metadata={
                    "id": record.id,
                    "relevance_score": record.score,
                    **({self.title_field: record.title} if self.title_field else {}),
                },
            )
            for record in response.records
        ]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using Vertex AI rerank API.

        Args:
            documents: Document instances to compress.
            query: Query string for document compression.
            callbacks: Callbacks to execute during compression.

        Returns:
            Compressed documents with relevance scores.
        """
        return self._rerank_documents(query, documents)

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
    )
