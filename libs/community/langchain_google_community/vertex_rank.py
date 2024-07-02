import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.auth.credentials import Credentials  # type: ignore
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Extra, Field

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud import discoveryengine_v1alpha  # type: ignore

if TYPE_CHECKING:
    from google.cloud import discoveryengine_v1alpha  # type: ignore


class VertexAIRank(BaseDocumentCompressor):
    """
    Initializes the Vertex AI Ranker with configurable parameters.

    Inherits from BaseDocumentCompressor for document processing
    and validation features, respectively.

    Attributes:
        project_id (str): Google Cloud project ID
        location_id (str): Location ID for the ranking service.
        ranking_config (str):
            Required. The  name of the rank service config, such as default_config.
            It is set to default_config by default if unspecified.
        model (str):
            The identifier of the model to use. It is one of:

            - ``semantic-ranker-512@latest``: Semantic ranking model
              with maximum input token size 512.

            It is set to ``semantic-ranker-512@latest`` by default if unspecified.
        top_n (int):
            The number of results to return. If this is
            unset or no bigger than zero, returns all
            results.
        ignore_record_details_in_response (bool):
            If true, the response will contain only
            record ID and score. By default, it is false,
            the response will contain record details.
        id_field (Optional[str]): Specifies a unique document metadata field
        to use as an id.
        title_field (Optional[str]): Specifies the document metadata field
        to use as title.
        credentials (Optional[Credentials]): Google Cloud credentials object.
        credentials_path (Optional[str]): Path to the Google Cloud service
        account credentials file.
    """

    project_id: str = Field(default=None)
    location_id: str = Field(default="global")
    ranking_config: str = Field(default="default_config")
    model: str = Field(default="semantic-ranker-512@latest")
    top_n: int = Field(default=10)
    ignore_record_details_in_response: bool = Field(default=False)
    id_field: Optional[str] = Field(default=None)
    title_field: Optional[str] = Field(default=None)
    credentials: Optional[Credentials] = Field(default=None)
    credentials_path: Optional[str] = Field(default=None)
    client: Any

    def __init__(self, **kwargs: Any):
        """
        Constructor for VertexAIRanker, allowing for specification of
        ranking configuration and initialization of Google Cloud services.

        The parameters accepted are the same as the attributes listed above.
        """
        super().__init__(**kwargs)
        self.client = kwargs.get("client")  # type: ignore
        if not self.client:
            self.client = self._get_rank_service_client()

    def _get_rank_service_client(self) -> "discoveryengine_v1alpha.RankServiceClient":
        """
        Returns a RankServiceClient instance for making API calls to the
        Vertex AI Ranking service.

        Returns:
            A RankServiceClient instance.
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
            credentials=(
                self.credentials
                or Credentials.from_service_account_file(self.credentials_path)  # type: ignore[attr-defined]
                if self.credentials_path
                else None
            ),
            client_info=get_client_info(module="vertex-ai-search"),
        )

    def _rerank_documents(
        self, query: str, documents: Sequence[Document]
    ) -> Sequence[Document]:
        """
        Reranks documents based on the provided query.

        Args:
            query: The query to use for reranking.
            documents: The list of documents to rerank.

        Returns:
            A list of reranked documents.
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
            response = self.client.rank(request=request)
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
        """
        Compresses documents using Vertex AI's rerank API.

        Args:
            documents: List of Document instances to compress.
            query: Query string to use for compressing the documents.
            callbacks: Callbacks to execute during compression (not used here).

        Returns:
            A list of Document instances, compressed.
        """
        return self._rerank_documents(query, documents)

    class Config:
        extra = Extra.ignore
        arbitrary_types_allowed = True
