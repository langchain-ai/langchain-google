from typing import TYPE_CHECKING, Any, Dict, List, Optional

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.auth.credentials import Credentials  # type: ignore
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from langchain_core.runnables import RunnableConfig, RunnableSerializable

from langchain_google_community._utils import get_client_info

if TYPE_CHECKING:
    from google.cloud import discoveryengine_v1alpha  # type: ignore


class VertexAICheckGroundingWrapper(
    RunnableSerializable[str, "VertexAICheckGroundingWrapper.CheckGroundingResponse"]
):
    """
    Initializes the Vertex AI CheckGroundingOutputParser with configurable parameters.

    Calls the Check Grounding API to validate the response against a given set of
    documents and returns back citations that support the claims along with the cited
    chunks. Output is of the type CheckGroundingResponse.

    Attributes:
        project_id (str): Google Cloud project ID
        location_id (str): Location ID for the ranking service.
        grounding_config (str):
            Required. The resource name of the grounding config, such as
            ``default_grounding_config``.
            It is set to ``default_grounding_config`` by default if unspecified
        citation_threshold (float):
            The threshold (in [0,1]) used for determining whether a fact
            must be cited for a claim in the answer candidate. Choosing
            a higher threshold will lead to fewer but very strong
            citations, while choosing a lower threshold may lead to more
            but somewhat weaker citations. If unset, the threshold will
            default to 0.6.
        credentials (Optional[Credentials]): Google Cloud credentials object.
        credentials_path (Optional[str]): Path to the Google Cloud service
        account credentials file.
    """

    project_id: str = Field(default=None)
    location_id: str = Field(default="global")
    grounding_config: str = Field(default="default_grounding_config")
    citation_threshold: Optional[float] = Field(default=0.6)
    client: Any
    credentials: Optional[Credentials] = Field(default=None)
    credentials_path: Optional[str] = Field(default=None)

    class CheckGroundingResponse(BaseModel):
        support_score: float = 0.0
        cited_chunks: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        answer_with_citations: str = ""

    def __init__(self, **kwargs: Any):
        """
        Constructor for CheckGroundingOutputParser.
        Initializes the grounding check service client with necessary credentials
        and configurations.
        """
        super().__init__(**kwargs)
        self.client = kwargs.get("client")
        if not self.client:
            self.client = self._get_check_grounding_service_client()

    def _get_check_grounding_service_client(
        self,
    ) -> "discoveryengine_v1alpha.GroundedGenerationServiceClient":
        """
        Returns a GroundedGenerationServiceClient instance using provided credentials.
        Raises ImportError if necessary packages are not installed.

        Returns:
            A GroundedGenerationServiceClient instance.
        """
        try:
            from google.cloud import discoveryengine_v1alpha  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please install vertexaisearch dependency group: "
                "`pip install langchain-google-community[vertexaisearch]`"
            ) from exc
        return discoveryengine_v1alpha.GroundedGenerationServiceClient(
            credentials=(
                self.credentials
                or Credentials.from_service_account_file(self.credentials_path)  # type: ignore[attr-defined]
                if self.credentials_path
                else None
            ),
            client_info=get_client_info(module="vertex-ai-search"),
        )

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> CheckGroundingResponse:
        """
        Calls the Vertex Check Grounding API for a given answer candidate and a list
        of documents (claims) to validate whether the set of claims support the 
        answer candidate.

        Args:
            answer_candidate (str): The candidate answer to be evaluated for grounding.
            documents (List[Document]): The documents against which grounding is
            checked. This will be converted to facts:
                facts (MutableSequence[google.cloud.discoveryengine_v1alpha.types.\
                    GroundingFact]):
                List of facts for the grounding check.
                We support up to 200 facts.
        Returns:
            Response of the type CheckGroundingResponse

            Attributes:
            support_score (float):
                The support score for the input answer
                candidate. Higher the score, higher is the
                fraction of claims that are supported by the
                provided facts. This is always set when a
                response is returned.

            cited_chunks (MutableSequence[google.cloud.discoveryengine_v1alpha.types.\
                FactChunk]):
                List of facts cited across all claims in the
                answer candidate. These are derived from the
                facts supplied in the request.

            claims (MutableSequence[google.cloud.discoveryengine_v1alpha.types.\
                CheckGroundingResponse.Claim]):
                Claim texts and citation info across all
                claims in the answer candidate.
            
            answer_with_citations (str):
                Complete formed answer formatted with inline citations
        """
        from google.cloud import discoveryengine_v1alpha  # type: ignore

        answer_candidate = input
        documents = self.extract_documents(config)

        grounding_spec = discoveryengine_v1alpha.CheckGroundingSpec(
            citation_threshold=self.citation_threshold,
        )

        facts = [
            discoveryengine_v1alpha.GroundingFact(
                fact_text=doc.page_content,
                attributes={
                    key: value
                    for key, value in (
                        doc.metadata or {}
                    ).items()  # Use an empty dict if metadata is None
                    if key not in ["id", "relevance_score"] and value is not None
                },
            )
            for doc in documents
            if doc.page_content  # Only check that page_content is not None or empty
        ]

        if not facts:
            raise ValueError("No valid documents provided for grounding.")

        request = discoveryengine_v1alpha.CheckGroundingRequest(
            grounding_config=f"projects/{self.project_id}/locations/{self.location_id}/groundingConfigs/{self.grounding_config}",
            answer_candidate=answer_candidate,
            facts=facts,
            grounding_spec=grounding_spec,
        )

        if self.client is None:
            raise ValueError("Client not initialized.")
        try:
            response = self.client.check_grounding(request=request)
        except core_exceptions.GoogleAPICallError as e:
            raise RuntimeError(
                f"Error in Vertex AI Check Grounding API call: {str(e)}"
            ) from e

        support_score = response.support_score
        cited_chunks = [
            {
                "chunk_text": chunk.chunk_text,
                "source": documents[int(chunk.source)],
            }
            for chunk in response.cited_chunks
        ]
        claims = [
            {
                "start_pos": claim.start_pos,
                "end_pos": claim.end_pos,
                "claim_text": claim.claim_text,
                "citation_indices": list(claim.citation_indices),
            }
            for claim in response.claims
        ]

        answer_with_citations = self.combine_claims_with_citations(claims)
        return self.CheckGroundingResponse(
            support_score=support_score,
            cited_chunks=cited_chunks,
            claims=claims,
            answer_with_citations=answer_with_citations,
        )

    def extract_documents(self, config: Optional[RunnableConfig]) -> List[Document]:
        if not config:
            raise ValueError("Configuration is required.")

        potential_documents = config.get("configurable", {}).get("documents", [])
        if not isinstance(potential_documents, list) or not all(
            isinstance(doc, Document) for doc in potential_documents
        ):
            raise ValueError("Invalid documents. Each must be an instance of Document.")

        if not potential_documents:
            raise ValueError("This wrapper requires documents for processing.")

        return potential_documents

    def combine_claims_with_citations(self, claims: List[Dict[str, Any]]) -> str:
        sorted_claims = sorted(claims, key=lambda x: x["start_pos"])
        result = []
        for claim in sorted_claims:
            if claim["citation_indices"]:
                citations = "".join([f"[{idx}]" for idx in claim["citation_indices"]])
                claim_text = f"{claim['claim_text']}{citations}"
            else:
                claim_text = claim["claim_text"]
            result.append(claim_text)
        return " ".join(result).strip()

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "utilities", "check_grounding"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    class Config:
        extra = Extra.ignore
        arbitrary_types_allowed = True
