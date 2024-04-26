from typing import Any, Dict, List, Optional

from google.api_core import exceptions as core_exceptions  # type: ignore
from google.auth.credentials import Credentials  # type: ignore
from google.cloud import discoveryengine_v1alpha  # type: ignore
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import BaseModel, Extra, Field


class VertexCheckGroundingOutputParser(
    BaseOutputParser["VertexCheckGroundingOutputParser.CheckGroundingResponse"]
):
    """
    Initializes the Vertex AI CheckGroundingOutputParser with configurable parameters.
    Inherits from BaseOutputParser and calls the Check Grounding API to validate the
    response against a given set of documents and returns back citations that support
    the claims along with the cited chunks. Output is of the type
    CheckGroundingResponse.

    Attributes:
        project_id (str): Google Cloud project ID.
        location_id (str): Location ID for the ranking service.
        grounding_config (str): Required. The resource name of the grounding config.
        citation_threshold (float): The threshold for determining citation necessity.
        credentials (Optional[Credentials]): Google Cloud credentials object.
        credentials_path (Optional[str]): Path to the Google Cloud service account
        credentials file.
    """

    project_id: str = Field(default=None)
    location_id: str = Field(default="global")
    grounding_config: str = Field(default="default_grounding_config")
    citation_threshold: Optional[float] = Field(default=0.6)
    client: Optional[discoveryengine_v1alpha.GroundedGenerationServiceClient] = Field(
        default=None
    )
    credentials: Optional[Credentials] = Field(default=None)
    credentials_path: Optional[str] = Field(default=None)

    class CheckGroundingResponse(BaseModel):
        support_score: float = 0.0
        cited_chunks: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        answer_with_citations: str = ""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = kwargs.get("client")
        if not self.client:
            self.client = self._get_check_grounding_service_client()

    def _get_check_grounding_service_client(
        self,
    ) -> discoveryengine_v1alpha.GroundedGenerationServiceClient:
        try:
            return discoveryengine_v1alpha.GroundedGenerationServiceClient(
                credentials=(
                    self.credentials
                    or Credentials.from_service_account_file(self.credentials_path)
                    if self.credentials_path
                    else None
                )
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-discoveryengine python package. "
                "Please, install vertexaisearch dependency group: "
                "`pip install langchain-google-community[vertexaisearch]`"
            ) from exc

    def parse(
        self, answer_candidate: str, documents: Optional[List[Document]] = None
    ) -> CheckGroundingResponse:
        if documents is None:
            raise NotImplementedError("This parser requires documents for processing.")

        grounding_spec = discoveryengine_v1alpha.CheckGroundingSpec(
            citation_threshold=self.citation_threshold,
        )

        facts = [
            discoveryengine_v1alpha.GroundingFact(
                fact_text=doc.page_content,
                attributes={
                    key: value
                    for key, value in doc.metadata.items()
                    if key not in ["id", "relevance_score"] and value is not None
                },
            )
            for doc in documents
            if doc.page_content and doc.metadata
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

    def get_format_instructions(self) -> str:
        return "The output should be of the type CheckGroundingResponse."

    @property
    def _type(self) -> str:
        return "check_grounding_output_parser"

    class Config:
        extra = Extra.ignore
        arbitrary_types_allowed = True
