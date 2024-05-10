from typing import Any, Dict, List, Optional

# TODO: remove ignore once the google package is published with types
from google.ai.generativelanguage_v1beta.types import (
    BatchEmbedContentsRequest,
    EmbedContentRequest,
)
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_google_genai._common import (
    GoogleGenerativeAIError,
    get_client_info,
)
from langchain_google_genai._genai_extension import build_generative_service


class GoogleGenerativeAIEmbeddings(BaseModel, Embeddings):
    """`Google Generative AI Embeddings`.

    To use, you must have either:

        1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
        2. Pass your API key using the google_api_key kwarg to the ChatGoogle
           constructor.

    Example:
        .. code-block:: python

            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings.embed_query("What's our Q1 revenue?")
    """

    client: Any  #: :meta private:
    model: str = Field(
        ...,
        description="The name of the embedding model to use. "
        "Example: models/embedding-001",
    )
    task_type: Optional[str] = Field(
        None,
        description="The task type. Valid options include: "
        "task_type_unspecified, retrieval_query, retrieval_document, "
        "semantic_similarity, classification, and clustering",
    )
    google_api_key: Optional[SecretStr] = Field(
        None,
        description="The Google API key to use. If not provided, "
        "the GOOGLE_API_KEY environment variable will be used.",
    )
    credentials: Any = Field(
        default=None,
        exclude=True,
        description="The default custom credentials "
        "(google.auth.credentials.Credentials) to use when making API calls. If not "
        "provided, credentials will be ascertained from the GOOGLE_API_KEY envvar",
    )
    client_options: Optional[Dict] = Field(
        None,
        description=(
            "A dictionary of client options to pass to the Google API client, "
            "such as `api_endpoint`."
        ),
    )
    transport: Optional[str] = Field(
        None,
        description="A string, one of: [`rest`, `grpc`, `grpc_asyncio`].",
    )
    request_options: Optional[Dict] = Field(
        None,
        description="A dictionary of request options to pass to the Google API client."
        "Example: `{'timeout': 10}`",
    )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates params and passes them to google-generativeai package."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        client_info = get_client_info("GoogleGenerativeAIEmbeddings")

        values["client"] = build_generative_service(
            credentials=values.get("credentials"),
            api_key=google_api_key,
            client_info=client_info,
            client_options=values.get("client_options"),
        )
        return values

    def _prepare_request(
        self,
        text: str,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> EmbedContentRequest:
        task_type = self.task_type or task_type or "RETRIEVAL_DOCUMENT"
        # https://ai.google.dev/api/rest/v1/models/batchEmbedContents#EmbedContentRequest
        request = EmbedContentRequest(
            content={"parts": [{"text": text}]},
            model=self.model,
            task_type=task_type.upper(),
            title=title,
            output_dimensionality=output_dimensionality,
        )
        return request

    def embed_documents(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
        titles: Optional[List[str]] = None,
        output_dimensionality: Optional[int] = None,
    ) -> List[List[float]]:
        """Embed a list of strings. Vertex AI currently
        sets a max batch size of 5 strings.

        Args:
            texts: List[str] The list of strings to embed.
            batch_size: [int] The batch size of embeddings to send to the model
            task_type: task_type (https://ai.google.dev/api/rest/v1/TaskType)
            titles: An optional list of titles for texts provided.
            Only applicable when TaskType is RETRIEVAL_DOCUMENT.
            output_dimensionality: Optional reduced dimension for the output embedding.
            https://ai.google.dev/api/rest/v1/models/batchEmbedContents#EmbedContentRequest

        Returns:
            List of embeddings, one for each text.
        """
        titles = titles if titles else [None] * len(texts)  # type: ignore[list-item]
        requests = [
            self._prepare_request(
                text=text,
                task_type=task_type,
                title=title,
                output_dimensionality=output_dimensionality,
            )
            for text, title in zip(texts, titles)
        ]

        try:
            result = self.client.batch_embed_contents(
                BatchEmbedContentsRequest(requests=requests, model=self.model)
            )
        except Exception as e:
            raise GoogleGenerativeAIError(f"Error embedding content: {e}") from e
        return [e.values for e in result.embeddings]

    def embed_query(
        self,
        text: str,
        task_type: Optional[str] = None,
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> List[float]:
        """Embed a text.

        Args:
            text: The text to embed.
            task_type: task_type (https://ai.google.dev/api/rest/v1/TaskType)
            title: An optional title for the text.
            Only applicable when TaskType is RETRIEVAL_DOCUMENT.
            output_dimensionality: Optional reduced dimension for the output embedding.
            https://ai.google.dev/api/rest/v1/models/batchEmbedContents#EmbedContentRequest

        Returns:
            Embedding for the text.
        """
        task_type = self.task_type or "RETRIEVAL_QUERY"
        return self.embed_documents(
            [text],
            task_type=task_type,
            titles=[title] if title else None,
            output_dimensionality=output_dimensionality,
        )[0]
