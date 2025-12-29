import logging
from collections.abc import Callable
from typing import Any, Literal

from google import genai
from google.genai.types import EmbedContentConfig
from langchain_core._api import deprecated
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self
from typing_extensions import deprecated as typing_deprecated

from langchain_google_vertexai._utils import create_retry_decorator

logger = logging.getLogger(__name__)


EmbeddingTaskTypes = Literal[
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
]


@typing_deprecated(
    "Use [`GoogleGenerativeAIEmbeddings`][langchain_google_genai.GoogleGenerativeAIEmbeddings] "  # noqa: E501
    "instead."
)
@deprecated(
    since="3.2.0",
    removal="4.0.0",
    alternative_import="langchain_google_genai.GoogleGenerativeAIEmbeddings",
)
class VertexAIEmbeddings(BaseModel, Embeddings):
    """Google Cloud VertexAI embedding models."""

    client: Any = Field(default=None, exclude=True)

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    project: str | None = None
    """The default GCP project to use when making Vertex API calls."""

    location: str = Field(default="us-central1")
    """The default location to use when making API calls."""

    model_name: str | None = Field(default=None, alias="model")
    """Underlying model name."""

    credentials: Any = Field(default=None, exclude=True)
    """The default custom credentials to use when making API calls.

    (`google.auth.credentials.Credentials`)

    If not provided, credentials will be ascertained from the environment.
    """

    max_retries: int = 6
    """The maximum number of retries to make when generating."""
    dimensions: int | None = None
    """Default output dimensionality for embeddings. If not specified, uses the
    model's default. Can be overridden per request in embed() method."""

    @model_validator(mode="before")
    @classmethod
    def validate_params_base(cls, values: dict) -> Any:
        if "model_name" in values and "model" not in values:
            values["model"] = values.pop("model_name")
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates that the python package exists in environment."""
        if self.model_name is None:
            msg = "model_name must be provided for VertexAI embeddings"
            raise ValueError(msg)
        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            credentials=self.credentials,
        )
        return self

    def _get_embeddings_with_retry(
        self,
        texts: list[str],
        embeddings_type: str | None = None,
        dimensions: int | None = None,
        title: str | None = None,
    ) -> list[list[float]]:
        """Makes a Vertex AI model request with retry logic."""
        retry_decorator = create_retry_decorator(max_retries=self.max_retries)

        @retry_decorator
        def _completion_with_retry_inner(
            generation_method: Callable, **kwargs: Any
        ) -> Any:
            return generation_method(**kwargs)

        params = {
            "model": self.model_name,
            "contents": texts,
            "config": EmbedContentConfig(
                task_type=embeddings_type, output_dimensionality=dimensions, title=title
            ),
        }
        embeddings = _completion_with_retry_inner(
            self.client.models.embed_content,
            **params,
        )
        return [e.values for e in embeddings.embeddings]

    def embed(
        self,
        texts: list[str],
        embeddings_task_type: EmbeddingTaskTypes | None = None,
        dimensions: int | None = None,
        title: str | None = None,
    ) -> list[list[float]]:
        """Embed a list of strings.

        Args:
            texts: The list of strings to embed.
            embeddings_task_type: Optional embeddings task type, one of the following:

                - `RETRIEVAL_QUERY` - Text is a query in a search/retrieval setting
                - `RETRIEVAL_DOCUMENT` - Text is a document in a search/retrieval
                    setting
                - `SEMANTIC_SIMILARITY` - Embeddings will be used for Semantic Textual
                    Similarity (STS).
                - `CLASSIFICATION` - Embeddings will be used for classification.
                - `CLUSTERING` - Embeddings will be used for clustering.
                - `CODE_RETRIEVAL_QUERY` - Embeddings will be used for code retrieval
                    for Java and Python.

                The following are only supported on preview models:
                    `QUESTION_ANSWERING`, `FACT_VERIFICATION`.
            dimensions: Output embeddings dimensions.

                Only supported on preview models. If not provided, uses the
                default dimensions specified in the constructor.
            title: Title for the text.

                Only applicable when `TaskType` is `RETRIEVAL_DOCUMENT`.

        Returns:
            List of embeddings, one for each text.
        """
        if len(texts) == 0:
            return []
        effective_dimensions = dimensions if dimensions is not None else self.dimensions
        embeddings = self._get_embeddings_with_retry(
            texts=texts,
            embeddings_type=embeddings_task_type,
            dimensions=effective_dimensions,
            title=title,
        )
        return embeddings

    def embed_documents(
        self,
        texts: list[str],
        *,
        embeddings_task_type: EmbeddingTaskTypes = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: The list of texts to embed.
            embeddings_task_type: The task type for embeddings.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts, embeddings_task_type, dimensions=self.dimensions)

    def embed_query(
        self,
        text: str,
        *,
        embeddings_task_type: EmbeddingTaskTypes = "RETRIEVAL_QUERY",
    ) -> list[float]:
        """Embed a text.

        Args:
            text: The text to embed.
            embeddings_task_type: The task type for embeddings.

        Returns:
            Embedding for the text.
        """
        return self.embed([text], embeddings_task_type, dimensions=self.dimensions)[0]
