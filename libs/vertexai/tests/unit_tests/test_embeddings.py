from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from pydantic import model_validator
from typing_extensions import Self

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.embeddings import (
    EmbeddingTaskTypes,
    GoogleEmbeddingModelType,
)


def test_langchain_google_vertexai_supports_gemini_embedding_model() -> None:
    mock_embeddings = MockVertexAIEmbeddings("gemini-embedding-001")
    assert mock_embeddings.model_type == GoogleEmbeddingModelType.TEXT


def test_langchain_google_vertexai_embed_image_multimodal_only() -> None:
    mock_embeddings = MockVertexAIEmbeddings("textembedding-gecko@001")
    assert mock_embeddings.model_type == GoogleEmbeddingModelType.TEXT
    with pytest.raises(NotImplementedError) as e:
        mock_embeddings.embed_images(["test"])[0]
        assert e.value == "Only supported for multimodal models"


def test_langchain_google_vertexai_no_dups_dynamic_batch_size() -> None:
    mock_embeddings = MockVertexAIEmbeddings("textembedding-gecko@001")
    default_batch_size = mock_embeddings.instance["batch_size"]
    texts = ["text {i}" for i in range(default_batch_size * 2)]
    # It should only return one batch (out of two) still to process
    _, batches = mock_embeddings._prepare_and_validate_batches(texts=texts)
    assert len(batches) == 1
    # The second time it should return the batches unchanged
    _, batches = mock_embeddings._prepare_and_validate_batches(texts=texts)
    assert len(batches) == 2


@patch.object(VertexAIEmbeddings, "embed")
def test_embed_documents_with_question_answering_task(mock_embed) -> None:
    mock_embeddings = MockVertexAIEmbeddings("text-embedding-005")
    texts = [f"text {i}" for i in range(5)]

    embedding_dimension = 768
    embeddings_task_type: EmbeddingTaskTypes = "QUESTION_ANSWERING"

    mock_embed.return_value = [[0.001] * embedding_dimension for _ in texts]

    embeddings = mock_embeddings.embed_documents(
        texts=texts, embeddings_task_type=embeddings_task_type
    )

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert len(embeddings[0]) == embedding_dimension

    # Verify embed() was called correctly
    mock_embed.assert_called_once_with(texts, 0, embeddings_task_type)


@patch.object(VertexAIEmbeddings, "embed")
def test_embed_query_with_question_answering_task(mock_embed) -> None:
    mock_embeddings = MockVertexAIEmbeddings("text-embedding-005")
    text = "text 0"

    embedding_dimension = 768
    embeddings_task_type: EmbeddingTaskTypes = "QUESTION_ANSWERING"

    mock_embed.return_value = [[0.001] * embedding_dimension]

    embedding = mock_embeddings.embed_query(
        text=text, embeddings_task_type=embeddings_task_type
    )

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_dimension

    # Verify embed() was called correctly
    mock_embed.assert_called_once_with([text], 1, embeddings_task_type)


class MockVertexAIEmbeddings(VertexAIEmbeddings):
    """
    A mock class for avoiding instantiating VertexAI and the EmbeddingModel client
    instance during init
    """

    def __init__(self, model_name, **kwargs: Any) -> None:
        super().__init__(model_name, project="test-proj", **kwargs)

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        pass

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        self.client = MagicMock()
        return self
