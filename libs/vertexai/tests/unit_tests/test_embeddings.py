from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic import model_validator
from typing_extensions import Self

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.embeddings import GoogleEmbeddingModelType


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


class MockVertexAIEmbeddings(VertexAIEmbeddings):
    """
    A mock class for avoiding instantiating VertexAI and the EmbeddingModel client
    instance during init
    """

    def __init__(self, model_name, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        pass

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        self.client = MagicMock()
        return self
