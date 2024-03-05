from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from pydantic.v1 import root_validator

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.embeddings import GoogleEmbeddingModelType


def test_langchain_google_vertexai_embed_image_multimodal_only() -> None:
    mock_embeddings = MockVertexAIEmbeddings("textembedding-gecko@001")
    assert mock_embeddings.model_type == GoogleEmbeddingModelType.TEXT
    with pytest.raises(NotImplementedError) as e:
        mock_embeddings.embed_image("test")
        assert e.value == "Only supported for multimodal models"


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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["client"] = MagicMock()
        return values
