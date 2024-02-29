from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_google_vertexai import VertexAIEmbeddings


def test_langchain_google_vertexai_embed_image_multimodal_only() -> None:
    mock_embeddings = MockVertexAIEmbeddings("textembedding-gecko@001")
    with pytest.raises(NotImplementedError) as e:
        mock_embeddings.embed_image("test")
        assert e.value == "Only supported for multimodal models"


def test_langchain_google_vertexai_embed_documents_text_only() -> None:
    mock_embeddings = MockVertexAIEmbeddings("multimodalembedding@001")
    with pytest.raises(NotImplementedError) as e:
        mock_embeddings.embed_documents(["test"])
        assert e.value == "Not supported for multimodal models"


def test_langchain_google_vertexai_embed_query_text_only() -> None:
    mock_embeddings = MockVertexAIEmbeddings("multimodalembedding@001")
    with pytest.raises(NotImplementedError) as e:
        mock_embeddings.embed_query("test")
        assert e.value == "Not supported for multimodal models"


class MockVertexAIEmbeddings(VertexAIEmbeddings):
    """
    A mock class for avoiding instantiating VertexAI and the EmbeddingModel client
    instance during init
    """

    def __init__(self, model_name, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.client = MagicMock()
