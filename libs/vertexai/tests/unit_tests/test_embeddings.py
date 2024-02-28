import pytest
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from langchain_google_vertexai import VertexAIEmbeddings


def test_langchain_google_vertexai_text_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    assert isinstance(embeddings_model.client, TextEmbeddingModel)
    assert not embeddings_model._is_multimodal_model(embeddings_model.model_name)


def test_langchain_google_vertexai_multimodal_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    assert isinstance(embeddings_model.client, MultiModalEmbeddingModel)
    assert embeddings_model._is_multimodal_model(embeddings_model.model_name)


def test_langchain_google_vertexai_embed_image_multimodal_only() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    with pytest.raises(NotImplementedError) as e:
        embeddings_model.embed_image("test")
        assert e.value == "Only supported for multimodal models"


def test_langchain_google_vertexai_embed_documents_text_only() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    with pytest.raises(NotImplementedError) as e:
        embeddings_model.embed_documents(["test"])
        assert e.value == "Not supported for multimodal models"


def test_langchain_google_vertexai_embed_query_text_only() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    with pytest.raises(NotImplementedError) as e:
        embeddings_model.embed_query("test")
        assert e.value == "Not supported for multimodal models"

