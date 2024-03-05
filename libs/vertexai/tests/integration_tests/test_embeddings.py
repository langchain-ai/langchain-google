"""Test Vertex AI API wrapper.

Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
"""
import pytest
from vertexai.language_models import TextEmbeddingModel  # type: ignore
from vertexai.vision_models import MultiModalEmbeddingModel  # type: ignore

from langchain_google_vertexai.embeddings import (
    GoogleEmbeddingModelType,
    VertexAIEmbeddings,
)


@pytest.mark.release
def test_initialization() -> None:
    """Test embedding model initialization."""
    VertexAIEmbeddings()


@pytest.mark.release
@pytest.mark.parametrize(
    "number_of_docs",
    [1, 8],
)
@pytest.mark.parametrize(
    "model_name, embeddings_dim",
    [("textembedding-gecko@001", 768), ("multimodalembedding@001", 1408)],
)
def test_langchain_google_vertexai_embedding_documents(
    number_of_docs: int, model_name: str, embeddings_dim: int
) -> None:
    documents = ["foo bar"] * number_of_docs
    model = VertexAIEmbeddings(model_name)
    output = model.embed_documents(documents)
    assert len(output) == number_of_docs
    for embedding in output:
        assert len(embedding) == embeddings_dim
    assert model.model_name == model.client._model_id
    assert model.model_name == model_name


@pytest.mark.release
@pytest.mark.parametrize(
    "model_name, embeddings_dim",
    [("textembedding-gecko@001", 768), ("multimodalembedding@001", 1408)],
)
def test_langchain_google_vertexai_embedding_query(model_name, embeddings_dim) -> None:
    document = "foo bar"
    model = VertexAIEmbeddings(model_name)
    output = model.embed_query(document)
    assert len(output) == embeddings_dim


@pytest.mark.release
def test_langchain_google_vertexai_large_batches() -> None:
    documents = ["foo bar" for _ in range(0, 251)]
    model_uscentral1 = VertexAIEmbeddings(location="us-central1")
    model_asianortheast1 = VertexAIEmbeddings(location="asia-northeast1")
    model_uscentral1.embed_documents(documents)
    model_asianortheast1.embed_documents(documents)
    assert model_uscentral1.instance["batch_size"] >= 250
    assert model_asianortheast1.instance["batch_size"] < 50


@pytest.mark.release
def test_warning(caplog: pytest.LogCaptureFixture) -> None:
    _ = VertexAIEmbeddings()
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    expected_message = (
        "Model_name will become a required arg for VertexAIEmbeddings starting from "
        "Feb-01-2024. Currently the default is set to textembedding-gecko@001"
    )
    assert record.message == expected_message


@pytest.mark.release
def test_langchain_google_vertexai_image_embeddings(tmp_image) -> None:
    model = VertexAIEmbeddings(model_name="multimodalembedding")
    output = model.embed_image(tmp_image)
    assert len(output) == 1408


@pytest.mark.release
def test_langchain_google_vertexai_text_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    assert isinstance(embeddings_model.client, TextEmbeddingModel)
    assert embeddings_model.model_type == GoogleEmbeddingModelType.TEXT


@pytest.mark.release
def test_langchain_google_vertexai_multimodal_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    assert isinstance(embeddings_model.client, MultiModalEmbeddingModel)
    assert embeddings_model.model_type == GoogleEmbeddingModelType.MULTIMODAL
