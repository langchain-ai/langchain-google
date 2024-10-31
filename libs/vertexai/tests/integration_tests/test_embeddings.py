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

_EMBEDDING_MODELS = [
    ("text-embedding-004", 768),
    ("multimodalembedding@001", 1408),
]


@pytest.mark.release
def test_initialization() -> None:
    """Test embedding model initialization."""
    for embeddings in [
        VertexAIEmbeddings(
            model_name="text-embedding-004",
        ),
        VertexAIEmbeddings(model="text-embedding-004"),
    ]:
        assert embeddings.model_name == "text-embedding-004"


@pytest.mark.release
@pytest.mark.parametrize(
    "number_of_docs",
    [1, 8],
)
@pytest.mark.parametrize(
    "model_name, embeddings_dim",
    _EMBEDDING_MODELS,
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
    _EMBEDDING_MODELS,
)
def test_langchain_google_vertexai_embedding_query(model_name, embeddings_dim) -> None:
    document = "foo bar"
    model = VertexAIEmbeddings(model_name)
    output = model.embed_query(document)
    assert len(output) == embeddings_dim


@pytest.mark.release
@pytest.mark.parametrize(
    "dim, expected_dim",
    [(None, 1408), (512, 512)],
)
def test_langchain_google_vertexai_image_embeddings(
    dim, expected_dim, base64_image
) -> None:
    model = VertexAIEmbeddings(model_name="multimodalembedding")
    kwargs = {}
    if dim:
        kwargs["dimensions"] = dim
    output = model.embed_images([base64_image for i in range(3)], **kwargs)
    assert len(output) == 3
    assert len(output[0]) == expected_dim


@pytest.mark.release
def test_langchain_google_vertexai_text_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="text-embedding-004")
    assert isinstance(embeddings_model.client, TextEmbeddingModel)
    assert embeddings_model.model_type == GoogleEmbeddingModelType.TEXT


@pytest.mark.release
def test_langchain_google_vertexai_multimodal_model() -> None:
    embeddings_model = VertexAIEmbeddings(model_name="multimodalembedding@001")
    assert isinstance(embeddings_model.client, MultiModalEmbeddingModel)
    assert embeddings_model.model_type == GoogleEmbeddingModelType.MULTIMODAL


@pytest.mark.release
@pytest.mark.parametrize(
    "model_name, embeddings_dim",
    [("text-embedding-004", 768), ("text-multilingual-embedding-002", 768)],
)
def test_langchain_google_vertexai_embedding_with_output_dimensionality(
    model_name: str, embeddings_dim: int
) -> None:
    model = VertexAIEmbeddings(model_name)
    output = model.embed(
        texts=["foo bar"],
        dimensions=embeddings_dim,
    )
    assert len(output) == 1
    for embedding in output:
        assert len(embedding) == embeddings_dim
    assert model.model_name == model.client._model_id
    assert model.model_name == model_name
