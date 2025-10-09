"""Test Vertex AI API wrapper.

Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
"""

import pytest

from langchain_google_vertexai.embeddings import (
    VertexAIEmbeddings,
)

_DEFAULT_MODEL = "gemini-embedding-001"
_EMBEDDING_MODELS = [
    (_DEFAULT_MODEL, 3072),
]


@pytest.mark.release
def test_initialization() -> None:
    """Test embedding model initialization."""
    for embeddings in [
        VertexAIEmbeddings(
            model_name=_DEFAULT_MODEL,  # type: ignore
        ),
        VertexAIEmbeddings(model=_DEFAULT_MODEL),
    ]:
        assert embeddings.model_name == _DEFAULT_MODEL


@pytest.mark.release
@pytest.mark.parametrize(
    "number_of_docs",
    [1, 8],
)
@pytest.mark.parametrize(
    ("model_name", "embeddings_dim"),
    _EMBEDDING_MODELS,
)
def test_langchain_google_vertexai_embedding_documents(
    number_of_docs: int, model_name: str, embeddings_dim: int
) -> None:
    documents = ["foo bar"] * number_of_docs
    model = VertexAIEmbeddings(model=model_name)
    output = model.embed_documents(documents)
    assert len(output) == number_of_docs
    for embedding in output:
        assert len(embedding) == embeddings_dim


@pytest.mark.release
@pytest.mark.parametrize(
    ("model_name", "embeddings_dim"),
    _EMBEDDING_MODELS,
)
def test_langchain_google_vertexai_embedding_documents_with_task_type(
    model_name: str,
    embeddings_dim: int,
) -> None:
    documents = ["foo bar"] * 8
    model = VertexAIEmbeddings(model=model_name)
    output = model.embed_documents(documents)
    assert len(output) == 8
    for embedding in output:
        assert len(embedding) == embeddings_dim


@pytest.mark.release
@pytest.mark.parametrize(
    ("model_name", "embeddings_dim"),
    _EMBEDDING_MODELS,
)
def test_langchain_google_vertexai_embedding_query(model_name, embeddings_dim) -> None:
    document = "foo bar"
    model = VertexAIEmbeddings(model=model_name)
    output = model.embed_query(document)
    assert len(output) == embeddings_dim


@pytest.mark.release
@pytest.mark.parametrize(
    ("model_name", "embeddings_dim"),
    _EMBEDDING_MODELS,
)
def test_langchain_google_vertexai_embedding_query_with_task_type(
    model_name: str,
    embeddings_dim: int,
) -> None:
    document = "foo bar"
    model = VertexAIEmbeddings(model=model_name)
    output = model.embed_query(document)
    assert len(output) == embeddings_dim


@pytest.mark.release
@pytest.mark.parametrize(
    ("model_name", "embeddings_dim"),
    [
        (_DEFAULT_MODEL, 768),
        ("text-multilingual-embedding-002", 768),
        (_DEFAULT_MODEL, 3072),
    ],
)
def test_langchain_google_vertexai_embedding_with_output_dimensionality(
    model_name: str, embeddings_dim: int
) -> None:
    model = VertexAIEmbeddings(model=model_name)
    output = model.embed(
        texts=["foo bar"],
        dimensions=embeddings_dim,
    )
    assert len(output) == 1
    for embedding in output:
        assert len(embedding) == embeddings_dim
