"""Test `GoogleGenerativeAIEmbeddings`."""

import numpy as np
import pytest

from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

_MODEL = "gemini-embedding-001"
_OUTPUT_DIMENSIONALITY = 768


@pytest.mark.parametrize(
    "query",
    [
        "Hi",
        "This is a longer query string to test the embedding functionality of the"
        " model against the pickle rick?",
    ],
)
def test_embed_query_different_lengths(query: str, backend_config: dict) -> None:
    """Test embedding queries of different lengths."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    result = model.embed_query(query, output_dimensionality=_OUTPUT_DIMENSIONALITY)
    assert len(result) == 768
    assert isinstance(result, list)


@pytest.mark.parametrize(
    "query",
    [
        "Hi",
        "This is a longer query string to test the embedding functionality of the"
        " model against the pickle rick?",
    ],
)
@pytest.mark.asyncio
async def test_aembed_query_different_lengths(query: str, backend_config: dict) -> None:
    """Test embedding queries of different lengths."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    result = await model.aembed_query(
        query, output_dimensionality=_OUTPUT_DIMENSIONALITY
    )
    assert len(result) == 768
    assert isinstance(result, list)


def test_embed_documents(backend_config: dict) -> None:
    """Test embedding a query."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    result = model.embed_documents(
        ["Hello world", "Good day, world"], output_dimensionality=_OUTPUT_DIMENSIONALITY
    )
    assert len(result) == 2
    assert len(result[0]) == 768
    assert len(result[1]) == 768
    assert isinstance(result, list)
    assert isinstance(result[0], list)


@pytest.mark.asyncio
async def test_aembed_documents(backend_config: dict) -> None:
    """Asynchronously test embedding a query."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    result = await model.aembed_documents(
        ["Hello world", "Good day, world"], output_dimensionality=_OUTPUT_DIMENSIONALITY
    )
    assert len(result) == 2
    assert len(result[0]) == 768
    assert len(result[1]) == 768
    assert isinstance(result, list)
    assert isinstance(result[0], list)


def test_invalid_model_error_handling(backend_config: dict) -> None:
    """Test error handling with an invalid model name."""
    with pytest.raises(GoogleGenerativeAIError):
        model = GoogleGenerativeAIEmbeddings(model="invalid_model", **backend_config)
        model.embed_query("Hello world", output_dimensionality=_OUTPUT_DIMENSIONALITY)


def test_invalid_api_key_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling with an invalid API key.

    Note: This test only runs on Google AI backend (not Vertex AI) since it tests
    API key authentication specifically.
    """
    # Set an invalid API key in the environment
    monkeypatch.setenv("GOOGLE_API_KEY", "invalid_key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GoogleGenerativeAIError):
        GoogleGenerativeAIEmbeddings(model=_MODEL).embed_query(
            "Hello world", output_dimensionality=_OUTPUT_DIMENSIONALITY
        )


def test_embed_documents_consistency(backend_config: dict) -> None:
    """Test embedding consistency for the same document."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    doc = "Consistent document for testing"
    result1 = model.embed_documents([doc], output_dimensionality=_OUTPUT_DIMENSIONALITY)
    result2 = model.embed_documents([doc], output_dimensionality=_OUTPUT_DIMENSIONALITY)
    assert result1 == result2


def test_embed_documents_quality(backend_config: dict) -> None:
    """Smoke test embedding quality by comparing similar and dissimilar documents."""
    model = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    similar_docs = ["Document A", "Similar Document A"]
    dissimilar_docs = ["Document A", "Completely Different Zebra"]
    similar_embeddings = model.embed_documents(
        similar_docs, output_dimensionality=_OUTPUT_DIMENSIONALITY
    )
    dissimilar_embeddings = model.embed_documents(
        dissimilar_docs, output_dimensionality=_OUTPUT_DIMENSIONALITY
    )
    similar_distance = np.linalg.norm(
        np.array(similar_embeddings[0]) - np.array(similar_embeddings[1])
    )
    dissimilar_distance = np.linalg.norm(
        np.array(dissimilar_embeddings[0]) - np.array(dissimilar_embeddings[1])
    )
    assert similar_distance < dissimilar_distance


def test_embed_query_task_type(backend_config: dict) -> None:
    """Test for task_type."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=_MODEL, task_type="clustering", **backend_config
    )
    emb = embeddings.embed_query(
        "How does alphafold work?", output_dimensionality=_OUTPUT_DIMENSIONALITY
    )

    embeddings2 = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    emb2 = embeddings2.embed_query(
        "How does alphafold work?",
        task_type="clustering",
        output_dimensionality=_OUTPUT_DIMENSIONALITY,
    )

    embeddings3 = GoogleGenerativeAIEmbeddings(model=_MODEL, **backend_config)
    emb3 = embeddings3.embed_query(
        "How does alphafold work?", output_dimensionality=_OUTPUT_DIMENSIONALITY
    )

    assert emb == emb2
    assert emb != emb3
