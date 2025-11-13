from unittest.mock import MagicMock, patch

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.embeddings import EmbeddingTaskTypes


@patch("langchain_google_vertexai.embeddings.genai.Client")
def test_langchain_google_vertexai_supports_gemini_embedding_model(mock_client) -> None:
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(  # type: ignore
        model_name="gemini-embedding-001"
    )
    assert embeddings.model_name == "gemini-embedding-001"
    embeddings = VertexAIEmbeddings(model="gemini-embedding-001")
    assert embeddings.model_name == "gemini-embedding-001"


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_embed_documents_with_question_answering_task(
    mock_get_embeddings, mock_client
) -> None:
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-005")
    texts = ["text {i}" for i in range(5)]

    embedding_dimension = 768
    embeddings_task_type: EmbeddingTaskTypes = "QUESTION_ANSWERING"

    mock_get_embeddings.return_value = [[0.001] * embedding_dimension for _ in texts]

    result = embeddings.embed_documents(
        texts=texts, embeddings_task_type=embeddings_task_type
    )
    assert len(result) == len(texts)


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_embed_query_with_question_answering_task(
    mock_get_embeddings, mock_client
) -> None:
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-005")
    text = "text 0"

    embedding_dimension = 768
    embeddings_task_type: EmbeddingTaskTypes = "QUESTION_ANSWERING"

    mock_get_embeddings.return_value = [[0.001] * embedding_dimension]

    embedding = embeddings.embed_query(
        text=text, embeddings_task_type=embeddings_task_type
    )

    assert isinstance(embedding, list)
    assert len(embedding) == embedding_dimension

    mock_get_embeddings.assert_called_once_with(
        texts=[text], embeddings_type=embeddings_task_type, dimensions=None, title=None
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
def test_initialization_client_call(mock_client):
    VertexAIEmbeddings(
        model="textembedding-gecko@001",
        project="test-project",
        location="test-location",
    )
    mock_client.assert_called_once_with(
        vertexai=True,
        project="test-project",
        location="test-location",
        credentials=None,
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_embed_parameters(mock_get_embeddings, mock_client):
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    texts = ["hello", "world"]
    embeddings.embed(
        texts,
        embeddings_task_type="CLASSIFICATION",
        dimensions=128,
        title="test-title",
    )
    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        embeddings_type="CLASSIFICATION",
        dimensions=128,
        title="test-title",
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_default_dimensions_used_when_not_specified(mock_get_embeddings, mock_client):
    """Test that constructor dimensions are used when not specified in embed()."""
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004", dimensions=256)
    texts = ["hello", "world"]

    mock_get_embeddings.return_value = [[0.001] * 256 for _ in texts]

    embeddings.embed(texts)

    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        embeddings_type=None,
        dimensions=256,
        title=None,
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_explicit_dimensions_override_default(mock_get_embeddings, mock_client):
    """Test that explicit dimensions in embed() override constructor default."""
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004", dimensions=256)
    texts = ["hello", "world"]

    mock_get_embeddings.return_value = [[0.001] * 512 for _ in texts]

    embeddings.embed(texts, dimensions=512)

    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        embeddings_type=None,
        dimensions=512,
        title=None,
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_no_default_dimensions_works_as_before(mock_get_embeddings, mock_client):
    """Test backward compatibility when no default dimensions specified."""
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    texts = ["hello", "world"]

    mock_get_embeddings.return_value = [[0.001] * 768 for _ in texts]

    embeddings.embed(texts)

    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        embeddings_type=None,
        dimensions=None,
        title=None,
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_default_dimensions_used_in_embed_documents(mock_get_embeddings, mock_client):
    """Test that constructor dimensions are used in embed_documents()."""
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004", dimensions=128)
    texts = ["hello", "world"]

    mock_get_embeddings.return_value = [[0.001] * 128 for _ in texts]

    embeddings.embed_documents(texts)

    mock_get_embeddings.assert_called_once_with(
        texts=texts,
        embeddings_type="RETRIEVAL_DOCUMENT",
        dimensions=128,
        title=None,
    )


@patch("langchain_google_vertexai.embeddings.genai.Client")
@patch.object(VertexAIEmbeddings, "_get_embeddings_with_retry")
def test_default_dimensions_used_in_embed_query(mock_get_embeddings, mock_client):
    """Test that constructor dimensions are used in embed_query()."""
    mock_client.return_value = MagicMock()
    embeddings = VertexAIEmbeddings(model="text-embedding-004", dimensions=128)
    text = "hello"

    mock_get_embeddings.return_value = [[0.001] * 128]

    embeddings.embed_query(text)

    mock_get_embeddings.assert_called_once_with(
        texts=[text],
        embeddings_type="RETRIEVAL_QUERY",
        dimensions=128,
        title=None,
    )
