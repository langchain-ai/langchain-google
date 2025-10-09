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
