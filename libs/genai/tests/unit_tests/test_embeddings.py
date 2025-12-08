"""Test embeddings model integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

MODEL_NAME = "gemini-embedding-001"


def _mock_embedding_response(values_list: list[list[float]]) -> MagicMock:
    """Create a mock embedding response with the given values."""
    mock_response = MagicMock()
    mock_embeddings = []
    for values in values_list:
        mock_embedding = MagicMock()
        mock_embedding.values = values
        mock_embeddings.append(mock_embedding)
    mock_response.embeddings = mock_embeddings
    return mock_response


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client:
        _ = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("..."),
        )
        mock_client.assert_called_once()
        # Check that http_options contains user agent
        call_kwargs = mock_client.call_args.kwargs
        assert "http_options" in call_kwargs
        http_options = call_kwargs["http_options"]
        assert "User-Agent" in http_options.headers
        assert "langchain-google-genai" in http_options.headers["User-Agent"]
        assert "GoogleGenerativeAIEmbeddings" in http_options.headers["User-Agent"]

    with patch("langchain_google_genai.embeddings.Client") as mock_client:
        _ = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("..."),
            task_type="retrieval_document",
        )
        mock_client.assert_called_once()


def test_api_key_is_string() -> None:
    with patch("langchain_google_genai.embeddings.Client"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("secret-api-key"),
        )
        assert isinstance(embeddings.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: pytest.CaptureFixture,
) -> None:
    with patch("langchain_google_genai.embeddings.Client"):
        embeddings = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("secret-api-key"),
        )
    print(embeddings.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_embed_query() -> None:
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = MagicMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0]])
        mock_client.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
            task_type="classification",
        )
        result = llm.embed_query("test text", output_dimensionality=524)

        # Verify the call was made with correct parameters
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == MODEL_NAME
        assert call_kwargs["contents"] == "test text"
        assert call_kwargs["config"].task_type == "CLASSIFICATION"
        assert call_kwargs["config"].output_dimensionality == 524

        # Verify the result
        assert result == [1.0, 2.0]


def test_embed_documents() -> None:
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = MagicMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0], [3.0, 4.0]])
        mock_client.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )

        result = llm.embed_documents(["test text", "test text2"])

        # Verify the call was made with correct parameters
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == MODEL_NAME
        assert call_kwargs["contents"] == ["test text", "test text2"]
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

        # Verify the result
        assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_embed_documents_with_numerous_texts() -> None:
    test_corpus_size = 100
    test_batch_size = 20
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = MagicMock()
        # Return embeddings for each batch
        mock_embed.return_value = _mock_embedding_response(
            [[1.0] for _ in range(test_batch_size)]
        )
        mock_client.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )

        llm.embed_documents(
            ["test text" for _ in range(test_corpus_size)],
            batch_size=test_batch_size,
        )

        # Should be called once per batch
        assert mock_embed.call_count == test_corpus_size / test_batch_size


def test_base_url_support() -> None:
    """Test that `base_url` is properly passed to the Client."""
    base_url = "https://example.com"
    param_api_key = "test-api-key"
    param_secret_api_key = SecretStr(param_api_key)

    with patch("langchain_google_genai.embeddings.Client") as mock_client:
        _ = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            base_url=base_url,
        )

    mock_client.assert_called_once()
    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs["http_options"].base_url == base_url
    # Verify api_key is passed (could be from param or env, but should be present)
    assert "api_key" in call_kwargs


def test_embed_query_default_task_type() -> None:
    """Test that embed_query uses default `RETRIEVAL_QUERY` when `task_type` is
    `None`."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = MagicMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0]])
        mock_client.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )
        llm.embed_query("test text")

        # Verify RETRIEVAL_QUERY is used as default
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"


def test_vertexai_backend() -> None:
    """Test that Vertex AI backend is properly configured."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client:
        _ = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
            project="test-project",
            vertexai=True,
        )

    mock_client.assert_called_once()
    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs["vertexai"] is True
    assert call_kwargs["project"] == "test-project"


def test_vertexai_auto_detection_with_project() -> None:
    """Test that Vertex AI is auto-detected when project is provided."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client:
        _ = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
            project="test-project",
        )

    mock_client.assert_called_once()
    call_kwargs = mock_client.call_args.kwargs
    assert call_kwargs["vertexai"] is True


def test_embed_documents_default_task_type() -> None:
    """Test that embed_documents uses default `RETRIEVAL_DOCUMENT` when `task_type` is
    `None`."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = MagicMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0]])
        mock_client.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )
        llm.embed_documents(["test text"])

        # Verify RETRIEVAL_DOCUMENT is used as default
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


@pytest.mark.asyncio
async def test_aembed_query() -> None:
    """Test async embed_query."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = AsyncMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0]])
        mock_client.aio.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )
        result = await llm.aembed_query("test text")

        # Verify the call was made with correct parameters
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == MODEL_NAME
        assert call_kwargs["contents"] == "test text"
        assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"

        # Verify the result
        assert result == [1.0, 2.0]


@pytest.mark.asyncio
async def test_aembed_documents() -> None:
    """Test async embed_documents."""
    with patch("langchain_google_genai.embeddings.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_embed = AsyncMock()
        mock_embed.return_value = _mock_embedding_response([[1.0, 2.0], [3.0, 4.0]])
        mock_client.aio.models.embed_content = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )

        result = await llm.aembed_documents(["test text", "test text2"])

        # Verify the call was made with correct parameters
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == MODEL_NAME
        assert call_kwargs["contents"] == ["test text", "test text2"]
        assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"

        # Verify the result
        assert result == [[1.0, 2.0], [3.0, 4.0]]
