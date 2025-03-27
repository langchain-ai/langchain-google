"""Test embeddings model integration."""

from unittest.mock import MagicMock, patch

from google.ai.generativelanguage_v1beta.types import (
    BatchEmbedContentsRequest,
    BatchEmbedContentsResponse,
    ContentEmbedding,
    EmbedContentRequest,
    EmbedContentResponse,
)
from pydantic import SecretStr
from pytest import CaptureFixture

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient"
    ) as mock_prediction_service:
        _ = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr("..."),
        )
        mock_prediction_service.assert_called_once()
        client_info = mock_prediction_service.call_args.kwargs["client_info"]
        assert "langchain-google-genai" in client_info.user_agent
        assert "GoogleGenerativeAIEmbeddings" in client_info.user_agent
        assert "GoogleGenerativeAIEmbeddings" in client_info.client_library_version

    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient"
    ) as mock_prediction_service:
        _ = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr("..."),
            task_type="retrieval_document",
        )
        mock_prediction_service.assert_called_once()


def test_api_key_is_string() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr("secret-api-key"),
    )
    assert isinstance(embeddings.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=SecretStr("secret-api-key"),
    )
    print(embeddings.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_embed_query() -> None:
    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient"
    ) as mock_prediction_service:
        mock_embed = MagicMock()
        mock_embed.return_value = EmbedContentResponse(
            embedding=ContentEmbedding(values=[1.0, 2])
        )
        mock_prediction_service.return_value.embed_content = mock_embed
        llm = GoogleGenerativeAIEmbeddings(
            model="models/embedding-test",
            google_api_key=SecretStr("test-key"),
            task_type="classification",
        )
        llm.embed_query("test text", output_dimensionality=524)
        request = EmbedContentRequest(
            model="models/embedding-test",
            content={"parts": [{"text": "test text"}]},
            task_type="CLASSIFICATION",
            output_dimensionality=524,
        )
        mock_embed.assert_called_once_with(request)


def test_embed_documents() -> None:
    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient"
    ) as mock_prediction_service:
        mock_embed = MagicMock()
        mock_embed.return_value = BatchEmbedContentsResponse(
            embeddings=[ContentEmbedding(values=[1.0, 2])]
        )
        mock_prediction_service.return_value.batch_embed_contents = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model="models/embedding-test",
            google_api_key=SecretStr("test-key"),
        )

        llm.embed_documents(["test text", "test text2"], titles=["title1", "title2"])
        request = BatchEmbedContentsRequest(
            model="models/embedding-test",
            requests=[
                EmbedContentRequest(
                    model="models/embedding-test",
                    content={"parts": [{"text": "test text"}]},
                    task_type="RETRIEVAL_DOCUMENT",
                    title="title1",
                ),
                EmbedContentRequest(
                    model="models/embedding-test",
                    content={"parts": [{"text": "test text2"}]},
                    task_type="RETRIEVAL_DOCUMENT",
                    title="title2",
                ),
            ],
        )
        mock_embed.assert_called_once_with(request)


def test_embed_documents_with_numerous_texts() -> None:
    TEST_CORPUS_SIZE = 100
    TEST_BATCH_SIZE = 20
    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient"
    ) as mock_prediction_service:
        mock_embed = MagicMock()
        mock_embed.return_value = BatchEmbedContentsResponse(
            embeddings=[ContentEmbedding(values=[1.0 for _ in range(TEST_BATCH_SIZE)])]
        )
        mock_prediction_service.return_value.batch_embed_contents = mock_embed

        llm = GoogleGenerativeAIEmbeddings(
            model="models/embedding-test",
            google_api_key=SecretStr("test-key"),
        )

        llm.embed_documents(
            ["test text" for _ in range(TEST_CORPUS_SIZE)],
            batch_size=TEST_BATCH_SIZE,
            titles=["title1" for _ in range(TEST_CORPUS_SIZE)],
        )
        request = BatchEmbedContentsRequest(
            model="models/embedding-test",
            requests=[
                EmbedContentRequest(
                    model="models/embedding-test",
                    content={"parts": [{"text": "test text"}]},
                    task_type="RETRIEVAL_DOCUMENT",
                    title="title1",
                )
                for _ in range(TEST_BATCH_SIZE)
            ],
        )
        mock_embed.assert_called_with(request)
        assert mock_embed.call_count == TEST_CORPUS_SIZE / TEST_BATCH_SIZE
