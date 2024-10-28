from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from google.auth import credentials as ga_credentials
from google.cloud.discoveryengine_v1beta import Document as DiscoveryEngineDocument
from google.cloud.discoveryengine_v1beta.types import SearchRequest, SearchResponse
from langchain_core.embeddings import FakeEmbeddings

from langchain_google_community.vertex_ai_search import VertexAISearchRetriever


def test_search_request_with_auto_populated_fields() -> None:
    """
    Test the creation of a search request with automatically populated fields.
    This test verifies that the VertexAISearchRetriever correctly creates a
    SearchRequest object with the expected auto-populated fields.
    """

    # Mock the SearchServiceClient to avoid real network calls
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"

        retriever = VertexAISearchRetriever(
            project_id="project_id_value",
            data_store_id="data_store_id_value",
            location_id="location_id_value",
            serving_config_id="serving_config_id_value",
            credentials=ga_credentials.AnonymousCredentials(),
            filter="filter_value",
            order_by="title desc",
            canonical_filter="true",
        )

        # Assert that serving_config_path was called with the correct arguments
        mock_client.serving_config_path.assert_called_once_with(
            project="project_id_value",
            location="location_id_value",
            data_store="data_store_id_value",
            serving_config="serving_config_id_value",
        )

        search_request = retriever._create_search_request(query="query_value")

        assert isinstance(search_request, SearchRequest)
        assert search_request.query == "query_value"
        assert search_request.filter == "filter_value"
        assert search_request.order_by == "title desc"
        assert search_request.canonical_filter == "true"
        assert search_request.serving_config == "serving_config_value"
        assert search_request.page_size == 5


@pytest.mark.parametrize(
    "engine_data_type, get_extractive_answers, config, expected_spec",
    [
        (0, True, None, {"max_extractive_answer_count": 1}),
        (
            0,
            False,
            {"num_previous_segments": 1, "num_next_segments": 1},
            {
                "max_extractive_segment_count": 1,
                "num_previous_segments": 1,
                "num_next_segments": 1,
                "return_extractive_segment_score": False,
            },
        ),
        (
            0,
            False,
            {"num_previous_segments": 2, "num_next_segments": 3},
            {
                "max_extractive_segment_count": 1,
                "num_previous_segments": 2,
                "num_next_segments": 3,
                "return_extractive_segment_score": False,
            },
        ),
        (
            0,
            False,
            {"num_previous_segments": 3, "num_next_segments": 2},
            {
                "max_extractive_segment_count": 1,
                "num_previous_segments": 3,
                "num_next_segments": 2,
                "return_extractive_segment_score": False,
            },
        ),
        (
            0,
            False,
            {
                "num_previous_segments": 3,
                "num_next_segments": 2,
                "return_extractive_segment_score": True,
            },
            {
                "max_extractive_segment_count": 1,
                "num_previous_segments": 3,
                "num_next_segments": 2,
                "return_extractive_segment_score": True,
            },
        ),
    ],
)
def test_get_content_spec_kwargs(
    engine_data_type: int,
    get_extractive_answers: bool,
    config: dict,
    expected_spec: dict,
) -> None:
    """
    Test the _get_content_spec_kwargs method of VertexAISearchRetriever.

    This test verifies that the _get_content_spec_kwargs method correctly generates
    the content specification based on various input parameters. It checks different
    combinations of engine_data_type, get_extractive_answers, and segment configurations
    to ensure the method produces the expected extractive content specification.

    Args:
        engine_data_type (int): The type of engine data (0 for unstructured data).
        get_extractive_answers (bool): Whether to get extractive answers.
        segment_config (dict): Configuration for extractive segments.
        expected_spec (dict): The expected specification for the given input.

    The test creates a VertexAISearchRetriever instance with the given parameters,
    calls _get_content_spec_kwargs, and asserts that the returned specification
    matches the expected values.
    """
    with patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient", autospec=True
    ) as mock_client:
        mock_client.return_value = MagicMock()

        retriever_params = {
            "project_id": "mock-project",
            "data_store_id": "mock-data-store",
            "location_id": "global",
            "engine_data_type": engine_data_type,
            "get_extractive_answers": get_extractive_answers,
            **(config or {}),
        }

        retriever = VertexAISearchRetriever(**retriever_params)
        content_spec_kwargs = retriever._get_content_spec_kwargs()

        assert content_spec_kwargs is not None
        assert "extractive_content_spec" in content_spec_kwargs
        extractive_content_spec = content_spec_kwargs["extractive_content_spec"]

        for key, value in expected_spec.items():
            assert hasattr(extractive_content_spec, key)
            assert getattr(extractive_content_spec, key) == value


@pytest.fixture
def mock_search_response() -> SearchResponse:
    """
    Fixture that creates a mock SearchResponse object for testing purposes.

    This fixture generates a SearchResponse with two SearchResult objects,
    each containing a DiscoveryEngineDocument with mock data. The mock data
    includes structured and derived structured data, simulating the response
    from a Vertex AI Search query.

    Returns:
        SearchResponse: A mock SearchResponse object with two SearchResult items.
    """
    return SearchResponse(
        results=[
            SearchResponse.SearchResult(
                id="mock-id-1",
                document=DiscoveryEngineDocument(
                    name="mock-name-1",
                    id="mock-id-1",
                    struct_data={"url": "mock-url-1", "title": "Mock Title 1"},
                    derived_struct_data={
                        "link": "mock-link-1",
                        "extractive_segments": [
                            {
                                "relevanceScore": 0.9,
                                "previous_segments": [
                                    {"content": "Mock previous segment 1"},
                                    {"content": "Mock previous segment 2"},
                                    {"content": "Mock previous segment 3"},
                                ],
                                "next_segments": [
                                    {"content": "Mock next segment 1"},
                                    {"content": "Mock next segment 2"},
                                    {"content": "Mock next segment 3"},
                                ],
                                "content": "Mock content 1",
                            }
                        ],
                        "extractive_answers": [{"content": "Mock extractive answer 1"}],
                    },
                ),
            ),
            SearchResponse.SearchResult(
                id="mock-id-2",
                document=DiscoveryEngineDocument(
                    name="mock-name-2",
                    id="mock-id-2",
                    struct_data={"url": "mock-url-2", "title": "Mock Title 2"},
                    derived_struct_data={
                        "link": "mock-link-2",
                        "extractive_segments": [
                            {"relevanceScore": 0.95, "content": "Mock content 2"}
                        ],
                        "extractive_answers": [{"content": "Mock extractive answer 2"}],
                    },
                ),
            ),
        ]
    )


def test_convert_unstructured_search_response_extractive_segments(
    mock_search_response: SearchResponse,
) -> None:
    """
    Test the _convert_unstructured_search_response method for extractive segments.
    """
    with patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient", autospec=True
    ) as mock_client:
        mock_client.return_value = MagicMock()
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            engine_data_type=0,
            get_extractive_answers=False,
            return_extractive_segment_score=True,
        )

        documents = retriever._convert_unstructured_search_response(
            mock_search_response.results, "extractive_segments"
        )

        assert len(documents) == 2

        # Check first document
        assert documents[0].page_content == "Mock content 1"
        assert documents[0].metadata["id"] == "mock-id-1"
        assert documents[0].metadata["source"] == "mock-link-1"
        assert documents[0].metadata["relevance_score"] == 0.9
        assert len(documents[0].metadata["previous_segments"]) == 3
        assert len(documents[0].metadata["next_segments"]) == 3

        # Check second document
        assert documents[1].page_content == "Mock content 2"
        assert documents[1].metadata["id"] == "mock-id-2"
        assert documents[1].metadata["source"] == "mock-link-2"
        assert documents[1].metadata["relevance_score"] == 0.95
        assert len(documents[1].metadata["previous_segments"]) == 0
        assert len(documents[1].metadata["next_segments"]) == 0


def test_convert_unstructured_search_response_extractive_answers(
    mock_search_response: SearchResponse,
) -> None:
    """
    Test the _convert_unstructured_search_response method for extractive answers.

    This test verifies that the _convert_unstructured_search_response method
    correctly converts a SearchResponse containing extractive answers into
    a list of Document objects. It checks the content and metadata of the
    resulting documents, ensuring that extractive answer-specific fields
    are present and that segment-specific fields are absent.

    Args:
        mock_search_response (SearchResponse): A fixture providing a mock
            SearchResponse object for testing.

    The test creates a VertexAISearchRetriever instance configured for
    extractive answers, calls _convert_unstructured_search_response with
    the mock response and "extractive_answers" as the chunk type, and then
    asserts that the returned documents have the expected content and metadata.
    """
    with patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient", autospec=True
    ) as mock_client:
        mock_client.return_value = MagicMock()
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            engine_data_type=0,
            get_extractive_answers=True,
        )

        documents = retriever._convert_unstructured_search_response(
            mock_search_response.results, "extractive_answers"
        )

        assert len(documents) == 2

        # Check first document
        assert documents[0].page_content == "Mock extractive answer 1"
        assert documents[0].metadata["id"] == "mock-id-1"
        assert documents[0].metadata["source"] == "mock-link-1"
        assert "relevance_score" not in documents[0].metadata
        assert "previous_segments" not in documents[0].metadata
        assert "next_segments" not in documents[0].metadata

        # Check second document
        assert documents[1].page_content == "Mock extractive answer 2"
        assert documents[1].metadata["id"] == "mock-id-2"
        assert documents[1].metadata["source"] == "mock-link-2"
        assert "relevance_score" not in documents[1].metadata
        assert "previous_segments" not in documents[1].metadata
        assert "next_segments" not in documents[1].metadata


def test_custom_embedding_with_valid_values() -> None:
    """
    Test with a valid custom embedding model and field path.
    """
    # Mock the SearchServiceClient to avoid real network calls
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"
        embeddings = FakeEmbeddings(size=100)

        retriever = VertexAISearchRetriever(
            project_id="project_id_value",
            data_store_id="data_store_id_value",
            location_id="location_id_value",
            serving_config_id="serving_config_id_value",
            credentials=ga_credentials.AnonymousCredentials(),
            filter="filter_value",
            order_by="title desc",
            canonical_filter="true",
            custom_embedding=embeddings,
            custom_embedding_field_path="embedding_field",
            custom_embedding_ratio=0.5,
        )

        # Assert that serving_config_path was called with the correct arguments
        mock_client.serving_config_path.assert_called_once_with(
            project="project_id_value",
            location="location_id_value",
            data_store="data_store_id_value",
            serving_config="serving_config_id_value",
        )

        search_request = retriever._create_search_request(query="query_value")
        assert search_request.embedding_spec is not None
        assert search_request.ranking_expression == (
            "0.5 * dotProduct(embedding_field) + 0.5 * relevance_score"
        )


def test_custom_embedding_with_invalid_ratio() -> None:
    """
    Test with an invalid custom embedding ratio.
    """
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"
        embeddings = FakeEmbeddings(size=100)
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            custom_embedding=embeddings,
            custom_embedding_field_path="embedding_field",
            custom_embedding_ratio=1.5,  # Invalid ratio
        )
        with pytest.raises(ValueError):
            retriever._create_search_request(query="query_value")


def test_custom_embedding_with_missing_field_path() -> None:
    """
    Test with a missing custom embedding field path.
    """
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"
        embeddings = FakeEmbeddings(size=100)
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            custom_embedding=embeddings,
            custom_embedding_ratio=0.5,  # Invalid ratio
        )
        with pytest.raises(ValueError):
            retriever._create_search_request(query="query_value")


def test_custom_embedding_with_missing_model() -> None:
    """
    Test with a missing custom embedding model.
    """
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            custom_embedding_field_path="embedding_field",
            custom_embedding_ratio=0.5,  # Invalid ratio
        )
        with pytest.raises(ValueError):
            retriever._create_search_request(query="query_value")
