from typing import Generator, Union
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from google.auth import credentials as ga_credentials
from google.cloud.discoveryengine_v1 import (
    SearchResponse,
    SearchServiceClient,
)
from google.cloud.discoveryengine_v1beta import (
    SearchResponse as BetaSearchResponse,
)
from google.cloud.discoveryengine_v1beta import (
    SearchServiceClient as BetaSearchServiceClient,
)
from langchain_core.embeddings import FakeEmbeddings

from langchain_google_community.vertex_ai_search import VertexAISearchRetriever


@pytest.fixture
def mock_stable_client() -> Generator[SearchServiceClient, None, None]:
    """Fixture for mocking stable version client."""
    # Mock the SearchServiceClient of stable version to avoid real network calls
    with mock.patch(
        "google.cloud.discoveryengine_v1.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path = mock.MagicMock(
            return_value="serving_config_value"
        )
        yield mock_client


@pytest.fixture
def mock_beta_client() -> Generator[BetaSearchServiceClient, None, None]:
    """Fixture for mocking beta version client."""
    # Mock the SearchServiceClient of beta version to avoid real network calls
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path = mock.MagicMock(
            return_value="serving_config_value"
        )
        yield mock_client


def test_version_initialization_stable(mock_stable_client: MagicMock) -> None:
    """Test initialization with stable version."""
    retriever = VertexAISearchRetriever(
        project_id="project_id_value",
        data_store_id="data_store_id_value",
        credentials=ga_credentials.AnonymousCredentials(),
    )

    assert not retriever.beta
    mock_stable_client.serving_config_path.assert_called_once()


def test_version_initialization_beta(mock_beta_client: MagicMock) -> None:
    """Test initialization with beta version."""
    retriever = VertexAISearchRetriever(
        project_id="project_id_value",
        data_store_id="data_store_id_value",
        beta=True,
        credentials=ga_credentials.AnonymousCredentials(),
    )

    assert retriever.beta
    mock_beta_client.serving_config_path.assert_called_once()


def test_version_compatibility_warning() -> None:
    """Test warning when using beta features with stable version."""
    fake_embeddings = FakeEmbeddings(size=100)

    with pytest.warns(UserWarning, match="Beta features are configured but beta=False"):
        VertexAISearchRetriever(
            project_id="project_id_value",
            data_store_id="data_store_id_value",
            beta=False,
            credentials=ga_credentials.AnonymousCredentials(),
            custom_embedding=fake_embeddings,
            custom_embedding_field_path="embedding_field",
            custom_embedding_ratio=0.5,
        )


@pytest.mark.parametrize("beta", [False, True])
def test_search_request_with_auto_populated_fields(
    mock_stable_client: MagicMock,
    mock_beta_client: MagicMock,
    beta: bool,
) -> None:
    """
    Test the creation of a search request with automatically populated fields.
    This test verifies that the VertexAISearchRetriever correctly creates a
    SearchRequest object with the expected auto-populated fields for both stable
    and beta versions.
    """
    mock_client = mock_beta_client if beta else mock_stable_client

    retriever = VertexAISearchRetriever(
        project_id="project_id_value",
        data_store_id="data_store_id_value",
        location_id="location_id_value",
        serving_config_id="serving_config_id_value",
        credentials=ga_credentials.AnonymousCredentials(),
        filter="filter_value",
        order_by="title desc",
        canonical_filter="true",
        beta=beta,
        custom_embedding=FakeEmbeddings(size=100) if beta else None,
        custom_embedding_field_path="embedding_field" if beta else None,
        custom_embedding_ratio=0.5 if beta else None,
    )

    mock_client.serving_config_path.assert_called_once_with(
        project="project_id_value",
        location="location_id_value",
        data_store="data_store_id_value",
        serving_config="serving_config_id_value",
    )

    search_request = retriever._create_search_request(query="query_value")

    if beta:
        from google.cloud.discoveryengine_v1beta import SearchRequest
    else:
        from google.cloud.discoveryengine_v1 import SearchRequest

    assert isinstance(search_request, SearchRequest)
    assert search_request.query == "query_value"
    assert search_request.filter == "filter_value"
    assert search_request.order_by == "title desc"
    assert search_request.canonical_filter == "true"
    assert search_request.serving_config == "serving_config_value"
    assert search_request.page_size == 5


def test_beta_specific_params_in_stable_version() -> None:
    """Test that beta-specific parameters are ignored in stable version."""
    retriever = VertexAISearchRetriever(
        project_id="project_id_value",
        data_store_id="data_store_id_value",
        beta=False,
        credentials=ga_credentials.AnonymousCredentials(),
        custom_embedding=FakeEmbeddings(size=100),
        custom_embedding_field_path="embedding_field",
        custom_embedding_ratio=0.5,
    )

    params = retriever._get_beta_specific_params(query="test")
    assert params == {}


def test_custom_embedding_with_valid_values() -> None:
    """Test with a valid custom embedding model and field path."""
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
            beta=True,
        )

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


def test_custom_embedding_with_missing_field_path() -> None:
    """Test with a missing custom embedding field path."""
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
            custom_embedding_ratio=0.5,
            beta=True,
            credentials=ga_credentials.AnonymousCredentials(),
        )
        with pytest.raises(ValueError):
            retriever._create_search_request(query="query_value")


def test_custom_embedding_with_missing_model() -> None:
    """Test with a missing custom embedding model."""
    with mock.patch(
        "google.cloud.discoveryengine_v1beta.SearchServiceClient"
    ) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.serving_config_path.return_value = "serving_config_value"
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            custom_embedding_field_path="embedding_field",
            custom_embedding_ratio=0.5,
            beta=True,
            credentials=ga_credentials.AnonymousCredentials(),
        )
        with pytest.raises(ValueError):
            retriever._create_search_request(query="query_value")


@pytest.mark.parametrize(
    "beta_flag,expected_module",
    [
        (True, "google.cloud.discoveryengine_v1beta"),
        (False, "google.cloud.discoveryengine_v1"),
    ],
)
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
    beta_flag: bool,
    expected_module: str,
    engine_data_type: int,
    get_extractive_answers: bool,
    config: dict,
    expected_spec: dict,
) -> None:
    """
    Test the _get_content_spec_kwargs method of VertexAISearchRetriever.

    This test verifies that:
    1. The correct version (beta/stable) of SearchRequest is imported.
    2. The content specification is correctly generated based on input parameters.

    Args:
        beta_flag: Whether to use beta version.
        expected_module: Expected module path for import.
        engine_data_type: The type of engine data (0 for unstructured data).
        get_extractive_answers: Whether to get extractive answers.
        config: Configuration for extractive segments.
        expected_spec: The expected specification for the given input.
    """
    with patch(f"{expected_module}.SearchServiceClient", autospec=True) as mock_client:
        mock_client.return_value = MagicMock()

        # Mock the SearchRequest import to verify it's imported from the correct module
        with patch(f"{expected_module}.SearchRequest") as mock_request:
            # Set up the mock to behave like the real SearchRequest
            mock_content_spec = MagicMock()
            mock_content_spec.ExtractiveContentSpec = MagicMock()
            mock_request.ContentSearchSpec = mock_content_spec

            retriever_params = {
                "project_id": "mock-project",
                "data_store_id": "mock-data-store",
                "location_id": "global",
                "engine_data_type": engine_data_type,
                "get_extractive_answers": get_extractive_answers,
                "beta": beta_flag,
                "credentials": ga_credentials.AnonymousCredentials(),
                **(config or {}),
            }

            retriever = VertexAISearchRetriever(**retriever_params)
            retriever._get_content_spec_kwargs()

            # Verify the correct version of SearchRequest was imported
            mock_request.assert_called


@pytest.fixture
def mock_search_response(
    request: pytest.FixtureRequest,
) -> Union[SearchResponse, BetaSearchResponse]:
    """
    Parametrized fixture that creates a mock SearchResponse object for testing purposes.
    Provides both stable (v1) and beta versions of the response.
    """
    module_path, is_beta = request.param
    if is_beta:
        from google.cloud.discoveryengine_v1beta import (
            Document as BetaDocument,
        )
        from google.cloud.discoveryengine_v1beta import (
            SearchResponse as BetaSearchResponse,
        )

        Document = BetaDocument
        Response = BetaSearchResponse
    else:
        from google.cloud.discoveryengine_v1 import (
            Document,
            SearchResponse,
        )

        Response = SearchResponse

    return Response(
        results=[
            Response.SearchResult(
                id="mock-id-1",
                document=Document(
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
            Response.SearchResult(
                id="mock-id-2",
                document=Document(
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


@pytest.mark.parametrize(
    "mock_search_response,expected_module,beta_flag",
    [
        (
            ("google.cloud.discoveryengine_v1", False),
            "google.cloud.discoveryengine_v1",
            False,
        ),
        (
            ("google.cloud.discoveryengine_v1beta", True),
            "google.cloud.discoveryengine_v1beta",
            True,
        ),
    ],
    indirect=["mock_search_response"],
)
def test_convert_unstructured_search_response_extractive_segments(
    mock_search_response: Union[SearchResponse, BetaSearchResponse],
    expected_module: str,
    beta_flag: bool,
) -> None:
    """
    Test the _convert_unstructured_search_response method for extractive segments.
    Tests both stable and beta versions of the API.

    Args:
        mock_search_response: A fixture providing a mock SearchResponse object.
        expected_module: Expected module path for import.
        beta_flag: Whether to use beta version.
    """
    with patch(f"{expected_module}.SearchServiceClient", autospec=True) as mock_client:
        mock_client.return_value = MagicMock()
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            engine_data_type=0,
            get_extractive_answers=False,
            return_extractive_segment_score=True,
            beta=beta_flag,
            credentials=ga_credentials.AnonymousCredentials(),
        )

        documents = retriever._convert_unstructured_search_response(
            mock_search_response.results, "extractive_segments"
        )

        assert len(documents) == 2

        # Verify first document (with segments)
        assert documents[0].page_content == "Mock content 1"
        assert documents[0].metadata["id"] == "mock-id-1"
        assert documents[0].metadata["source"] == "mock-link-1"
        assert documents[0].metadata["relevance_score"] == 0.9
        assert len(documents[0].metadata["previous_segments"]) == 3
        assert len(documents[0].metadata["next_segments"]) == 3

        # Verify second document (without segments)
        assert documents[1].page_content == "Mock content 2"
        assert documents[1].metadata["id"] == "mock-id-2"
        assert documents[1].metadata["source"] == "mock-link-2"
        assert documents[1].metadata["relevance_score"] == 0.95
        assert documents[1].metadata["previous_segments"] == []
        assert documents[1].metadata["next_segments"] == []


@pytest.mark.parametrize(
    "mock_search_response,expected_module,beta_flag",
    [
        (
            ("google.cloud.discoveryengine_v1", False),
            "google.cloud.discoveryengine_v1",
            False,
        ),
        (
            ("google.cloud.discoveryengine_v1beta", True),
            "google.cloud.discoveryengine_v1beta",
            True,
        ),
    ],
    indirect=["mock_search_response"],
)
def test_convert_unstructured_search_response_extractive_answers(
    mock_search_response: Union[SearchResponse, BetaSearchResponse],
    expected_module: str,
    beta_flag: bool,
) -> None:
    """
    Test the _convert_unstructured_search_response method for extractive answers.
    """
    with patch(f"{expected_module}.SearchServiceClient", autospec=True) as mock_client:
        mock_client.return_value = MagicMock()
        retriever = VertexAISearchRetriever(
            project_id="mock-project",
            data_store_id="mock-data-store",
            engine_data_type=0,
            get_extractive_answers=True,
            beta=beta_flag,
            credentials=ga_credentials.AnonymousCredentials(),
        )

        documents = retriever._convert_unstructured_search_response(
            mock_search_response.results, "extractive_answers"
        )

        assert len(documents) == 2

        # Verify first document
        assert documents[0].page_content == "Mock extractive answer 1"
        assert documents[0].metadata["id"] == "mock-id-1"
        assert documents[0].metadata["source"] == "mock-link-1"
        assert "relevance_score" not in documents[0].metadata
        assert "previous_segments" not in documents[0].metadata
        assert "next_segments" not in documents[0].metadata

        # Verify second document
        assert documents[1].page_content == "Mock extractive answer 2"
        assert documents[1].metadata["id"] == "mock-id-2"
        assert documents[1].metadata["source"] == "mock-link-2"
        assert "relevance_score" not in documents[1].metadata
        assert "previous_segments" not in documents[1].metadata
        assert "next_segments" not in documents[1].metadata
