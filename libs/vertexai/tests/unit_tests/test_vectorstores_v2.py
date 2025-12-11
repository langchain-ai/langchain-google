from unittest import mock

import pytest

from langchain_google_vertexai.vectorstores import _v2_operations
from langchain_google_vertexai.vectorstores._searcher import VectorSearchSearcher

upsert_datapoints_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.upsert_datapoints"
)
find_neighbors_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.find_neighbors"
)
remove_datapoints_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.remove_datapoints"
)
semantic_search_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.semantic_search"
)
text_search_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.text_search"
)
hybrid_search_mock = mock.patch(
    "langchain_google_vertexai.vectorstores._v2_operations.hybrid_search"
)


@pytest.fixture
def mock_searcher_v1():
    mock_index = mock.MagicMock()
    mock_index.resource_name = (
        "projects/test-project/locations/test-region/indexes/test-index"
    )
    mock_endpoint = mock.MagicMock()
    mock_endpoint.resource_name = (
        "projects/test-project/locations/test-region/indexEndpoints/test-endpoint"
    )
    mock_deployed = mock.MagicMock()
    mock_deployed.index = mock_index.resource_name
    mock_deployed.deployed_index_id = "deployed_index_123"
    mock_endpoint.deployed_indexes = [mock_deployed]
    return VectorSearchSearcher(
        endpoint=mock_endpoint,
        index=mock_index,
        stream_update=True,
        api_version="v1",
    )


@pytest.fixture
def mock_searcher_v2():
    mock_collection = mock.MagicMock()
    mock_collection.resource_name = (
        "projects/test-project/locations/test-region/collections/test-collection"
    )
    return VectorSearchSearcher(
        endpoint=None,
        index=None,
        collection=mock_collection,
        stream_update=True,
        api_version="v2",
        project_id="test-project",
        region="test-region",
        credentials=None,
    )


@mock.patch("langchain_google_vertexai.vectorstores._searcher.stream_update_index")
@mock.patch("langchain_google_vertexai.vectorstores._searcher.batch_update_index")
def test_add_to_index_v1(mock_batch_update, mock_stream_update, mock_searcher_v1):
    """Test that add_to_index calls the V1 operations."""
    mock_searcher_v1.add_to_index(ids=["1"], embeddings=[[0.1, 0.2]])
    mock_stream_update.assert_called_once()
    mock_batch_update.assert_not_called()


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@upsert_datapoints_mock
def test_add_to_index_v2(mock_upsert_datapoints, mock_searcher_v2):
    """Test that add_to_index calls the V2 operations."""
    ids = ["1"]
    embeddings = [[0.1, 0.2]]
    mock_searcher_v2.add_to_index(ids=ids, embeddings=embeddings)
    mock_upsert_datapoints.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection=mock_searcher_v2._collection.resource_name,
        ids=ids,
        embeddings=embeddings,
        metadatas=None,
        credentials=None,
        vector_field_name="embedding",
        sparse_embeddings=None,
    )


@mock.patch(
    "google.cloud.aiplatform.matching_engine.MatchingEngineIndexEndpoint.find_neighbors"
)
def test_find_neighbors_v1(mock_find_neighbors_class, mock_searcher_v1):
    """Test that find_neighbors calls the V1 operations and passes V1 filters."""
    embeddings = [[0.1, 0.2]]
    filter_: list = []
    numeric_filter = mock.MagicMock()
    mock_searcher_v1._endpoint.find_neighbors = mock_find_neighbors_class
    mock_searcher_v1.find_neighbors(
        embeddings=embeddings, filter_=filter_, numeric_filter=numeric_filter
    )
    mock_find_neighbors_class.assert_called_once_with(
        deployed_index_id=mock_searcher_v1._deployed_index_id,
        queries=embeddings,
        num_neighbors=4,
        filter=filter_,
        numeric_filter=numeric_filter,
    )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@find_neighbors_mock
def test_find_neighbors_v2_dense_search(mock_find_neighbors, mock_searcher_v2):
    """Test that V2 find_neighbors calls the V2 operations for dense search."""
    embeddings = [[0.1, 0.2]]
    mock_searcher_v2.find_neighbors(embeddings=embeddings)
    mock_find_neighbors.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection_id="test-collection",
        queries=embeddings,
        num_neighbors=4,
        filter_=None,
        credentials=None,
        vector_field_name="embedding",
        sparse_queries=None,
        rrf_ranking_alpha=1,
    )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@find_neighbors_mock
def test_find_neighbors_v2_hybrid_search(mock_find_neighbors, mock_searcher_v2):
    """Test that V2 find_neighbors calls the V2 operations for hybrid search."""
    embeddings = [[0.1, 0.2]]
    sparse_embeddings = [{"values": [0.5], "indices": [10]}]
    mock_searcher_v2.find_neighbors(
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
    )
    mock_find_neighbors.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection_id="test-collection",
        queries=embeddings,
        num_neighbors=4,
        filter_=None,
        credentials=None,
        vector_field_name="embedding",
        sparse_queries=sparse_embeddings,
        rrf_ranking_alpha=1,
    )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@remove_datapoints_mock
def test_remove_datapoints_v2(mock_remove_datapoints, mock_searcher_v2):
    """Test that remove_datapoints calls the V2 operations."""
    datapoint_ids = ["1"]
    mock_searcher_v2.remove_datapoints(datapoint_ids=datapoint_ids)
    mock_remove_datapoints.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection=mock_searcher_v2._collection.resource_name,
        datapoint_ids=datapoint_ids,
        credentials=None,
    )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@semantic_search_mock
def test_semantic_search_v2(mock_semantic_search, mock_searcher_v2):
    """Test that semantic_search calls the V2 operations."""
    search_text = "test query"
    search_field = "embedding"
    k = 5
    task_type = "RETRIEVAL_QUERY"
    filter_dict = {"category": {"$eq": "test"}}

    mock_searcher_v2.semantic_search(
        search_text=search_text,
        search_field=search_field,
        k=k,
        task_type=task_type,
        filter_=filter_dict,
    )

    mock_semantic_search.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection_id="test-collection",
        search_text=search_text,
        search_field=search_field,
        num_neighbors=k,
        task_type=task_type,
        filter_=filter_dict,
        credentials=None,
    )


def test_semantic_search_v1_raises_error(mock_searcher_v1):
    """Test that semantic_search raises NotImplementedError for V1."""
    with pytest.raises(NotImplementedError, match="only supported in v2"):
        mock_searcher_v1.semantic_search(
            search_text="test query",
            search_field="embedding",
            k=5,
        )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@text_search_mock
def test_text_search_v2(mock_text_search, mock_searcher_v2):
    """Test that text_search calls the V2 operations."""
    search_text = "test keyword"
    data_field_names = ["page_content", "title"]
    k = 10

    mock_searcher_v2.text_search(
        search_text=search_text,
        data_field_names=data_field_names,
        k=k,
    )

    mock_text_search.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection_id="test-collection",
        search_text=search_text,
        data_field_names=data_field_names,
        num_neighbors=k,
        credentials=None,
    )


def test_text_search_v1_raises_error(mock_searcher_v1):
    """Test that text_search raises NotImplementedError for V1."""
    with pytest.raises(NotImplementedError, match="only supported in v2"):
        mock_searcher_v1.text_search(
            search_text="test keyword",
            data_field_names=["page_content"],
            k=5,
        )


@pytest.mark.skipif(
    _v2_operations.vectorsearch_v1beta is None, reason="V2 SDK not installed"
)
@hybrid_search_mock
def test_hybrid_search_v2(mock_hybrid_search, mock_searcher_v2):
    """Test that hybrid_search calls the V2 operations."""
    search_text = "Men's outfit for beach"
    search_field = "embedding"
    data_field_names = ["page_content"]
    k = 10
    task_type = "RETRIEVAL_QUERY"
    filter_dict = {"category": {"$eq": "clothing"}}
    semantic_weight = 1.5
    text_weight = 1.0

    mock_searcher_v2.hybrid_search(
        search_text=search_text,
        search_field=search_field,
        data_field_names=data_field_names,
        k=k,
        task_type=task_type,
        filter_=filter_dict,
        semantic_weight=semantic_weight,
        text_weight=text_weight,
    )

    mock_hybrid_search.assert_called_once_with(
        project_id="test-project",
        region="test-region",
        collection_id="test-collection",
        search_text=search_text,
        search_field=search_field,
        data_field_names=data_field_names,
        num_neighbors=k,
        task_type=task_type,
        filter_=filter_dict,
        semantic_weight=semantic_weight,
        text_weight=text_weight,
        credentials=None,
    )


def test_hybrid_search_v1_raises_error(mock_searcher_v1):
    """Test that hybrid_search raises NotImplementedError for V1."""
    with pytest.raises(NotImplementedError, match="only supported in v2"):
        mock_searcher_v1.hybrid_search(
            search_text="test query",
            search_field="embedding",
            data_field_names=["page_content"],
            k=5,
        )
