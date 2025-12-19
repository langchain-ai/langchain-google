from typing import TYPE_CHECKING, Any, List, Optional

from google.oauth2.service_account import Credentials

if TYPE_CHECKING:
    from google.cloud import vectorsearch_v1beta
else:
    try:
        from google.cloud import vectorsearch_v1beta
    except ImportError:
        vectorsearch_v1beta = None  # type: ignore[assignment]


def _process_search_results(response) -> List[dict[str, Any]]:
    """Processes search response into standardized result format.

    Args:
        response: Search response from V2 API.

    Returns:
        List of result dictionaries with doc_id, score, and metadata.
    """
    results = []
    for result in response:
        data_obj = result.data_object
        result_dict: dict[str, Any] = {
            "doc_id": data_obj.name.split("/")[-1],
            "score": result.score if hasattr(result, "score") else 1.0,
        }

        # Include the metadata from the data object
        if hasattr(data_obj, "data") and data_obj.data:
            result_dict["metadata"] = dict(data_obj.data)

        results.append(result_dict)

    return results


def upsert_datapoints(
    project_id: str,
    region: str,
    collection: str,
    ids: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict] | None = None,
    credentials: Optional[Credentials] = None,
    vector_field_name: str = "embedding",
    sparse_embeddings: List[dict[str, List[int] | List[float]]] | None = None,
) -> None:
    """Upserts data points into a Vertex AI Vector Search 2.0 Collection.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection: The resource name of the collection.
        ids: List of datapoint IDs.
        embeddings: List of embedding vectors.
        metadatas: Optional list of metadata dictionaries.
        credentials: Optional credentials to use.
        vector_field_name: Name of the vector field in the collection schema.
        sparse_embeddings: Optional list of sparse embedding dictionaries.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_service_client"]

    # Prepare metadatas
    if metadatas is None:
        metadatas = [{}] * len(ids)

    # Prepare sparse embeddings
    if sparse_embeddings is None:
        sparse_embeddings = [None] * len(ids)  # type: ignore

    # Convert to v2 batch requests
    batch_requests = []
    for data_id, embedding, metadata, sparse_embedding in zip(
        ids, embeddings, metadatas, sparse_embeddings, strict=False
    ):
        # Build the vector object
        vector_obj = vectorsearch_v1beta.Vector(
            dense=vectorsearch_v1beta.DenseVector(values=embedding)
        )

        # Add sparse vector if provided
        if sparse_embedding is not None:
            vector_obj.sparse = vectorsearch_v1beta.SparseVector(
                indices=sparse_embedding["indices"],
                values=sparse_embedding["values"],
            )

        # Build data object
        data_object = vectorsearch_v1beta.DataObject(
            data=metadata,
            vectors={vector_field_name: vector_obj},
        )

        # Add as dictionary (not as CreateDataObjectRequest)
        batch_requests.append({"data_object_id": data_id, "data_object": data_object})

    # Batch create data objects
    batch_size = 100
    for i in range(0, len(batch_requests), batch_size):
        batch = batch_requests[i : i + batch_size]

        request = vectorsearch_v1beta.BatchCreateDataObjectsRequest(
            parent=collection,
            requests=batch,
        )

        client.batch_create_data_objects(request=request)


def find_neighbors(
    project_id: str,
    region: str,
    collection_id: str,
    queries: List[List[float]],
    num_neighbors: int,
    filter_: dict | None = None,
    credentials: Optional[Credentials] = None,
    vector_field_name: str = "embedding",
    sparse_queries: List[dict[str, List[int] | List[float]]] | None = None,
    rrf_ranking_alpha: float = 1.0,
) -> List[List[dict[str, Any]]]:
    """Searches for neighbors in a Vertex AI Vector Search 2.0 Collection.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        queries: List of query embeddings.
        num_neighbors: Number of neighbors to return.
        filter_: Optional filter dict.
            Examples: {"genre": {"$eq": "Drama"}},
                     {"$and": [{"year": {"$gte": 1990}}, {"genre": {"$eq": "Action"}}]}
        credentials: Optional credentials to use.
        vector_field_name: Name of the vector field in the collection schema.
        sparse_queries: Optional list of sparse query embeddings for hybrid search.
            Each sparse query should be: {"values": [...], "indices": [...]}
        rrf_ranking_alpha: RRF ranking alpha parameter for hybrid search (0.0 to 1.0).
            NOTE: This parameter is currently not used in V2 API.

    Returns:
        List of neighbor results for each query.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_search_service_client"]
    parent = f"projects/{project_id}/locations/{region}/collections/{collection_id}"

    # Prepare sparse queries
    if sparse_queries is None:
        sparse_queries = [None] * len(queries)  # type: ignore

    # Process each query
    all_results = []
    for query_embedding, sparse_query in zip(queries, sparse_queries, strict=False):
        # Build the vector for search
        query_vector = vectorsearch_v1beta.DenseVector(values=query_embedding)

        # Build VectorSearch parameters
        vector_search_params = {
            "search_field": vector_field_name,
            "vector": query_vector,
            "top_k": num_neighbors,
            "output_fields": vectorsearch_v1beta.OutputFields(
                data_fields=["*"],
                vector_fields=["*"],
                metadata_fields=["*"],
            ),
        }

        # Add filter if provided
        if filter_:
            vector_search_params["filter"] = filter_

        vector_search = vectorsearch_v1beta.VectorSearch(**vector_search_params)

        # Add sparse vector if provided for hybrid search
        if sparse_query is not None:
            vector_search.sparse_vector = vectorsearch_v1beta.SparseVector(
                indices=sparse_query["indices"],
                values=sparse_query["values"],
            )

        search_request = vectorsearch_v1beta.SearchDataObjectsRequest(
            parent=parent,
            vector_search=vector_search,
        )

        response = client.search_data_objects(request=search_request)

        # Process results
        query_results = []
        for result in response:
            data_obj = result.data_object
            result_dict: dict[str, Any] = {
                "doc_id": data_obj.name.split("/")[-1],
                "dense_score": result.score if hasattr(result, "score") else 1.0,
            }

            # Add sparse score if hybrid search
            if sparse_query is not None and hasattr(result, "sparse_score"):
                result_dict["sparse_score"] = result.sparse_score

            # Include the metadata from the data object
            if hasattr(data_obj, "data") and data_obj.data:
                result_dict["metadata"] = dict(data_obj.data)

            query_results.append(result_dict)

        all_results.append(query_results)

    return all_results


def remove_datapoints(
    project_id: str,
    region: str,
    collection: str,
    datapoint_ids: List[str],
    credentials: Optional[Credentials] = None,
) -> None:
    """Deletes data points from a Vertex AI Vector Search 2.0 Collection.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection: The resource name of the collection.
        datapoint_ids: List of datapoint IDs to delete.
        credentials: Optional credentials to use.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_service_client"]

    # Build delete requests
    requests = [
        vectorsearch_v1beta.DeleteDataObjectRequest(
            name=f"{collection}/dataObjects/{datapoint_id}"
        )
        for datapoint_id in datapoint_ids
    ]

    # Batch delete
    batch_delete_request = vectorsearch_v1beta.BatchDeleteDataObjectsRequest(
        parent=collection,
        requests=requests,
    )
    client.batch_delete_data_objects(request=batch_delete_request)


def get_datapoints_by_filter(
    project_id: str,
    region: str,
    collection_id: str,
    filter_: dict,
    credentials: Optional[Credentials] = None,
) -> List[str]:
    """Gets datapoint IDs that match a filter in a Vertex AI Vector Search 2.0.

    Retrieves IDs from the Collection matching the given filter.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        filter_: Filter dict to match datapoints.
            Examples: {"genre": {"$eq": "Drama"}},
                     {"$and": [{"year": {"$gte": 1990}}, {"genre": {"$eq": "Action"}}]}
        credentials: Optional credentials to use.

    Returns:
        List of datapoint IDs matching the filter.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_search_service_client"]
    parent = f"projects/{project_id}/locations/{region}/collections/{collection_id}"

    # Query datapoints with filter
    request = vectorsearch_v1beta.QueryDataObjectsRequest(
        parent=parent,
        filter=filter_,
        output_fields=vectorsearch_v1beta.OutputFields(
            metadata_fields=["*"],
        ),
    )

    # Query and collect all datapoint IDs
    datapoint_ids = []
    response = client.query_data_objects(request)
    for data_object in response:
        # Extract the ID from the resource name (last part after /)
        datapoint_id = data_object.name.split("/")[-1]
        datapoint_ids.append(datapoint_id)

    return datapoint_ids


def semantic_search(
    project_id: str,
    region: str,
    collection_id: str,
    search_text: str,
    search_field: str,
    num_neighbors: int,
    task_type: str = "RETRIEVAL_QUERY",
    filter_: dict | None = None,
    credentials: Optional[Credentials] = None,
) -> List[dict[str, Any]]:
    """Performs semantic search in a Vertex AI Vector Search 2.0 Collection.

    Semantic search automatically generates embeddings from the search text
    using Vertex AI models, so you don't need to manually create embeddings.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        search_text: Natural language query text.
        search_field: Name of the vector field to search (must have auto-embedding
            config).
        num_neighbors: Number of neighbors to return.
        task_type: Embedding task type. Options:
            - "RETRIEVAL_QUERY": For search queries (default)
            - "RETRIEVAL_DOCUMENT": For document indexing
            - "SEMANTIC_SIMILARITY": For semantic similarity
            - "CLASSIFICATION": For classification tasks
            - "CLUSTERING": For clustering tasks
        filter_: Optional filter dict.
            Examples: {"genre": {"$eq": "Drama"}},
                     {"$and": [{"year": {"$gte": 1990}}, {"genre": {"$eq": "Action"}}]}
        credentials: Optional credentials to use.

    Returns:
        List of search results with doc_id, score, and metadata.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_search_service_client"]
    parent = f"projects/{project_id}/locations/{region}/collections/{collection_id}"

    # Build SemanticSearch parameters
    semantic_search_params = {
        "search_text": search_text,
        "search_field": search_field,
        "task_type": task_type,
        "top_k": num_neighbors,
        "output_fields": vectorsearch_v1beta.OutputFields(
            data_fields=["*"],
            vector_fields=["*"],
            metadata_fields=["*"],
        ),
    }

    # Add filter if provided
    if filter_:
        semantic_search_params["filter"] = filter_

    semantic_search_obj = vectorsearch_v1beta.SemanticSearch(**semantic_search_params)

    search_request = vectorsearch_v1beta.SearchDataObjectsRequest(
        parent=parent,
        semantic_search=semantic_search_obj,
    )

    response = client.search_data_objects(request=search_request)

    return _process_search_results(response)


def text_search(
    project_id: str,
    region: str,
    collection_id: str,
    search_text: str,
    data_field_names: List[str],
    num_neighbors: int,
    credentials: Optional[Credentials] = None,
) -> List[dict[str, Any]]:
    """Performs text search in a Vertex AI Vector Search 2.0 Collection.

    Text search performs traditional keyword/full-text search on data fields
    without using embeddings.

    Note: Text search does not support filters. Use semantic_search or
    vector_search if you need filtering.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        search_text: Keyword search query text.
        data_field_names: List of data field names to search in (e.g., ["title",
            "description"]).
        num_neighbors: Number of neighbors to return.
        credentials: Optional credentials to use.

    Returns:
        List of search results with doc_id, score, and metadata.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_search_service_client"]
    parent = f"projects/{project_id}/locations/{region}/collections/{collection_id}"

    # Build TextSearch parameters
    text_search_params = {
        "search_text": search_text,
        "data_field_names": data_field_names,
        "top_k": num_neighbors,
        "output_fields": vectorsearch_v1beta.OutputFields(
            data_fields=["*"],
            vector_fields=["*"],
            metadata_fields=["*"],
        ),
    }

    text_search_obj = vectorsearch_v1beta.TextSearch(**text_search_params)

    search_request = vectorsearch_v1beta.SearchDataObjectsRequest(
        parent=parent,
        text_search=text_search_obj,
    )

    response = client.search_data_objects(request=search_request)

    return _process_search_results(response)


def hybrid_search(
    project_id: str,
    region: str,
    collection_id: str,
    search_text: str,
    search_field: str,
    data_field_names: List[str],
    num_neighbors: int,
    task_type: str = "RETRIEVAL_QUERY",
    filter_: dict | None = None,
    semantic_weight: float = 1.0,
    text_weight: float = 1.0,
    credentials: Optional[Credentials] = None,
) -> List[dict[str, Any]]:
    """Performs hybrid search combining semantic and text search with RRF.

    Hybrid search runs both semantic search (with auto-generated embeddings) and
    text search (keyword matching) in parallel, then combines results using
    Reciprocal Rank Fusion (RRF) algorithm for optimal ranking.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        search_text: Query text used for both semantic and text search.
        search_field: Name of the vector field to search (must have auto-embedding
            config).
        data_field_names: List of data field names to search in for text search.
        num_neighbors: Number of neighbors to return from each search before fusion.
        task_type: Embedding task type for semantic search.
            Options: "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", etc.
        filter_: Optional filter dict for semantic search only.
            Example: {"category": {"$eq": "Dresses"}}
        semantic_weight: Weight for semantic search results in RRF (0.0 to 1.0+).
        text_weight: Weight for text search results in RRF (0.0 to 1.0+).
        credentials: Optional credentials to use.

    Returns:
        List of search results ranked by RRF with doc_id, score, and metadata.
    """
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    clients = sdk_manager.get_v2_client()
    client = clients["data_object_search_service_client"]
    parent = f"projects/{project_id}/locations/{region}/collections/{collection_id}"

    # Build semantic search
    semantic_search_params = {
        "search_text": search_text,
        "search_field": search_field,
        "task_type": task_type,
        "top_k": num_neighbors,
        "output_fields": vectorsearch_v1beta.OutputFields(
            data_fields=["*"],
            vector_fields=["*"],
            metadata_fields=["*"],
        ),
    }

    # Add filter if provided
    if filter_:
        semantic_search_params["filter"] = filter_

    semantic_search_obj = vectorsearch_v1beta.SemanticSearch(**semantic_search_params)

    # Build text search
    text_search_params = {
        "search_text": search_text,
        "data_field_names": data_field_names,
        "top_k": num_neighbors,
        "output_fields": vectorsearch_v1beta.OutputFields(
            data_fields=["*"],
            vector_fields=["*"],
            metadata_fields=["*"],
        ),
    }

    text_search_obj = vectorsearch_v1beta.TextSearch(**text_search_params)

    # Create batch search request with RRF combining
    batch_search_request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(
        parent=parent,
        searches=[
            vectorsearch_v1beta.Search(semantic_search=semantic_search_obj),
            vectorsearch_v1beta.Search(text_search=text_search_obj),
        ],
        combine=vectorsearch_v1beta.BatchSearchDataObjectsRequest.CombineResultsOptions(
            ranker=vectorsearch_v1beta.Ranker(
                rrf=vectorsearch_v1beta.ReciprocalRankFusion(
                    weights=[semantic_weight, text_weight]
                )
            )
        ),
    )

    batch_results = client.batch_search_data_objects(batch_search_request)

    # When a ranker is used, batch_results.results contains a single ranked list
    # results[0] is the SearchDataObjectsResponse with the combined RRF-ranked results
    if batch_results.results:
        combined_results = batch_results.results[0]
        return _process_search_results(combined_results.results)
    else:
        return []
