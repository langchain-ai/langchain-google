

from typing import Any, List, Optional

from google.oauth2.service_account import Credentials

try:
    from google.cloud import vectorsearch_v1beta
except ImportError:
    vectorsearch_v1beta = None


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
                indices=sparse_embedding["indices"],  # type: ignore
                values=sparse_embedding["values"],  # type: ignore
            )

        # Build data object
        data_object = vectorsearch_v1beta.DataObject(
            data=metadata,
            vectors={vector_field_name: vector_obj},
        )

        # Add as dictionary (not as CreateDataObjectRequest)
        batch_requests.append({
            "data_object_id": data_id,
            "data_object": data_object
        })

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
    filter_dict: dict | None = None,
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
        filter_dict: Optional filter dict.
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
        if filter_dict:
            vector_search_params["filter"] = filter_dict

        vector_search = vectorsearch_v1beta.VectorSearch(**vector_search_params)

        # Add sparse vector if provided for hybrid search
        if sparse_query is not None:
            vector_search.sparse_vector = vectorsearch_v1beta.SparseVector(
                indices=sparse_query["indices"],  # type: ignore
                values=sparse_query["values"],  # type: ignore
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
    filter_dict: dict,
    credentials: Optional[Credentials] = None,
) -> List[str]:
    """Gets datapoint IDs that match a filter in a Vertex AI Vector Search 2.0 Collection.

    Args:
        project_id: The GCP project ID.
        region: The GCP region.
        collection_id: The collection ID.
        filter_dict: Filter dict to match datapoints.
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
        filter=filter_dict,
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