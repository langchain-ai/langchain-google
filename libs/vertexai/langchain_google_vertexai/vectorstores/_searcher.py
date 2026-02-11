from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Tuple

from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from google.cloud.aiplatform import telemetry
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    HybridQuery,
    MatchNeighbor,
    Namespace,
    NumericNamespace,
)

from langchain_google_vertexai._utils import get_user_agent
from langchain_google_vertexai.vectorstores import _v2_operations
from langchain_google_vertexai.vectorstores._utils import (
    batch_update_index,
    stream_update_index,
    to_data_points,
)

MAX_DATA_POINTS = 10000


class Searcher(ABC):
    """Abstract implementation of a similarity searcher."""

    @abstractmethod
    def find_neighbors(
        self,
        embeddings: list[list[float]],
        k: int = 4,
        filter_: list[Namespace] | dict | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        *,
        sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
        rrf_ranking_alpha: float = 1,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Finds the `k` closes neighbors of each instance of embeddings.

        Args:
            embeddings: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: For v1: list of `Namespace` objects. For v2: dict.
            numeric_filter: List of `NumericNamespace` objects for filtering (v1 only).
            sparse_embeddings: List of Sparse embedding dictionaries which represents an
                embedding as a list of indices and as a list of sparse values:
                ie. `[{"values": [0.7, 0.5], "indices": [10, 20]}]`
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between `0` and
                `1.0`
                Weights Dense Search VS Sparse Search, as an example:
                - `rrf_ranking_alpha=1`: Only Dense
                - `rrf_ranking_alpha=0`: Only Sparse
                - `rrf_ranking_alpha=0.7`: `0.7` weighting for dense and `0.3` for
                    sparse

        Returns:
            List of records:
                ```python
                [
                    {
                        "doc_id": doc_id,
                        "dense_score": dense_score,
                        "sparse_score": sparse_score,
                    }
                ]
                ```
        """
        raise NotImplementedError

    @abstractmethod
    def add_to_index(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        is_complete_overwrite: bool = False,
        *,
        sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index.

        Args:
            ids: List of unique ids.
            embeddings: List of embeddings for each record.
            metadatas: List of metadata of each record.
            is_complete_overwrite: Whether to overwrite the entire index.
            sparse_embeddings: List of sparse embeddings for each record.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_datapoints(
        self,
        datapoint_ids: list[str],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_datapoints_by_filter(
        self,
        metadata: dict,
        max_datapoints: int = MAX_DATA_POINTS,
        **kwargs: Any,
    ) -> list[str]:
        """Gets datapoint IDs that match the given metadata filter.

        Args:
            metadata: Dictionary of metadata key-value pairs to filter by.
            max_datapoints: Maximum number of datapoints to return. Note: This
                parameter is ignored in v2 as the API returns all matching results.

        Returns:
            List of datapoint IDs matching the filter.
        """
        raise NotImplementedError

    @abstractmethod
    def semantic_search(
        self,
        search_text: str,
        search_field: str,
        k: int = 4,
        task_type: str = "RETRIEVAL_QUERY",
        filter_: dict | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs semantic search using auto-generated embeddings.

        Args:
            search_text: Natural language query text.
            search_field: Name of the vector field to search (must have auto-embedding
                config).
            k: Number of neighbors to return.
            task_type: Embedding task type (e.g., "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT").
            filter_: Filter dict (v2 only).

        Returns:
            List of records with doc_id, score, and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def text_search(
        self,
        search_text: str,
        data_field_names: list[str],
        k: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs keyword/full-text search on data fields.

        Note: Text search does not support filters. Use semantic_search or
        vector_search if you need filtering.

        Args:
            search_text: Keyword search query text.
            data_field_names: List of data field names to search in.
            k: Number of neighbors to return.

        Returns:
            List of records with doc_id, score, and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        search_text: str,
        search_field: str,
        data_field_names: list[str],
        k: int = 4,
        task_type: str = "RETRIEVAL_QUERY",
        filter_: dict | None = None,
        semantic_weight: float = 1.0,
        text_weight: float = 1.0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs hybrid search combining semantic and text search with RRF.

        Hybrid search runs both semantic search (with auto-generated embeddings) and
        text search (keyword matching) in parallel, then combines results using
        Reciprocal Rank Fusion (RRF) algorithm for optimal ranking.

        Args:
            search_text: Query text used for both semantic and text search.
            search_field: Name of the vector field to search (must have auto-embedding
                config).
            data_field_names: List of data field names to search in for text search.
            k: Number of neighbors to return from each search before fusion.
            task_type: Embedding task type for semantic search.
            filter_: Optional filter dict for semantic search only (v2 only).
            semantic_weight: Weight for semantic search results in RRF.
            text_weight: Weight for text search results in RRF.

        Returns:
            List of records with doc_id, score, and metadata ranked by RRF.
        """
        raise NotImplementedError


class VectorSearchSearcher(Searcher):
    """Class to interface with Vector Search indexes (v1) and collections (v2).

    Args:
        endpoint: The index endpoint (v1 only, None for v2).
        index: The index object (v1 only, None for v2).
        collection: The collection object (v2 only, None for v1).
        staging_bucket: GCS bucket for staging data (v1 only).
        stream_update: Whether to use streaming updates. (v1 only).
        api_version: Version of the Vector Search API ("v1" or "v2").
        project_id: GCP project ID (v2 only).
        region: GCP region (v2 only).
        credentials: GCP credentials (v2 only).
        vector_field_name: Name of the vector field in the schema (v2 only).
    """

    def __init__(
        self,
        endpoint: MatchingEngineIndexEndpoint | None,
        index: MatchingEngineIndex | None = None,
        staging_bucket: storage.Bucket | None = None,
        stream_update: bool = False,
        *,
        collection: SimpleNamespace | None = None,
        api_version: str = "v1",
        project_id: str | None = None,
        region: str | None = None,
        credentials: Any = None,
        vector_field_name: str = "embedding",
    ):
        self._api_version = api_version
        self._stream_update = stream_update
        self._staging_bucket = staging_bucket

        if self._api_version == "v1":
            if index is None:
                raise ValueError("`index` is required for V1.")
            self._index = index
            self._endpoint = endpoint
            self._deployed_index_id = self._get_deployed_index_id()
        elif self._api_version == "v2":
            if collection is None:
                raise ValueError("collection is required for v2")
            # Store collection in _index for compatibility
            self._index = collection  # type: ignore[assignment]
            self._collection = collection
            # Store v2-specific parameters
            self._project_id = project_id
            self._region = region
            self._credentials = credentials
            self._vector_field_name = vector_field_name
            # Parse collection_id from resource name if not provided
            if hasattr(self._collection, "resource_name") and not self._project_id:
                project_id, region, collection_id = self._parse_v2_resource_name(
                    self._collection.resource_name
                )
                self._project_id = project_id
                self._region = region
                self._collection_id = collection_id
            elif hasattr(self._collection, "resource_name"):
                _, _, collection_id = self._parse_v2_resource_name(
                    self._collection.resource_name
                )
                self._collection_id = collection_id
        else:
            msg = f"Unsupported API version: {api_version}"
            raise ValueError(msg)

    def remove_datapoints(
        self,
        datapoint_ids: list[str],
        **kwargs: Any,
    ) -> None:
        if self._api_version == "v1":
            if not self._index:
                msg = "`index` is required for V1."
                raise ValueError(msg)
            self._index.remove_datapoints(datapoint_ids=datapoint_ids)
        elif self._api_version == "v2":
            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            _v2_operations.remove_datapoints(
                project_id=self._project_id,
                region=self._region,
                collection=self._collection.resource_name,
                datapoint_ids=datapoint_ids,
                credentials=self._credentials,
            )

    def add_to_index(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        is_complete_overwrite: bool = False,
        *,
        sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index."""
        if self._api_version == "v1":
            # v1 needs data points with restricts
            data_points = to_data_points(
                ids=ids,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
                metadatas=metadatas,
            )
            if not self._index:
                msg = "`index` is required for V1."
                raise ValueError(msg)
            if self._stream_update:
                stream_update_index(index=self._index, data_points=data_points)
            else:
                if self._staging_bucket is None:
                    msg = (
                        "A staging bucket must be defined to update a "
                        "Vector Search index."
                    )
                    raise ValueError(msg)
                batch_update_index(
                    index=self._index,
                    data_points=data_points,
                    staging_bucket=self._staging_bucket,
                    is_complete_overwrite=is_complete_overwrite,
                )
        elif self._api_version == "v2":
            # v2 uses raw ids, embeddings, and metadatas
            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            _v2_operations.upsert_datapoints(
                project_id=self._project_id,
                region=self._region,
                collection=self._collection.resource_name,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                credentials=self._credentials,
                vector_field_name=self._vector_field_name,
                sparse_embeddings=sparse_embeddings,
            )

    def find_neighbors(
        self,
        embeddings: list[list[float]],
        k: int = 4,
        filter_: list[Namespace] | dict | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        *,
        sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
        rrf_ranking_alpha: float = 1,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Finds the `k` closes neighbors of each instance of embeddings."""
        if self._api_version == "v1":
            # v1 implementation
            _, user_agent = get_user_agent("vertex-ai-matching-engine")
            with telemetry.tool_context_manager(user_agent):
                if sparse_embeddings is None:
                    queries = embeddings
                else:
                    if len(sparse_embeddings) != len(embeddings):
                        msg = (
                            "The number of `sparse_embeddings` should match "
                            "the number of `embeddings` "
                            f"{len(sparse_embeddings)} != {len(embeddings)}"
                        )
                        raise ValueError(msg)
                    queries = []

                    for embedding, sparse_embedding in zip(
                        embeddings, sparse_embeddings, strict=False
                    ):
                        hybrid_query = HybridQuery(
                            sparse_embedding_dimensions=sparse_embedding["dimensions"],  # type: ignore
                            sparse_embedding_values=sparse_embedding["values"],  # type: ignore
                            dense_embedding=embedding,
                            rrf_ranking_alpha=rrf_ranking_alpha,
                        )
                        queries.append(hybrid_query)  # type: ignore

                # v1 only accepts list of Namespace for filters
                if isinstance(filter_, dict):
                    msg = (
                        "Dict filters are not supported in v1. "
                        "Use list[Namespace] instead."
                    )
                    raise ValueError(msg)

                if self._endpoint is None:
                    msg = "`endpoint` is required for V1 operations."
                    raise ValueError(msg)
                response = self._endpoint.find_neighbors(
                    deployed_index_id=self._deployed_index_id,
                    queries=queries,
                    num_neighbors=k,
                    filter=filter_,
                    numeric_filter=numeric_filter,
                    **kwargs,
                )

            return self._postprocess_response(response)
        elif self._api_version == "v2":
            # v2 implementation - accepts dict filters
            if filter_ is not None and not isinstance(filter_, dict):
                msg = "v2 requires dict filters. Example: {'genre': {'$eq': 'Drama'}}"
                raise ValueError(msg)

            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            return _v2_operations.find_neighbors(
                project_id=self._project_id,
                region=self._region,
                collection_id=self._collection_id,
                queries=embeddings,
                num_neighbors=k,
                filter_=filter_,
                credentials=self._credentials,
                vector_field_name=self._vector_field_name,
                sparse_queries=sparse_embeddings,
                rrf_ranking_alpha=rrf_ranking_alpha,
            )
        else:
            msg = f"Unsupported API version: {self._api_version}"
            raise ValueError(msg)

    def _postprocess_response(
        self, response: list[list[MatchNeighbor]]
    ) -> list[list[dict[str, Any]]]:
        """Postprocesses a v1 endpoint response.

        Args:
            response: Endpoint response.

        Returns:
            List of neighbor records.
        """
        queries_results = []
        for matching_neighbor_list in response:
            query_results = []
            for neighbor in matching_neighbor_list:
                dense_score = neighbor.distance if neighbor.distance else 0.0
                sparse_score = (
                    neighbor.sparse_distance if neighbor.sparse_distance else 0.0
                )
                result = {
                    "doc_id": neighbor.id,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                }
                query_results.append(result)
            queries_results.append(query_results)
        return queries_results

    def _get_deployed_index_id(self) -> str:
        """Gets the deployed index ID from the endpoint."""
        if not self._endpoint:
            msg = "Endpoint is required to get deployed index ID."
            raise ValueError(msg)

        for index in self._endpoint.deployed_indexes:
            if index.index == self._index.resource_name:
                return index.id

        msg = (
            f"Index {self._index.resource_name} is not deployed to "
            f"endpoint {self._endpoint.resource_name}"
        )
        raise ValueError(msg)

    def _parse_v2_resource_name(self, resource_name: str) -> Tuple[str, str, str]:
        """Extracts project, location, and collection ID from a v2 resource name.

        Args:
            resource_name: The resource name in the format
                `projects/{project}/locations/{location}/collections/{collection}`

        Returns:
            Tuple of (project_id, region, collection_id)
        """
        parts = resource_name.split("/")
        if (
            len(parts) != 6
            or parts[0] != "projects"
            or parts[2] != "locations"
            or parts[4] != "collections"
        ):
            msg = f"Invalid v2 resource name: {resource_name}"
            raise ValueError(msg)
        return parts[1], parts[3], parts[5]

    def _metadata_to_filter_dict(self, metadata: dict) -> dict:
        """Converts a metadata dictionary to a v2 filter dict.

        Args:
            metadata: Dictionary of metadata key-value pairs.

        Returns:
            Filter dict.
        """
        if not metadata:
            return {}

        # Build an $and query with $eq conditions for each metadata field
        conditions = []
        for key, value in metadata.items():
            conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_datapoints_by_filter(
        self,
        metadata: dict,
        max_datapoints: int = MAX_DATA_POINTS,
        **kwargs: Any,
    ) -> list[str]:
        """Gets datapoint IDs that match the given metadata filter.

        Args:
            metadata: Dictionary of metadata key-value pairs to filter by.
            max_datapoints: Maximum number of datapoints to return. Note: This
                parameter is ignored in v2 as the API returns all matching results.

        Returns:
            List of datapoint IDs matching the filter.
        """
        if self._api_version == "v1":
            msg = "Filtering by metadata for deletion is not supported in v1."
            raise NotImplementedError(msg)
        elif self._api_version == "v2":
            # Convert metadata to filter dict
            filter_ = self._metadata_to_filter_dict(metadata)

            if not filter_:
                return []

            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            # Note: max_datapoints is ignored for v2 as the API returns all results
            results = _v2_operations.get_datapoints_by_filter(
                project_id=self._project_id,
                region=self._region,
                collection_id=self._collection_id,
                filter_=filter_,
                credentials=self._credentials,
            )

            return results
        else:
            msg = f"Unsupported API version: {self._api_version}"
            raise ValueError(msg)

    def semantic_search(
        self,
        search_text: str,
        search_field: str,
        k: int = 4,
        task_type: str = "RETRIEVAL_QUERY",
        filter_: dict | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs semantic search using auto-generated embeddings.

        Args:
            search_text: Natural language query text.
            search_field: Name of the vector field to search (must have auto-embedding
                config).
            k: Number of neighbors to return.
            task_type: Embedding task type (e.g., "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT").
            filter_: Filter dict (v2 only).

        Returns:
            List of records with doc_id, score, and metadata.
        """
        if self._api_version == "v1":
            msg = "Semantic search is only supported in v2."
            raise NotImplementedError(msg)
        elif self._api_version == "v2":
            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            return _v2_operations.semantic_search(
                project_id=self._project_id,
                region=self._region,
                collection_id=self._collection_id,
                search_text=search_text,
                search_field=search_field,
                num_neighbors=k,
                task_type=task_type,
                filter_=filter_,
                credentials=self._credentials,
            )
        else:
            msg = f"Unsupported API version: {self._api_version}"
            raise ValueError(msg)

    def text_search(
        self,
        search_text: str,
        data_field_names: list[str],
        k: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs keyword/full-text search on data fields.

        Note: Text search does not support filters. Use semantic_search or
        vector_search if you need filtering.

        Args:
            search_text: Keyword search query text.
            data_field_names: List of data field names to search in.
            k: Number of neighbors to return.

        Returns:
            List of records with doc_id, score, and metadata.
        """
        if self._api_version == "v1":
            msg = "Text search is only supported in v2."
            raise NotImplementedError(msg)
        elif self._api_version == "v2":
            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            return _v2_operations.text_search(
                project_id=self._project_id,
                region=self._region,
                collection_id=self._collection_id,
                search_text=search_text,
                data_field_names=data_field_names,
                num_neighbors=k,
                credentials=self._credentials,
            )
        else:
            msg = f"Unsupported API version: {self._api_version}"
            raise ValueError(msg)

    def hybrid_search(
        self,
        search_text: str,
        search_field: str,
        data_field_names: list[str],
        k: int = 4,
        task_type: str = "RETRIEVAL_QUERY",
        filter_: dict | None = None,
        semantic_weight: float = 1.0,
        text_weight: float = 1.0,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Performs hybrid search combining semantic and text search with RRF.

        Hybrid search runs both semantic search (with auto-generated embeddings) and
        text search (keyword matching) in parallel, then combines results using
        Reciprocal Rank Fusion (RRF) algorithm for optimal ranking.

        Args:
            search_text: Query text used for both semantic and text search.
            search_field: Name of the vector field to search (must have auto-embedding
                config).
            data_field_names: List of data field names to search in for text search.
            k: Number of neighbors to return from each search before fusion.
            task_type: Embedding task type for semantic search.
            filter_: Optional filter dict for semantic search only (v2 only).
            semantic_weight: Weight for semantic search results in RRF.
            text_weight: Weight for text search results in RRF.

        Returns:
            List of records with doc_id, score, and metadata ranked by RRF.
        """
        if self._api_version == "v1":
            msg = "Hybrid search is only supported in v2."
            raise NotImplementedError(msg)
        elif self._api_version == "v2":
            if self._project_id is None or self._region is None:
                msg = "`project_id` and `region` are required for V2 operations."
                raise ValueError(msg)
            return _v2_operations.hybrid_search(
                project_id=self._project_id,
                region=self._region,
                collection_id=self._collection_id,
                search_text=search_text,
                search_field=search_field,
                data_field_names=data_field_names,
                num_neighbors=k,
                task_type=task_type,
                filter_=filter_,
                semantic_weight=semantic_weight,
                text_weight=text_weight,
                credentials=self._credentials,
            )
        else:
            msg = f"Unsupported API version: {self._api_version}"
            raise ValueError(msg)
