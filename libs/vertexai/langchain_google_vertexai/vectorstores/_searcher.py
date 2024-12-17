from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

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
        embeddings: List[List[float]],
        k: int = 4,
        filter_: Union[List[Namespace], None] = None,
        numeric_filter: Union[List[NumericNamespace], None] = None,
        *,
        sparse_embeddings: Optional[
            List[Dict[str, Union[List[int], List[float]]]]
        ] = None,
        rrf_ranking_alpha: float = 1,
    ) -> List[List[Dict[str, Any]]]:
        """Finds the k closes neighbors of each instance of embeddings.
        Args:
            embedding: List of embeddings vectors.
            sparse_embeddings: List of Sparse embedding dictionaries which represents an
                embedding as a list of dimensions and as a list of sparse values:
                    ie. [{"values": [0.7, 0.5], "dimensions": [10, 20]}]
            k: Number of neighbors to be retrieved.
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between 0 and 1.0
                Weights Dense Search VS Sparse Search, as an example:
                - rrf_ranking_alpha=1: Only Dense
                - rrf_ranking_alpha=0: Only Sparse
                - rrf_ranking_alpha=0.7: 0.7 weighting for dense and 0.3 for sparse
            filter_: List of filters to apply.
        Returns:
            List of records: [
                {
                    "doc_id": doc_id,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score
                }
            ]
        """
        raise NotImplementedError()

    @abstractmethod
    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Union[List[dict], None] = None,
        is_complete_overwrite: bool = False,
        *,
        sparse_embeddings: Optional[
            List[Dict[str, Union[List[int], List[float]]]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index.

        Args:
            ids: List of unique ids.
            embeddings: List of embedddings for each record.
            sparse_embeddings: List of sparse embedddings for each record.
            metadatas: List of metadata of each record.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_datapoints(
        self,
        datapoint_ids: List[str],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_datapoints_by_filter(
        self,
        metadata: dict,
        max_datapoints: int = MAX_DATA_POINTS,
    ) -> List[str]:
        raise NotImplementedError()

    def _postprocess_response(
        self, response: List[List[MatchNeighbor]]
    ) -> List[List[Dict[str, Any]]]:
        """Posproceses an endpoint response and converts it to a list of list of records
        instead of using vertexai objects.
        Args:
            response: Endpoint response.
        Returns:
            List of records: [
                {
                    "doc_id": doc_id,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score
                }
            ]
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


class VectorSearchSearcher(Searcher):
    """Class to interface with a VectorSearch index and endpoint."""

    def __init__(
        self,
        endpoint: MatchingEngineIndexEndpoint,
        index: MatchingEngineIndex,
        staging_bucket: Union[storage.Bucket, None] = None,
        stream_update: bool = False,
    ) -> None:
        """Constructor.
        Args:
            endpoint: Endpoint that will be used to make find_neighbors requests.
            index: Underlying index deployed in that endpoint.
            staging_bucket: Necessary only if updating the index. Bucket where the
                embeddings and metadata will be staged.
        Raises:
            ValueError: If the index provided is not deployed in the endpoint.
        """
        super().__init__()
        self._endpoint = endpoint
        self._index = index
        self._deployed_index_id = self._get_deployed_index_id()
        self._staging_bucket = staging_bucket
        self._stream_update = stream_update

    def get_datapoints_by_filter(
        self,
        metadata: dict,
        max_datapoints: int = MAX_DATA_POINTS,
    ) -> List[str]:
        """Gets all the datapoints matching the metadata filters (text only)
        on the specified deployed index.
        """
        index_config = self._index.to_dict()["metadata"]["config"]
        embeddings = [[0.0] * int(index_config.get("dimensions", 1))]
        filter_ = [
            Namespace(name=key, allow_tokens=[value]) for key, value in metadata.items()
        ]
        neighbors = self.find_neighbors(
            embeddings=embeddings, k=max_datapoints, filter_=filter_
        )
        return [elem["doc_id"] for elem in neighbors[0]] if neighbors else []

    def remove_datapoints(
        self,
        datapoint_ids: List[str],
        **kwargs: Any,
    ) -> None:
        self._index.remove_datapoints(datapoint_ids=datapoint_ids)

    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Union[List[dict], None] = None,
        is_complete_overwrite: bool = False,
        *,
        sparse_embeddings: Optional[
            List[Dict[str, Union[List[int], List[float]]]]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index.

        Args:
            ids: List of unique ids.
            embeddings: List of embedddings for each record.
            sparse_embeddings: List of sparse embedddings for each record.
            metadatas: List of metadata of each record.
            is_complete_overwrite: Whether to overwrite everything.
        """

        data_points = to_data_points(
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            metadatas=metadatas,
        )

        if self._stream_update:
            stream_update_index(index=self._index, data_points=data_points)
        else:
            if self._staging_bucket is None:
                raise ValueError(
                    "In order to update a Vector Search index a staging bucket must"
                    " be defined."
                )
            batch_update_index(
                index=self._index,
                data_points=data_points,
                staging_bucket=self._staging_bucket,
                is_complete_overwrite=is_complete_overwrite,
            )

    def find_neighbors(
        self,
        embeddings: List[List[float]],
        k: int = 4,
        filter_: Union[List[Namespace], None] = None,
        numeric_filter: Union[List[NumericNamespace], None] = None,
        *,
        sparse_embeddings: Optional[
            List[Dict[str, Union[List[int], List[float]]]]
        ] = None,
        rrf_ranking_alpha: float = 1,
    ) -> List[List[Dict[str, Any]]]:
        """Finds the k closes neighbors of each instance of embeddings.
        Args:
            embeddings: List of embedding vectors.
            sparse_embeddings: List of Sparse embedding dictionaries which represents an
                embedding as a list of dimensions and as a list of sparse values:
                    ie. [{"values": [0.7, 0.5], "dimensions": [10, 20]}]
            k: Number of neighbors to be retrieved.
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between 0 and 1.0
                Weights Dense Search VS Sparse Search, as an example:
                - rrf_ranking_alpha=1: Only Dense
                - rrf_ranking_alpha=0: Only Sparse
                - rrf_ranking_alpha=0.7: 0.7 weighting for dense and 0.3 for sparse
            filter_: List of filters to apply.
        Returns:
            List of records: [
                {
                    "doc_id": doc_id,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score
                }
            ]
        """

        # No need to implement other method for private VPC, find_neighbors now works
        # with public and private.
        _, user_agent = get_user_agent("vertex-ai-matching-engine")
        with telemetry.tool_context_manager(user_agent):
            if sparse_embeddings is None:
                queries = embeddings
            else:
                if len(sparse_embeddings) != len(embeddings):
                    raise ValueError(
                        "The number of `sparse_embeddings` should match the number of "
                        f"`embeddings` {len(sparse_embeddings)} != {len(embeddings)}"
                    )
                queries = []

                for embedding, sparse_embedding in zip(embeddings, sparse_embeddings):
                    hybrid_query = HybridQuery(
                        sparse_embedding_dimensions=sparse_embedding["dimensions"],  # type: ignore
                        sparse_embedding_values=sparse_embedding["values"],  # type: ignore
                        dense_embedding=embedding,
                        rrf_ranking_alpha=rrf_ranking_alpha,
                    )
                    queries.append(hybrid_query)  # type: ignore

            response = self._endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=queries,
                num_neighbors=k,
                filter=filter_,
                numeric_filter=numeric_filter,
            )

        return self._postprocess_response(response)

    def _get_deployed_index_id(self) -> str:
        """Gets the deployed index id that matches with the provided index.
        Raises:
            ValueError if the index provided is not found in the endpoint.
        """
        for index in self._endpoint.deployed_indexes:
            if index.index == self._index.resource_name:
                return index.id

        raise ValueError(
            f"No index with id {self._index.resource_name} "
            f"deployed on endpoint "
            f"{self._endpoint.display_name}."
        )
