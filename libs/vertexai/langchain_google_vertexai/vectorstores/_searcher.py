from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, cast

from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from google.cloud.aiplatform import telemetry
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
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


class Searcher(ABC):
    """Abstract implementation of a similarity searcher."""

    @abstractmethod
    def find_neighbors(
        self,
        embeddings: List[List[float]],
        k: int = 4,
        filter_: Union[List[Namespace], None] = None,
        numeric_filter: Union[List[NumericNamespace], None] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Finds the k closes neighbors of each instance of embeddings.
        Args:
            embedding: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: List of filters to apply.
        Returns:
            List of lists of Tuples (id, distance) for each embedding vector.
        """
        raise NotImplementedError()

    @abstractmethod
    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Union[List[dict], None] = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index.

        Args:
            ids: List of unique ids.
            embeddings: List of embedddings for each record.
            metadatas: List of metadata of each record.
        """
        raise NotImplementedError()

    def _postprocess_response(
        self, response: List[List[MatchNeighbor]]
    ) -> List[List[Tuple[str, float]]]:
        """Posproceses an endpoint response and converts it to a list of list of
        tuples instead of using vertexai objects.
        Args:
            response: Endpoint response.
        Returns:
            List of list of tuples of (id, distance).
        """
        return [
            [
                (neighbor.id, cast(float, neighbor.distance))
                for neighbor in matching_neighbor_list
            ]
            for matching_neighbor_list in response
        ]


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

    def add_to_index(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Union[List[dict], None] = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Adds documents to the index.

        Args:
            ids: List of unique ids.
            embeddings: List of embedddings for each record.
            metadatas: List of metadata of each record.
            is_complete_overwrite: Whether to overwrite everything.
        """

        data_points = to_data_points(ids, embeddings, metadatas)

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
    ) -> List[List[Tuple[str, float]]]:
        """Finds the k closes neighbors of each instance of embeddings.
        Args:
            embedding: List of embeddings vectors.
            k: Number of neighbors to be retrieved.
            filter_: List of filters to apply.
        Returns:
            List of lists of Tuples (id, distance) for each embedding vector.
        """

        # No need to implement other method for private VPC, find_neighbors now works
        # with public and private.
        _, user_agent = get_user_agent("vertex-ai-matching-engine")
        with telemetry.tool_context_manager(user_agent):
            response = self._endpoint.find_neighbors(
                deployed_index_id=self._deployed_index_id,
                queries=embeddings,
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
