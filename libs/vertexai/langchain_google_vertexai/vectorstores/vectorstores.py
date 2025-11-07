import uuid
import warnings
from collections.abc import Iterable
from typing import Any

from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from google.oauth2.service_account import Credentials
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_google_vertexai.vectorstores._sdk_manager import VectorSearchSDKManager
from langchain_google_vertexai.vectorstores._searcher import (
    Searcher,
    VectorSearchSearcher,
)
from langchain_google_vertexai.vectorstores.document_storage import (
    DataStoreDocumentStorage,
    DocumentStorage,
    GCSDocumentStorage,
)


class _BaseVertexAIVectorStore(VectorStore):
    """Represents a base `VectorStore` based on VertexAI."""

    def __init__(
        self,
        searcher: Searcher,
        document_storage: DocumentStorage,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Constructor.

        Args:
            searcher: Object in charge of searching and storing the index.
            document_storage: Object in charge of storing and retrieving documents.
            embeddings: Object in charge of transforming text to embeddings.
        """
        super().__init__()
        self._searcher = searcher
        self._document_storage = document_storage

        self._embeddings = embeddings or self._get_default_embeddings()

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: list[Namespace] | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | dict[str, float]]]:
        """Return docs most similar to query and their cosine distance from the query.

        Args:
            query: String query look up documents similar to.
            k: Number of Documents to return.
            filter: A list of `Namespace` objects for filtering the matching results.

                For example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results.

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)

        Returns:
            List of `Document` objects most similar to the query text and cosine
                distance in float for each.

                Higher score represents more similarity.
        """
        embedding = self._embeddings.embed_query(query)

        return self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter,
            numeric_filter=numeric_filter,
            **kwargs,
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: list[float],
        sparse_embedding: dict[str, list[int] | list[float]] | None = None,
        k: int = 4,
        rrf_ranking_alpha: float = 1,
        filter: list[Namespace] | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | dict[str, float]]]:
        """Return docs most similar to the embedding and their cosine distance.

        Args:
            embedding: Embedding to look up documents similar to.
            sparse_embedding: Sparse embedding dictionary which represents an embedding
                as a list of dimensions and as a list of sparse values:

                i.e. `{"values": [0.7, 0.5], "dimensions": [10, 20]}`
            k: Number of documents to return.
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between `0` and
                `1.0`

                Weights Dense Search VS Sparse Search, as an example:
                - `rrf_ranking_alpha=1`: Only Dense
                - `rrf_ranking_alpha=0`: Only Sparse
                - `rrf_ranking_alpha=0.7`: `0.7` weighting for dense and `0.3` for
                    sparse
            filter: A list of `Namespace` objects for filtering the matching results.

                For example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results.

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)

        Returns:
            List of `Document` objects most similar to the query text and either
                cosine distance in float for each or dictionary with both dense and
                sparse scores if running hybrid search.

                Higher score represents more similarity.
        """
        if sparse_embedding is not None and not isinstance(sparse_embedding, dict):
            msg = (  # type: ignore[unreachable]
                "`sparse_embedding` should be a dictionary with the following format: "
                "{'values': [0.7, 0.5, ...], 'dimensions': [10, 20, ...]}\n"
                f"{type(sparse_embedding)} != {type({})}"
            )
            raise ValueError(msg)

        sparse_embeddings = [sparse_embedding] if sparse_embedding is not None else None
        neighbors_list = self._searcher.find_neighbors(
            embeddings=[embedding],
            sparse_embeddings=sparse_embeddings,
            k=k,
            rrf_ranking_alpha=rrf_ranking_alpha,
            filter_=filter,
            numeric_filter=numeric_filter,
            **kwargs,
        )
        if not neighbors_list:
            return []

        keys = [elem["doc_id"] for elem in neighbors_list[0]]
        if sparse_embedding is None:
            distances = [elem["dense_score"] for elem in neighbors_list[0]]
        else:
            distances = [
                {
                    "dense_score": elem["dense_score"],
                    "sparse_score": elem["sparse_score"],
                }
                for elem in neighbors_list[0]
            ]
        documents = self._document_storage.mget(keys)

        if all(document is not None for document in documents):
            # Ignore typing because mypy doesn't seem to be able to identify that
            # in documents there is no possibility to have None values with the
            # check above.
            return list(zip(documents, distances, strict=False))  # type: ignore
        missing_docs = [
            key for key, doc in zip(keys, documents, strict=False) if doc is None
        ]
        message = f"Documents with ids: {missing_docs} not found in the storage"
        raise ValueError(message)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by vector ID.

        Args:
            ids: List of IDs to delete.
            **kwargs: If added, `metadata={}`, deletes the documents
                that match the metadata filter and the parameter IDs is not needed.

        Returns:
            `True` if deletion is successful.

        Raises:
            ValueError: If `ids` is `None` or an empty list.
            RuntimeError: If an error occurs during the deletion process.
        """
        metadata = kwargs.get("metadata")
        if (not ids and not metadata) or (ids and metadata):
            msg = (
                "You should provide ids (as list of IDs) or a metadata"
                "filter for deleting documents."
            )
            raise ValueError(msg)
        if metadata:
            ids = self._searcher.get_datapoints_by_filter(metadata=metadata)
            if not ids:
                return False
        try:
            self._searcher.remove_datapoints(datapoint_ids=ids)  # type: ignore[arg-type]
            self._document_storage.mdelete(ids)  # type: ignore[arg-type]
            return True
        except Exception as e:
            msg = f"Error during deletion: {e!s}"
            raise RuntimeError(msg) from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: list[Namespace] | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.
            filter: A list of `Namespace` objects for filtering the matching results.

                For example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results.

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)

        Returns:
            A list of `k` matching documents.
        """
        return [
            document
            for document, _ in self.similarity_search_with_score(
                query, k, filter, numeric_filter, **kwargs
            )
        ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: Iterable of strings to add to the `VectorStore`.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of IDs to be assigned to the texts in the index.

                If `None`, unique ids will be generated.
            is_complete_overwrite: Optional, determines whether this is an append or
                overwrite operation.

                Only relevant for `BATCH UPDATE` indexes.
            kwargs: `VectorStore` specific parameters.

        Returns:
            List of IDs from adding the texts into the `VectorStore`.
        """
        # Makes sure is a list and can get the length, should we support iterables?
        # metadata is a list so probably not?
        texts = [texts] if isinstance(texts, str) else list(texts)

        embeddings = self._embeddings.embed_documents(texts)

        return self.add_texts_with_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            is_complete_overwrite=is_complete_overwrite,
            **kwargs,
        )

    def add_texts_with_embeddings(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        *,
        sparse_embeddings: list[dict[str, list[int] | list[float]]] | None = None,
        ids: list[str] | None = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        if ids is not None and len(set(ids)) != len(ids):
            msg = (
                "All provided IDs should be unique."
                f"There are {len(ids) - len(set(ids))} duplicates."
            )
            raise ValueError(msg)

        if ids is not None and len(ids) != len(texts):
            msg = (
                "The number of `ids` should match the number of `texts` "
                f"{len(ids)} != {len(texts)}"
            )
            raise ValueError(msg)

        if isinstance(embeddings, list) and len(embeddings) != len(texts):
            msg = (
                "The number of `embeddings` should match the number of `texts` "
                f"{len(embeddings)} != {len(texts)}"
            )
            raise ValueError(msg)

        if ids is None:
            ids = self._generate_unique_ids(len(texts))

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            msg = (
                "`metadatas` should be the same length as `texts` "
                f"{len(metadatas)} != {len(texts)}"
            )
            raise ValueError(msg)

        documents = [
            Document(id=id_, page_content=text, metadata=metadata)
            for id_, text, metadata in zip(ids, texts, metadatas, strict=False)
        ]

        self._document_storage.mset(list(zip(ids, documents, strict=False)))

        self._searcher.add_to_index(
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            metadatas=metadatas,
            is_complete_overwrite=is_complete_overwrite,
            **kwargs,
        )

        return ids

    @classmethod
    def from_texts(
        cls: type["_BaseVertexAIVectorStore"],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> "_BaseVertexAIVectorStore":
        """Use from components instead."""
        msg = (
            "This method is not implemented. Instead, you should initialize the class"
            " with `VertexAIVectorSearch.from_components(...)` and then call "
            "`add_texts`"
        )
        raise NotImplementedError(msg)

    @classmethod
    def _get_default_embeddings(cls) -> Embeddings:
        """This function returns the default embedding.

        Returns:
            Default `TensorflowHubEmbeddings` to use.
        """
        warnings.warn(
            message=(
                "`TensorflowHubEmbeddings` as a default embeddings is deprecated."
                " Will change to `VertexAIEmbeddings`. Please specify the embedding "
                "type in the constructor."
            ),
            category=DeprecationWarning,
        )

        # TODO: Change to vertexai embeddings
        from langchain_community.embeddings import (  # type: ignore[import-not-found, unused-ignore]
            TensorflowHubEmbeddings,
        )

        return TensorflowHubEmbeddings()

    def _generate_unique_ids(self, number: int) -> list[str]:
        """Generates a list of unique ids of length `number`.

        Args:
            number: Number of ids to generate.

        Returns:
            List of unique ids.
        """
        return [str(uuid.uuid4()) for _ in range(number)]


class VectorSearchVectorStore(_BaseVertexAIVectorStore):
    """VertexAI `VectorStore` that handles the search and indexing using Vector Search
    and stores the documents in Google Cloud Storage.
    """

    @classmethod
    def from_components(  # Implemented in order to keep the current API
        cls: type["VectorSearchVectorStore"],
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        index_id: str,
        endpoint_id: str,
        private_service_connect_ip_address: str | None = None,
        credentials: Credentials | None = None,
        credentials_path: str | None = None,
        embedding: Embeddings | None = None,
        stream_update: bool = False,
        **kwargs: Any,
    ) -> "VectorSearchVectorStore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
                the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
                order for the index to be created.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            private_service_connect_ip_address: The IP address of the private
                service connect instance.
            credentials: Google cloud `Credentials` object.
            credentials_path: The path of the Google credentials on the local file
                system.
            embedding: The `Embeddings` that will be used for embedding the texts.
            stream_update: Whether to update with streaming or batching. `VectorSearch`
                index must be compatible with stream/batch updates.
            kwargs: Additional keyword arguments to pass to
                `VertexAIVectorSearch.__init__()`.

        Returns:
            A configured `VertexAIVectorSearch`.
        """
        sdk_manager = VectorSearchSDKManager(
            project_id=project_id,
            region=region,
            credentials=credentials,
            credentials_path=credentials_path,
        )
        bucket = sdk_manager.get_gcs_bucket(bucket_name=gcs_bucket_name)
        index = sdk_manager.get_index(index_id=index_id)
        endpoint = sdk_manager.get_endpoint(endpoint_id=endpoint_id)

        if private_service_connect_ip_address:
            endpoint.private_service_connect_ip_address = (
                private_service_connect_ip_address
            )

        return cls(
            document_storage=GCSDocumentStorage(bucket=bucket),
            searcher=VectorSearchSearcher(
                endpoint=endpoint,
                index=index,
                staging_bucket=bucket,
                stream_update=stream_update,
            ),
            embeddings=embedding,
        )


class VectorSearchVectorStoreGCS(VectorSearchVectorStore):
    """Alias of `VectorSearchVectorStore` for consistency with the rest of vector
    stores with different document storage backends.
    """


class VectorSearchVectorStoreDatastore(_BaseVertexAIVectorStore):
    """VectorSearch with DataStore document storage."""

    @classmethod
    def from_components(
        cls: type["VectorSearchVectorStoreDatastore"],
        project_id: str,
        region: str,
        index_id: str,
        endpoint_id: str,
        index_staging_bucket_name: str | None = None,
        credentials: Credentials | None = None,
        credentials_path: str | None = None,
        embedding: Embeddings | None = None,
        stream_update: bool = False,
        datastore_client_kwargs: dict[str, Any] | None = None,
        exclude_from_indexes: list[str] | None = None,
        datastore_kind: str = "document_id",
        datastore_text_property_name: str = "text",
        datastore_metadata_property_name: str = "metadata",
        **kwargs: dict[str, Any],
    ) -> "VectorSearchVectorStoreDatastore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls.

                Must have the same location as the GCS bucket and must be regional.
            index_id: The ID of the created index.
            endpoint_id: The ID of the created endpoint.
            index_staging_bucket_name: If the index is updated by batch,
                bucket where the data will be staged before updating the index.

                Only required when updating the index.
            credentials: Google cloud `Credentials` object.
            credentials_path: The path of the Google credentials on the local file
                system.
            embedding: The `Embeddings` that will be used for embedding the texts.
            stream_update: Whether to update with streaming or batching. VectorSearch
                index must be compatible with stream/batch updates.
            datastore_client_kwargs: Additional keyword arguments to pass to the
                datastore client.
            exclude_from_indexes: Fields to exclude from datastore indexing.
            datastore_kind: Datastore kind name.
            datastore_text_property_name: Property name for storing text content.
            datastore_metadata_property_name: Property name for storing metadata.
            kwargs: Additional keyword arguments to pass to
                `VertexAIVectorSearch.__init__()`.

        Returns:
            A configured `VectorSearchVectorStoreDatastore`.
        """
        sdk_manager = VectorSearchSDKManager(
            project_id=project_id,
            region=region,
            credentials=credentials,
            credentials_path=credentials_path,
        )

        if index_staging_bucket_name is not None:
            bucket = sdk_manager.get_gcs_bucket(bucket_name=index_staging_bucket_name)
        else:
            bucket = None

        index = sdk_manager.get_index(index_id=index_id)
        endpoint = sdk_manager.get_endpoint(endpoint_id=endpoint_id)

        if datastore_client_kwargs is None:
            datastore_client_kwargs = {}

        datastore_client = sdk_manager.get_datastore_client(**datastore_client_kwargs)

        if exclude_from_indexes is None:
            exclude_from_indexes = []
        document_storage = DataStoreDocumentStorage(
            datastore_client=datastore_client,
            kind=datastore_kind,
            text_property_name=datastore_text_property_name,
            metadata_property_name=datastore_metadata_property_name,
            exclude_from_indexes=exclude_from_indexes,
        )

        return cls(
            document_storage=document_storage,
            searcher=VectorSearchSearcher(
                endpoint=endpoint,
                index=index,
                staging_bucket=bucket,
                stream_update=stream_update,
            ),
            embeddings=embedding,
        )
