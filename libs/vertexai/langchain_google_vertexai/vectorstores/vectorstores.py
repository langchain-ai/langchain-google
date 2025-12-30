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
        filter: list[Namespace] | dict | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | dict[str, float]]]:
        """Return docs most similar to query and their cosine distance from the query.

        Args:
            query: String query look up documents similar to.
            k: Number of Documents to return.
            filter: For V1: A list of `Namespace` objects for filtering.
                For V2: A dict filter.

                V1 example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                V2 example:
                `{"color": {"$eq": "blue"}}` or
                `{"$and": [{"color": {"$eq": "blue"}}, {"price": {"$lt": 1000}}]}`

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results. (V1 only)

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
        filter: list[Namespace] | dict | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float | dict[str, float]]]:
        """Return docs most similar to the embedding and their cosine distance.

        Args:
            embedding: Embedding to look up documents similar to.
            sparse_embedding: Sparse embedding dictionary which represents an embedding
                as a list of indices and as a list of sparse values:

                i.e. `{"values": [0.7, 0.5], "indices": [10, 20]}`
            k: Number of documents to return.
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between `0` and
                `1.0`

                Weights Dense Search VS Sparse Search, as an example:
                - `rrf_ranking_alpha=1`: Only Dense
                - `rrf_ranking_alpha=0`: Only Sparse
                - `rrf_ranking_alpha=0.7`: `0.7` weighting for dense and `0.3` for
                    sparse
            filter: For V1: A list of `Namespace` objects for filtering.
                For V2: A dict filter.

                V1 example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                V2 example:
                `{"color": {"$eq": "blue"}}` or
                `{"$and": [{"color": {"$eq": "blue"}}, {"price": {"$lt": 15000}}]}`

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results. Only supported in V1.

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

        # V2: Documents stored in collection metadata, reconstruct from search results
        # V1: Documents in GCS, retrieve from document storage
        if self._searcher._api_version == "v2" and self._document_storage is None:  # type: ignore[attr-defined]
            documents = []  # type: ignore[unreachable]
            for elem in neighbors_list[0]:
                metadata = elem.get("metadata", {})
                page_content = metadata.pop("page_content", "")
                doc = Document(
                    id=elem["doc_id"],
                    page_content=page_content,
                    metadata=metadata,
                )
                documents.append(doc)
        else:
            # V1: Retrieve documents from GCS storage
            documents = self._document_storage.mget(keys)

            if all(document is not None for document in documents):
                # Ignore typing because mypy doesn't seem to be able to identify that
                # in documents there is no possibility to have None values with the
                # check above.
                pass
            else:
                missing_docs = [
                    key
                    for key, doc in zip(keys, documents, strict=False)
                    if doc is None
                ]
                message = f"Documents with ids: {missing_docs} not found in the storage"
                raise ValueError(message)

        return list(zip(documents, distances, strict=False))  # type: ignore

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
            # V2: No separate storage to delete from
            # V1 and others: Also delete from GCS document storage
            if self._searcher._api_version == "v2":  # type: ignore[attr-defined]
                pass  # V2 doesn't use separate document storage
            else:
                # Original V1 behavior
                self._document_storage.mdelete(ids)  # type: ignore[arg-type]
            return True
        except Exception as e:
            msg = f"Error during deletion: {e!s}"
            raise RuntimeError(msg) from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: list[Namespace] | dict | None = None,
        numeric_filter: list[NumericNamespace] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.
            filter: For V1: A list of `Namespace` objects for filtering.
                For V2: A dict filter.

                V1 example:
                `[Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]`
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape".

                V2 example:
                `{"color": {"$eq": "blue"}}` or
                `{"$and": [{"color": {"$eq": "blue"}}, {"price": {"$lt": 15000}}]}`

                [More details](https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json)
            numeric_filter: A list of `NumericNamespace` objects for filtering the
                matching results. Only supported in V1.

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

    def semantic_search(
        self,
        query: str,
        k: int = 4,
        search_field: str = "embedding",
        task_type: str = "RETRIEVAL_QUERY",
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Performs semantic search using auto-generated embeddings.

        Semantic search automatically generates embeddings from the query text using
        Vertex AI models, so you don't need to manually create embeddings. This is
        only supported in Vector Search 2.0.

        Args:
            query: Natural language query text.
            k: Number of documents to return.
            search_field: Name of the vector field to search (must have auto-embedding
                config in the collection schema).
            task_type: Embedding task type. Options:
                - "RETRIEVAL_QUERY": For search queries (default)
                - "RETRIEVAL_DOCUMENT": For document indexing
                - "SEMANTIC_SIMILARITY": For semantic similarity
                - "CLASSIFICATION": For classification tasks
                - "CLUSTERING": For clustering tasks
            filter: Filter dict (v2 only).
                Example: `{"color": {"$eq": "blue"}}` or
                `{"$and": [{"year": {"$gte": 1990}}, {"genre": {"$eq": "Action"}}]}`

        Returns:
            List of matching documents.
        """
        results = self._searcher.semantic_search(
            search_text=query,
            search_field=search_field,
            k=k,
            task_type=task_type,
            filter_=filter,
            **kwargs,
        )

        return self._results_to_documents(results)

    def text_search(
        self,
        query: str,
        k: int = 4,
        data_field_names: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Performs keyword/full-text search on data fields.

        Text search performs traditional keyword matching on data fields without using
        embeddings. This is only supported in Vector Search 2.0.

        Note: Text search does not support filters. Use `semantic_search()` or
        `similarity_search()` if you need filtering.

        Args:
            query: Keyword search query text.
            k: Number of documents to return.
            data_field_names: List of data field names to search in
                (e.g., `["page_content", "title"]`).
                If `None`, defaults to `["page_content"]`.

        Returns:
            List of matching documents.
        """
        if data_field_names is None:
            data_field_names = ["page_content"]

        results = self._searcher.text_search(
            search_text=query,
            data_field_names=data_field_names,
            k=k,
            **kwargs,
        )

        return self._results_to_documents(results)

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        search_field: str = "embedding",
        data_field_names: list[str] | None = None,
        task_type: str = "RETRIEVAL_QUERY",
        filter: dict | None = None,
        semantic_weight: float = 1.0,
        text_weight: float = 1.0,
        **kwargs: Any,
    ) -> list[Document]:
        """Performs hybrid search combining semantic and text search with RRF.

        Hybrid search automatically combines semantic search (with auto-generated
        embeddings) and text search (keyword matching) using Reciprocal Rank Fusion
        (RRF) algorithm to produce optimally ranked results. This is only supported
        in Vector Search 2.0.

        Products appearing high in both semantic and text search results will rank
        highest in the final merged results.

        Args:
            query: Query text used for both semantic and text search.
            k: Number of documents to return from each search before fusion.
            search_field: Name of the vector field to search (must have auto-embedding
                config in the collection schema).
            data_field_names: List of data field names to search in for text search
                (e.g., `["page_content", "title"]`).
                If `None`, defaults to `["page_content"]`.
            task_type: Embedding task type for semantic search. Options:
                - "RETRIEVAL_QUERY": For search queries (default)
                - "RETRIEVAL_DOCUMENT": For document indexing
                - "SEMANTIC_SIMILARITY": For semantic similarity
                - "CLASSIFICATION": For classification tasks
                - "CLUSTERING": For clustering tasks
            filter: Filter dict for semantic search only (v2 only).
                Example: `{"color": {"$eq": "blue"}}` or
                `{"$and": [{"year": {"$gte": 1990}}, {"genre": {"$eq": "Action"}}]}`
            semantic_weight: Weight for semantic search results in RRF (default: 1.0).
                Higher values give more importance to semantic similarity.
            text_weight: Weight for text search results in RRF (default: 1.0).
                Higher values give more importance to keyword matches.

        Returns:
            List of documents ranked by RRF combining semantic and text search.

        Example:
            ```python
            # Equal weighting (default)
            results = vector_store.hybrid_search("Men's outfit for beach", k=10)

            # Prefer semantic understanding over keyword matching
            results = vector_store.hybrid_search(
                "beach wear", k=10, semantic_weight=2.0, text_weight=1.0
            )

            # With filtering on semantic search
            results = vector_store.hybrid_search(
                "summer dress", k=10, filter={"price": {"$lt": 100}}
            )
            ```
        """
        if data_field_names is None:
            data_field_names = ["page_content"]

        results = self._searcher.hybrid_search(
            search_text=query,
            search_field=search_field,
            data_field_names=data_field_names,
            k=k,
            task_type=task_type,
            filter_=filter,
            semantic_weight=semantic_weight,
            text_weight=text_weight,
            **kwargs,
        )

        return self._results_to_documents(results)

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

        # Add document IDs and page_content to metadata
        metadatas_with_ids = []
        for id_, text, metadata in zip(ids, texts, metadatas, strict=False):
            metadata_copy = metadata.copy()
            metadata_copy["id"] = id_
            # V2: Store page_content in metadata (no separate document storage)
            # V1: page_content stored separately in GCS
            if self._searcher._api_version == "v2" and self._document_storage is None:  # type: ignore[attr-defined]
                metadata_copy["page_content"] = text  # type: ignore[unreachable]
            metadatas_with_ids.append(metadata_copy)

        documents = [
            Document(id=id_, page_content=text, metadata=metadata)
            for id_, text, metadata in zip(ids, texts, metadatas_with_ids, strict=False)
        ]

        # V2: No separate storage needed (stored in collection data objects)
        # V1 and others: Store documents in GCS
        if self._searcher._api_version == "v2":  # type: ignore[attr-defined]
            pass  # V2 stores in collection data objects
        else:
            # Original V1 behavior
            self._document_storage.mset(list(zip(ids, documents, strict=False)))

        self._searcher.add_to_index(
            ids=ids,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
            metadatas=metadatas_with_ids,
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

    def _results_to_documents(self, results: list[dict[str, Any]]) -> list[Document]:
        """Converts search results to Document objects.

        Args:
            results: List of result dictionaries from search operations.
                Each result should have doc_id, and optionally metadata.

        Returns:
            List of Document objects.
        """
        documents = []
        for result in results:
            metadata = result.get("metadata", {})
            page_content = metadata.pop("page_content", "")
            doc = Document(
                id=result["doc_id"],
                page_content=page_content,
                metadata=metadata,
            )
            documents.append(doc)
        return documents


class VectorSearchVectorStore(_BaseVertexAIVectorStore):
    """VertexAI `VectorStore` that handles the search and indexing using Vector Search
    and stores the documents in Google Cloud Storage.
    """

    @classmethod
    def from_components(  # Implemented in order to keep the current API
        cls: type["VectorSearchVectorStore"],
        project_id: str,
        region: str,
        gcs_bucket_name: str | None = None,
        index_id: str | None = None,
        endpoint_id: str | None = None,
        collection_id: str | None = None,
        credentials: Credentials | None = None,
        embedding: Embeddings | None = None,
        stream_update: bool = False,
        api_version: str = "v1",
        vector_field_name: str = "embedding",
        **kwargs: Any,
    ) -> "VectorSearchVectorStore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
                the same location as the GCS bucket and must be regional.
            gcs_bucket_name: The location where the vectors will be stored in
                order for the index to be created. Required for V1, not used
                in V2.
            index_id: The id of the created index. Required for V1, not used
                in V2.
            endpoint_id: The id of the created endpoint. Required for V1, not
                used in V2.
            collection_id: The id of the created collection. Required for V2,
                not used in V1.
            credentials: Google cloud `Credentials` object.
            embedding: The `Embeddings` that will be used for embedding the texts.
            stream_update: Whether to update with streaming or batching. `VectorSearch`
                index must be compatible with stream/batch updates.
            api_version: The version of the Vector Search API to use ("v1" or "v2").
            vector_field_name: Name of the vector field in the V2 collection schema.
                Only used for V2.
            kwargs: Additional keyword arguments to pass to
                `VertexAIVectorSearch.__init__()`.

        Returns:
            A configured `VertexAIVectorSearch`.

        Raises:
            ValueError: If required parameters for the specified API version are missing
                or if incompatible parameters are provided.
        """
        # Validate parameters based on API version
        if api_version == "v1":
            # V1 requires index_id, endpoint_id, and gcs_bucket_name
            if not index_id:
                raise ValueError(
                    "index_id is required for api_version='v1'. "
                    "Please provide a valid index ID."
                )
            if not endpoint_id:
                raise ValueError(
                    "endpoint_id is required for api_version='v1'. "
                    "Please provide a valid endpoint ID."
                )
            if not gcs_bucket_name:
                raise ValueError(
                    "gcs_bucket_name is required for api_version='v1'. "
                    "Please provide a valid GCS bucket name."
                )
            # V2-exclusive parameters must not be set in V1
            if collection_id is not None:
                raise ValueError(
                    "Parameter 'collection_id' is only valid for api_version='v2'. "
                    "For v1, use index_id and endpoint_id instead."
                )
        elif api_version == "v2":
            # V2 requires collection_id
            if not collection_id:
                raise ValueError(
                    "collection_id is required for api_version='v2'. "
                    "Please provide a valid collection ID."
                )
            # V1-exclusive parameters must not be set in V2
            if index_id is not None:
                raise ValueError(
                    "Parameter 'index_id' is only valid for api_version='v1'. "
                    "For v2, use collection_id instead."
                )
            if endpoint_id is not None:
                raise ValueError(
                    "Parameter 'endpoint_id' is only valid for api_version='v1'. "
                    "For v2, collections do not use endpoints."
                )
            if gcs_bucket_name is not None:
                raise ValueError(
                    "Parameter 'gcs_bucket_name' is only valid for api_version='v1'. "
                    "V2 does not require a staging bucket."
                )
        else:
            raise ValueError(
                f"Invalid api_version: '{api_version}'. Must be 'v1' or 'v2'."
            )

        sdk_manager = VectorSearchSDKManager(
            project_id=project_id,
            region=region,
            credentials=credentials,
            api_version=api_version,
        )

        if api_version == "v1":
            bucket = sdk_manager.get_gcs_bucket(bucket_name=gcs_bucket_name)  # type: ignore[arg-type]
            index = sdk_manager.get_index(index_id=index_id)  # type: ignore[arg-type]
            endpoint = sdk_manager.get_endpoint(endpoint_id=endpoint_id)  # type: ignore[arg-type]
            searcher = VectorSearchSearcher(
                endpoint=endpoint,
                index=index,
                staging_bucket=bucket,
                stream_update=stream_update,
                api_version=api_version,
            )
            return cls(
                document_storage=GCSDocumentStorage(bucket=bucket),
                searcher=searcher,
                embeddings=embedding,
            )
        else:  # v2
            collection = sdk_manager.get_collection(collection_id=collection_id)  # type: ignore[arg-type]
            searcher = VectorSearchSearcher(
                endpoint=None,
                index=None,
                collection=collection,
                staging_bucket=None,
                stream_update=stream_update,
                api_version=api_version,
                project_id=project_id,
                region=region,
                credentials=credentials,
                vector_field_name=vector_field_name,
            )
            # V2 stores documents directly in collection metadata
            # No separate storage needed
            return cls(
                document_storage=None,  # type: ignore[arg-type]
                searcher=searcher,
                embeddings=embedding,
            )

    def get_datapoints_by_ids(self, ids: list[str]) -> Any:
        """Gets the full datapoint information from the index by ID.

        Args:
            ids: A list of datapoint IDs to retrieve.

        Returns:
            A list of the requested datapoints.
        """
        return self._searcher.get_datapoints(datapoint_ids=ids)  # type: ignore[attr-defined]


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
        index_id: str | None = None,
        endpoint_id: str | None = None,
        collection_id: str | None = None,
        index_staging_bucket_name: str | None = None,
        credentials: Credentials | None = None,
        embedding: Embeddings | None = None,
        stream_update: bool = False,
        api_version: str = "v1",
        vector_field_name: str = "embedding",
        datastore_client_kwargs: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
    ) -> "VectorSearchVectorStoreDatastore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls.

                Must have the same location as the GCS bucket and must be regional.
            index_id: The ID of the created index. Required for V1, not used
                in V2.
            endpoint_id: The ID of the created endpoint. Required for V1, not
                used in V2.
            collection_id: The ID of the created collection. Required for V2,
                not used in V1.
            index_staging_bucket_name: If the index is updated by batch,
                bucket where the data will be staged before updating the index.
                Only used in V1.
            credentials: Google cloud `Credentials` object.
            embedding: The `Embeddings` that will be used for embedding the texts.
            stream_update: Whether to update with streaming or batching. VectorSearch
                index must be compatible with stream/batch updates.
            api_version: The version of the Vector Search API to use ("v1" or "v2").
            vector_field_name: Name of the vector field in the V2 collection schema.
                Only used for V2.
            datastore_client_kwargs: Additional keyword arguments to pass to the
                datastore client.
            kwargs: Additional keyword arguments to pass to
                `VertexAIVectorSearch.__init__()`.

        Returns:
            A configured `VectorSearchVectorStoreDatastore`.

        Raises:
            ValueError: If required parameters for the specified API version are missing
                or if incompatible parameters are provided.
        """
        # Validate parameters based on API version
        if api_version == "v1":
            # V1 requires index_id and endpoint_id
            if not index_id:
                raise ValueError(
                    "index_id is required for api_version='v1'. "
                    "Please provide a valid index ID."
                )
            if not endpoint_id:
                raise ValueError(
                    "endpoint_id is required for api_version='v1'. "
                    "Please provide a valid endpoint ID."
                )
            # V2-exclusive parameters must not be set in V1
            if collection_id is not None:
                raise ValueError(
                    "Parameter 'collection_id' is only valid for api_version='v2'. "
                    "For v1, use index_id and endpoint_id instead."
                )
        elif api_version == "v2":
            # V2 requires collection_id
            if not collection_id:
                raise ValueError(
                    "collection_id is required for api_version='v2'. "
                    "Please provide a valid collection ID."
                )
            # V1-exclusive parameters must not be set in V2
            if index_id is not None:
                raise ValueError(
                    "Parameter 'index_id' is only valid for api_version='v1'. "
                    "For v2, use collection_id instead."
                )
            if endpoint_id is not None:
                raise ValueError(
                    "Parameter 'endpoint_id' is only valid for api_version='v1'. "
                    "For v2, collections do not use endpoints."
                )
        else:
            raise ValueError(
                f"Invalid api_version: '{api_version}'. Must be 'v1' or 'v2'."
            )

        sdk_manager = VectorSearchSDKManager(
            project_id=project_id,
            region=region,
            credentials=credentials,
            api_version=api_version,
        )

        bucket = None
        if index_staging_bucket_name:
            bucket = sdk_manager.get_gcs_bucket(bucket_name=index_staging_bucket_name)

        if api_version == "v1":
            index = sdk_manager.get_index(index_id=index_id)  # type: ignore[arg-type]
            endpoint = sdk_manager.get_endpoint(endpoint_id=endpoint_id)  # type: ignore[arg-type]
            searcher = VectorSearchSearcher(
                endpoint=endpoint,
                index=index,
                staging_bucket=bucket,
                stream_update=stream_update,
                api_version=api_version,
            )
        elif api_version == "v2":
            collection = sdk_manager.get_collection(collection_id=collection_id)  # type: ignore[arg-type]
            searcher = VectorSearchSearcher(
                endpoint=None,
                index=None,
                collection=collection,
                staging_bucket=bucket,
                stream_update=stream_update,
                api_version=api_version,
                project_id=project_id,
                region=region,
                credentials=credentials,
                vector_field_name=vector_field_name,
            )
        else:
            msg = f"Unsupported API version: {api_version}"
            raise ValueError(msg)

        if datastore_client_kwargs is None:
            datastore_client_kwargs = {}

        datastore_client = sdk_manager.get_datastore_client(**datastore_client_kwargs)

        document_storage = DataStoreDocumentStorage(
            datastore_client=datastore_client,
            **kwargs,  # type: ignore[arg-type]
        )

        return cls(
            document_storage=document_storage,
            searcher=searcher,
            embeddings=embedding,
        )
