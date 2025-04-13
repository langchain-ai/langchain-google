import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from google.oauth2.service_account import Credentials
from langchain_core._api.deprecation import deprecated
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
    """Represents a base vector store based on VertexAI."""

    def __init__(
        self,
        searcher: Searcher,
        document_storage: DocumentStorage,
        embbedings: Optional[Embeddings] = None,  # Deprecated parameter
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """Constructor.

        Args:
            searcher: Object in charge of searching and storing the index.
            document_storage: Object in charge of storing and retrieving documents.
            embbedings: Object in charge of transforming text to embbeddings.
                Deprecated: Use 'embeddings' instead.
            embeddings: Object in charge of transforming text to embeddings.
        """
        super().__init__()
        self._searcher = searcher
        self._document_storage = document_storage

        # Add explicit warning when the misspelled parameter is used
        if embbedings is not None:
            warnings.warn(
                message=(
                    "The parameter `embbedings` is deprecated due to a spelling error. "
                    "Please use `embeddings` instead. "
                    "Support for `embbedings` will be removed in a future version."
                ),
                category=DeprecationWarning,
            )
        self._embeddings = embeddings or embbedings or self._get_default_embeddings()

    @property
    @deprecated(since="0.1.0", removal="3.0.0", alternative="embeddings")
    def embbedings(self) -> Embeddings:
        """Returns the embeddings object."""
        return self._embeddings

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        numeric_filter: Optional[List[NumericNamespace]] = None,
    ) -> List[Tuple[Document, Union[float, Dict[str, float]]]]:
        """Return docs most similar to query and their cosine distance from the query.

        Args:
            query: String query look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Optional. A list of Namespaces for filtering
                the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.
            numeric_filter: Optional. A list of NumericNamespaces for filterning
                the matching results. Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Higher score represents more similarity.
        """

        embedding = self._embeddings.embed_query(query)

        return self.similarity_search_by_vector_with_score(
            embedding=embedding, k=k, filter=filter, numeric_filter=numeric_filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        sparse_embedding: Optional[Dict[str, Union[List[int], List[float]]]] = None,
        k: int = 4,
        rrf_ranking_alpha: float = 1,
        filter: Optional[List[Namespace]] = None,
        numeric_filter: Optional[List[NumericNamespace]] = None,
    ) -> List[Tuple[Document, Union[float, Dict[str, float]]]]:
        """Return docs most similar to the embedding and their cosine distance.

        Args:
            embedding: Embedding to look up documents similar to.
            sparse_embedding: Sparse embedding dictionary which represents an embedding
                as a list of dimensions and as a list of sparse values:
                    ie. {"values": [0.7, 0.5], "dimensions": [10, 20]}
            k: Number of Documents to return. Defaults to 4.
            rrf_ranking_alpha: Reciprocal Ranking Fusion weight, float between 0 and 1.0
                Weights Dense Search VS Sparse Search, as an example:
                - rrf_ranking_alpha=1: Only Dense
                - rrf_ranking_alpha=0: Only Sparse
                - rrf_ranking_alpha=0.7: 0.7 weighting for dense and 0.3 for sparse
            filter: Optional. A list of Namespaces for filtering
                the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.
            numeric_filter: Optional. A list of NumericNamespaces for filterning
                the matching results. Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.

        Returns:
            List[Tuple[Document, Union[float, Dict[str, float]]]]:
            List of documents most similar to the query text and either
            cosine distance in float for each or dictionary with both dense and sparse
            scores if running hybrid search.
            Higher score represents more similarity.
        """
        if sparse_embedding is not None and not isinstance(sparse_embedding, dict):
            raise ValueError(
                "`sparse_embedding` should be a dictionary with the following format: "
                "{'values': [0.7, 0.5, ...], 'dimensions': [10, 20, ...]}\n"
                f"{type(sparse_embedding)} != {type({})}"
            )

        sparse_embeddings = [sparse_embedding] if sparse_embedding is not None else None
        neighbors_list = self._searcher.find_neighbors(
            embeddings=[embedding],
            sparse_embeddings=sparse_embeddings,
            k=k,
            rrf_ranking_alpha=rrf_ranking_alpha,
            filter_=filter,
            numeric_filter=numeric_filter,
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
            return list(zip(documents, distances))  # type: ignore
        else:
            missing_docs = [key for key, doc in zip(keys, documents) if doc is None]
            message = f"Documents with ids: {missing_docs} not found in the storage"
            raise ValueError(message)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete by vector ID.
        Args:
            ids (Optional[List[str]]): List of ids to delete.
            **kwargs (Any): If added metadata={}, deletes the documents
            that match the metadata filter and the parameter ids is not needed.
        Returns:
            Optional[bool]: True if deletion is successful.
        Raises:
            ValueError: If ids is None or an empty list.
            RuntimeError: If an error occurs during the deletion process.
        """
        metadata = kwargs.get("metadata")
        if (not ids and not metadata) or (ids and metadata):
            raise ValueError(
                "You should provide ids (as list of id's) or a metadata"
                "filter for deleting documents."
            )
        if metadata:
            ids = self._searcher.get_datapoints_by_filter(metadata=metadata)
            if not ids:
                return False
        try:
            self._searcher.remove_datapoints(datapoint_ids=ids)  # type: ignore[arg-type]
            self._document_storage.mdelete(ids)  # type: ignore[arg-type]
            return True
        except Exception as e:
            raise RuntimeError(f"Error during deletion: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        numeric_filter: Optional[List[NumericNamespace]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: The string that will be used to search for similar documents.
            k: The amount of neighbors that will be retrieved.
            filter: Optional. A list of Namespaces for filtering the matching results.
                For example:
                [Namespace("color", ["red"], []), Namespace("shape", [], ["squared"])]
                will match datapoints that satisfy "red color" but not include
                datapoints with "squared shape". Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                 for more detail.
            numeric_filter: Optional. A list of NumericNamespaces for filterning
                the matching results. Please refer to
                https://cloud.google.com/vertex-ai/docs/matching-engine/filtering#json
                for more detail.

        Returns:
            A list of k matching documents.
        """
        return [
            document
            for document, _ in self.similarity_search_with_score(
                query, k, filter, numeric_filter
            )
        ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Union[List[dict], None] = None,
        *,
        ids: Optional[List[str]] = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to be assigned to the texts in the index.
                If None, unique ids will be generated.
            is_complete_overwrite: Optional, determines whether this is an append or
                overwrite operation. Only relevant for BATCH UPDATE indexes.
            kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        # Makes sure is a list and can get the length, should we support iterables?
        # metadata is a list so probably not?
        texts = list(texts)
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
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Union[List[dict], None] = None,
        *,
        sparse_embeddings: Optional[
            List[Dict[str, Union[List[int], List[float]]]]
        ] = None,
        ids: Optional[List[str]] = None,
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        if ids is not None and len(set(ids)) != len(ids):
            raise ValueError(
                "All provided ids should be unique."
                f"There are {len(ids)-len(set(ids))} duplicates."
            )

        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                "The number of `ids` should match the number of `texts` "
                f"{len(ids)} != {len(texts)}"
            )

        if isinstance(embeddings, list) and len(embeddings) != len(texts):
            raise ValueError(
                "The number of `embeddings` should match the number of `texts` "
                f"{len(embeddings)} != {len(texts)}"
            )

        if ids is None:
            ids = self._generate_unique_ids(len(texts))

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            raise ValueError(
                "`metadatas` should be the same length as `texts` "
                f"{len(metadatas)} != {len(texts)}"
            )

        documents = [
            Document(id=id_, page_content=text, metadata=metadata)
            for id_, text, metadata in zip(ids, texts, metadatas)
        ]

        self._document_storage.mset(list(zip(ids, documents)))

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
        cls: Type["_BaseVertexAIVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Union[List[dict], None] = None,
        **kwargs: Any,
    ) -> "_BaseVertexAIVectorStore":
        """Use from components instead."""
        raise NotImplementedError(
            "This method is not implemented. Instead, you should initialize the class"
            " with `VertexAIVectorSearch.from_components(...)` and then call "
            "`add_texts`"
        )

    @classmethod
    def _get_default_embeddings(cls) -> Embeddings:
        """This function returns the default embedding.

        Returns:
            Default TensorflowHubEmbeddings to use.
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

    def _generate_unique_ids(self, number: int) -> List[str]:
        """Generates a list of unique ids of length `number`

        Args:
            number: Number of ids to generate.

        Returns:
            List of unique ids.
        """
        return [str(uuid.uuid4()) for _ in range(number)]


class VectorSearchVectorStore(_BaseVertexAIVectorStore):
    """VertexAI VectorStore that handles the search and indexing using Vector Search
    and stores the documents in Google Cloud Storage.
    """

    @classmethod
    def from_components(  # Implemented in order to keep the current API
        cls: Type["VectorSearchVectorStore"],
        project_id: str,
        region: str,
        gcs_bucket_name: str,
        index_id: str,
        endpoint_id: str,
        private_service_connect_ip_address: Optional[str] = None,
        credentials: Optional[Credentials] = None,
        credentials_path: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
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
            credentials: Google cloud Credentials object.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.
            stream_update: Whether to update with streaming or batching. VectorSearch
                index must be compatible with stream/batch updates.
            kwargs: Additional keyword arguments to pass to
                VertexAIVectorSearch.__init__().

        Returns:
            A configured VertexAIVectorSearch.
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
    """VectorSearch with DatasTore document storage."""

    @classmethod
    def from_components(
        cls: Type["VectorSearchVectorStoreDatastore"],
        project_id: str,
        region: str,
        index_id: str,
        endpoint_id: str,
        index_staging_bucket_name: Optional[str] = None,
        credentials: Optional[Credentials] = None,
        credentials_path: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        stream_update: bool = False,
        datastore_client_kwargs: Optional[Dict[str, Any]] = None,
        exclude_from_indexes: Optional[List[str]] = None,
        datastore_kind: str = "document_id",
        datastore_text_property_name: str = "text",
        datastore_metadata_property_name: str = "metadata",
        **kwargs: Dict[str, Any],
    ) -> "VectorSearchVectorStoreDatastore":
        """Takes the object creation out of the constructor.

        Args:
            project_id: The GCP project id.
            region: The default location making the API calls. It must have
                the same location as the GCS bucket and must be regional.
            index_id: The id of the created index.
            endpoint_id: The id of the created endpoint.
            index_staging_bucket_name: (Optional) If the index is updated by batch,
                bucket where the data will be staged before updating the index. Only
                required when updating the index.
            credentials: Google cloud Credentials object.
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.
            stream_update: Whether to update with streaming or batching. VectorSearch
                index must be compatible with stream/batch updates.
            kwargs: Additional keyword arguments to pass to
                VertexAIVectorSearch.__init__().
            exclude_from_indexes: Fields to exclude from datastore indexing

        Returns:
            A configured VectorSearchVectorStoreDatastore.
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
