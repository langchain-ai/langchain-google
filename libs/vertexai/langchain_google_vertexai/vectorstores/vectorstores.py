import uuid
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_google_vertexai.vectorstores._document_storage import (
    DocumentStorage,
    GCSDocumentStorage,
)
from langchain_google_vertexai.vectorstores._sdk_manager import VectorSearchSDKManager
from langchain_google_vertexai.vectorstores._searcher import (
    Searcher,
    VectorSearchSearcher,
)


class _BaseVertexAIVectorStore(VectorStore):
    """Represents a base vector store based on VertexAI."""

    def __init__(
        self,
        searcher: Searcher,
        document_storage: DocumentStorage,
        embbedings: Optional[Embeddings] = None,
    ) -> None:
        """Constructor.
        Args:
            searcher: Object in charge of searching and storing the index.
            document_storage: Object in charge of storing and retrieving documents.
            embbedings: Object in charge of transforming text to embbeddings.
        """
        super().__init__()
        self._searcher = searcher
        self._document_storage = document_storage
        self._embeddings = embbedings or self._get_default_embeddings()

    @property
    def embbedings(self) -> Embeddings:
        """Returns the embeddings object."""
        return self._embeddings

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        numeric_filter: Optional[List[NumericNamespace]] = None,
    ) -> List[Tuple[Document, float]]:
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
            Lower score represents more similarity.
        """

        embbedings = self._embeddings.embed_query(query)

        return self.similarity_search_by_vector_with_score(
            embedding=embbedings, k=k, filter=filter, numeric_filter=numeric_filter
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[List[Namespace]] = None,
        numeric_filter: Optional[List[NumericNamespace]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to the embedding and their cosine distance.
        Args:
            embedding: Embedding to look up documents similar to.
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
            Lower score represents more similarity.
        """

        neighbors_list = self._searcher.find_neighbors(
            embeddings=[embedding], k=k, filter_=filter, numeric_filter=numeric_filter
        )

        results = []

        for neighbor_id, distance in neighbors_list[0]:
            document = self._document_storage.get_by_id(neighbor_id)

            if document is None:
                raise ValueError(
                    f"Document with id {neighbor_id} not found in document" "storage."
                )

            results.append((document, distance))

        return results

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
        is_complete_overwrite: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        # Makes sure is a list an can get the length, should we support iterables?
        # metadata is a list so probably not?
        texts = list(texts)
        ids = self._generate_unique_ids(len(texts))

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(metadatas) != len(texts):
            raise ValueError(
                "`metadatas` should be the same length as `texts` "
                f"{len(metadatas)} != {len(texts)}"
            )

        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        self._document_storage.batch_store_by_id(ids=ids, documents=documents)

        embeddings = self._embeddings.embed_documents(texts)

        self._searcher.add_to_index(
            ids, embeddings, metadatas, is_complete_overwrite, **kwargs
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
                "`TensorflowHubEmbeddings` as a default embbedings is deprecated."
                " Will change to `VertexAIEmbbedings`. Please specify the embedding "
                "type in the constructor."
            ),
            category=DeprecationWarning,
        )

        # TODO: Change to vertexai embbedingss
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
            credentials_path: (Optional) The path of the Google credentials on
            the local file system.
            embedding: The :class:`Embeddings` that will be used for
            embedding the texts.
            stream_update: Whether to update with streaming or batching. VectorSearch
                index must be compatible with stream/batch updates.
            kwargs: Additional keyword arguments to pass to
                VertexAIVectorSearch.__init__().
        Returns:
            A configured VertexAIVectorSearch with the texts added to the index.
        """

        sdk_manager = VectorSearchSDKManager(
            project_id=project_id, region=region, credentials_path=credentials_path
        )
        bucket = sdk_manager.get_gcs_bucket(bucket_name=gcs_bucket_name)
        index = sdk_manager.get_index(index_id=index_id)
        endpoint = sdk_manager.get_endpoint(endpoint_id=endpoint_id)

        return cls(
            document_storage=GCSDocumentStorage(bucket=bucket),
            searcher=VectorSearchSearcher(
                endpoint=endpoint,
                index=index,
                staging_bucket=bucket,
                stream_update=stream_update,
            ),
            embbedings=embedding,
        )
