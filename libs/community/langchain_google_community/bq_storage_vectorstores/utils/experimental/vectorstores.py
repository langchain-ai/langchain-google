import pandas as pd
import copy

from typing import List, Dict, Optional, Any
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_google_community.bq_storage_vectorstores._base import (
    BaseBigQueryVectorStore,
)


class FakeEmbeddings(Embeddings, BaseModel):
    """Fake embedding model for temporary use in vector store synchronization."""

    texts_embedding_mapping: Dict = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using the pre-defined mapping.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[List[float]]: List of embeddings for the input texts.
        """
        return [self.texts_embedding_mapping[text] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text (not implemented for FakeEmbeddings).

        Args:
            text (str): Query text to embed.

        Raises:
            NotImplementedError: This method is not implemented for FakeEmbeddings.
        """
        raise NotImplementedError(
            "Query embedding is not implemented for FakeEmbeddings")


class BigQueryStorageBasedVectorStore(BaseBigQueryVectorStore):
    """
    A class that combines a VectorStore with BigQuery storage capabilities.

    This class wraps an existing VectorStore instance and adds BigQuery persistence.
    It maintains the full interface of the original VectorStore while providing
    additional methods for BigQuery integration.
    The class exposes all the VectorStore Interface search, get, delete methods from
    the original vector_store object. It provides instead customized `add_texts`,
    `add_texts_with_embeddings` and `add_documents` to ensure that operations affect
    both the in-memory vector store and the BigQuery storage.

    Attributes:
        vector_store (VectorStore): The original VectorStore instance being wrapped.
        vector_store_embedding_attribute (str): The attribute name for the embedding
            function in the vector store.
        embedding: Embedding model for generating and comparing embeddings.
        project_id: Google Cloud Project ID where BigQuery resources are located.
        dataset_name: BigQuery dataset name.
        table_name: BigQuery table name.
        location: BigQuery region/location.
        content_field: Name of the column storing document content (default: "content").
        embedding_field: Name of the column storing text embeddings (default:
            "embedding").
        doc_id_field: Name of the column storing document IDs (default: "doc_id").
        credentials: Optional Google Cloud credentials object.
        embedding_dimension: Dimension of the embedding vectors (inferred if not
            provided).

    Methods introduced:
        sync_data(): Synchronize data between BigQuery and the in-memory vector store.

    Example Usage:
        ```python
        from langchain_community.vectorstores import Chroma

        store = Chroma(embedding_function=embedding_model)
        patched_store = BigQueryStorageBasedVectorStore(
            vector_store=store,
            vector_store_embedding_attribute="_embedding_function",
            project_id=PROJECT_ID,
            location=LOCATION,
            dataset_name="test_bq",
            table_name="mytable",
        )

        # Use VectorStore methods (affects both in-memory and BigQuery storage)
        patched_store.similarity_search("query")
        patched_store.add_texts(["new document"])

        # Use BigQuery-specific methods
        patched_store.sync_data()  # Synchronize data between BigQuery and vector store

        # Access the original vector store if needed
        original_store = patched_store.vector_store
        ```
    """

    vector_store: VectorStore
    vector_store_embedding_attribute: str = "_embedding_function"
    embedding: Optional[Embeddings] = None

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=False, skip_on_failure=True)
    def validate_vals(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set up the vector store and its methods.
        Args:
            values (Dict[str, Any]): Dictionary of class attribute values.
        Returns:
            Dict[str, Any]: Updated dictionary of class attribute values.
        """
        values["_unpatched_vector_store"] = copy.copy(values["vector_store"])

        if values["embedding"] is None:
            values["embedding"] = getattr(values["vector_store"],
                                          values["vector_store_embedding_attribute"])

        # Override all methods from the vector_store
        for method_name in [
            "delete",
            "get_by_ids",
            "max_marginal_relevance_search",
            "max_marginal_relevance_search_by_vector",
            "search",
            "similarity_search",
            "similarity_search_by_vector",
            "similarity_search_with_relevance_scores",
            "similarity_search_with_score"
        ]:
            if hasattr(values["vector_store"], method_name):
                setattr(cls, method_name, getattr(values["vector_store"], method_name))

        return super().validate_vals(values=values)

    def get_documents(self, **kwargs: Any) -> Any:
        """
        Get documents from the vector store.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Documents from the vector store.
        """
        return self.vector_store.get_by_ids(**kwargs)

    def _similarity_search_by_vectors_with_scores_and_embeddings(
            self, **kwargs: Any
    ) -> List[List[List[Any]]]:
        """
        Perform similarity search by vectors with scores and embeddings (not implemented).

        Args:
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()

    def sync_data(self) -> None:
        """
        Synchronize data between BigQuery and the vector store.
        """
        self.vector_store = copy.copy(self._unpatched_vector_store)
        self._validate_bq_table()
        df = self._query_table_to_df()
        ids = df[self.doc_id_field].to_list()
        texts = df[self.content_field].to_list()
        texts_embedding_mapping = {text: list(emb) for text, emb in zip(texts, df[self.embedding_field])}
        metadatas = df[self.extra_fields.keys()].to_dict(orient='records')
        fake_embeddings = FakeEmbeddings(texts_embedding_mapping=texts_embedding_mapping)
        setattr(self.vector_store, self.vector_store_embedding_attribute, fake_embeddings)
        self.vector_store.add_texts(ids=ids, texts=texts, metadatas=metadatas)
        setattr(self.vector_store, self.vector_store_embedding_attribute, self.embedding)
        
        
    def _query_table_to_df(self) -> pd.DataFrame:
        """
        Query the BigQuery table and return the result as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the queried data.
        """
        table_id = f"{self.project_id}.{self.dataset_name}.{self.table_name}"
        table = self._bq_client.get_table(table_id)
        self._logger.info(f"Loading data from BigQuery from table {table_id}")
        df = self._bq_client.list_rows(table).to_dataframe()
        return df
