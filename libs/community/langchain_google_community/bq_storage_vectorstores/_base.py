from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import partial
from importlib.util import find_spec
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from langchain_google_community._utils import get_client_info
from langchain_google_community.bq_storage_vectorstores.utils import (
    check_bq_dataset_exists,
    validate_column_in_bq_schema,
)

_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock

# Constants for index creation
MIN_INDEX_ROWS = 5
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"


class BaseBigQueryVectorStore(VectorStore, BaseModel, ABC):
    """Abstract base class for BigQuery-based vector stores.

    This class provides a foundation for storing, retrieving, and searching documents
    and their corresponding embeddings in BigQuery.

    Abstract Methods:
        sync_data: Synchronizes data between the vector store and BigQuery.
        get_documents: Retrieves documents based on IDs or filters.
        _similarity_search_by_vectors_with_scores_and_embeddings: Performs
            similarity search with scores and embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: Embeddings
    """Embedding model for generating and comparing embeddings."""

    project_id: str
    """Google Cloud Project ID where BigQuery resources are located."""

    dataset_name: str
    """BigQuery dataset name."""

    table_name: str
    """BigQuery table name."""

    location: str
    """BigQuery region/location."""

    content_field: str = "content"
    """Name of the column storing document content."""

    embedding_field: str = "embedding"
    """Name of the column storing text embeddings."""

    doc_id_field: str = "doc_id"
    """Name of the column storing document IDs."""

    temp_dataset_name: Optional[str] = None
    """Name of the BigQuery dataset to be used to upload temporary BQ tables.
    
    If `None`, will default to `'{dataset_name}_temp'`.
    """

    credentials: Optional[Any] = None
    """Optional Google Cloud credentials object."""

    embedding_dimension: Optional[int] = None
    """Dimension of the embedding vectors (inferred if not provided)."""

    extra_fields: Union[Dict[str, str], None] = None

    table_schema: Any = None

    _bq_client: Any = None

    _logger: Any = None

    _full_table_id: Optional[str] = None

    @abstractmethod
    def sync_data(self) -> None: ...

    @abstractmethod
    def get_documents(
        self,
        ids: Optional[List[str]],
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search documents by their IDs or metadata values.

        Args:
            ids: List of IDs of documents to retrieve from the `VectorStore`.
            filter: Filter on metadata properties, e.g.
                ```json
                {
                    "str_property": "foo",
                    "int_property": 123
                }
                ```
        Returns:
            List of IDs from adding the texts into the `VectorStore`.
        """
        ...

    @abstractmethod
    def _similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
        **kwargs: Any,
    ) -> List[List[List[Any]]]: ...

    @model_validator(mode="after")
    def validate_vals(self) -> Self:
        try:
            import pandas  # noqa: F401
            from google.cloud import bigquery  # type: ignore[attr-defined]
            from google.cloud.aiplatform import base

            find_spec("pyarrow")
            find_spec("db_types")
        except ModuleNotFoundError:
            raise ImportError(
                "Please, install feature store dependency group: "
                "`pip install langchain-google-community[featurestore]`"
            )
        self._logger = base.Logger(__name__)
        self._bq_client = bigquery.Client(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
            client_info=get_client_info(module="bigquery-vector-search"),
        )
        if self.embedding_dimension is None:
            self.embedding_dimension = len(self.embedding.embed_query("test"))
        if self.temp_dataset_name is None:
            self.temp_dataset_name = f"{self.dataset_name}_temp"
        full_table_id = f"{self.project_id}.{self.dataset_name}.{self.table_name}"
        self._full_table_id = full_table_id
        if not check_bq_dataset_exists(
            client=self._bq_client, dataset_id=self.dataset_name
        ):
            self._bq_client.create_dataset(dataset=self.dataset_name, exists_ok=True)
        if not check_bq_dataset_exists(
            client=self._bq_client, dataset_id=self.temp_dataset_name
        ):
            self._bq_client.create_dataset(
                dataset=self.temp_dataset_name, exists_ok=True
            )
        table_ref = bigquery.TableReference.from_string(full_table_id)
        self._bq_client.create_table(table_ref, exists_ok=True)
        self._logger.info(
            f"BigQuery table {full_table_id} "
            f"initialized/validated as persistent storage. "
            f"Access via BigQuery console:\n "
            f"https://console.cloud.google.com/bigquery?project={self.project_id}"
            f"&ws=!1m5!1m4!4m3!1s{self.project_id}!2s{self.dataset_name}!3s"
            f"{self.table_name}"
        )
        return self

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @property
    def full_table_id(self) -> str:
        return cast(str, self._full_table_id)

    def _validate_bq_table(self) -> Any:
        from google.cloud import bigquery  # type: ignore[attr-defined]
        from google.cloud.exceptions import NotFound

        table_ref = bigquery.TableReference.from_string(self.full_table_id)

        try:
            # Attempt to retrieve the table information
            self._bq_client.get_table(self.full_table_id)
        except NotFound:
            self._logger.debug(
                f"Couldn't find table {self.full_table_id}. "
                f"Table will be created once documents are added"
            )
            return

        table = self._bq_client.get_table(table_ref)
        schema = table.schema.copy()
        if schema:  ## Check if table has a schema
            self.table_schema = {field.name: field.field_type for field in schema}
            columns = {c.name: c for c in schema}
            validate_column_in_bq_schema(
                column_name=self.doc_id_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )
            validate_column_in_bq_schema(
                column_name=self.content_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )
            validate_column_in_bq_schema(
                column_name=self.embedding_field,
                columns=columns,
                expected_types=["FLOAT", "FLOAT64"],
                expected_modes=["REPEATED"],
            )
            if self.extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [
                        self.doc_id_field,
                        self.content_field,
                        self.embedding_field,
                    ]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. "
                                f"REPEATED fields are not supported in this context."
                            )
                        extra_fields[column.name] = column.field_type
                self.extra_fields = extra_fields
            else:
                for field, type in self.extra_fields.items():
                    validate_column_in_bq_schema(
                        column_name=field,
                        columns=columns,
                        expected_types=[type],
                        expected_modes=["NULLABLE", "REQUIRED"],
                    )
            self._logger.debug(f"Table {self.full_table_id} validated")
        return table_ref

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the `VectorStore`.

        Args:
            texts: List of strings to add to the `VectorStore`.
            metadatas: Optional list of metadata records associated with the texts.

                (i.e. `[{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}]`)

        Returns:
            List of IDs from adding the texts into the `VectorStore`.
        """
        embs = self.embedding.embed_documents(texts)
        return self.add_texts_with_embeddings(
            texts=texts, embs=embs, metadatas=metadatas, **kwargs
        )

    def add_texts_with_embeddings(
        self,
        texts: List[str],
        embs: List[List[float]],
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Add precomputed embeddings & relative texts / metadatas to the `VectorStore`.

        Args:
            ids: List of unique IDs in string format
            texts: List of strings to add to the `VectorStore`.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata records associated with the texts.

                (i.e. `[{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}]`)
        Returns:
            List of IDs from adding the texts into the `VectorStore`.
        """
        import pandas as pd

        ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: List[Dict[str, List[Any]]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.content_field: text,
                self.embedding_field: emb,
            }
            record.update(metadata_dict)
            values_dict.append(record)  # type: ignore[arg-type]

        table = self._bq_client.get_table(
            self.full_table_id
        )  # Attempt to retrieve the table information
        df = pd.DataFrame(values_dict)
        job = self._bq_client.load_table_from_dataframe(df, table)
        job.result()
        self._validate_bq_table()
        self._logger.debug(f"stored {len(ids)} records in BQ")
        self.sync_data()
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by record IDs

        Args:
            ids: List of IDs to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        from google.cloud import bigquery  # type: ignore[attr-defined]

        if not ids or len(ids) == 0:
            return True

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)],
        )
        self._bq_client.query(
            f"""
                    DELETE FROM `{self.full_table_id}` WHERE {self.doc_id_field}
                    IN UNNEST(@ids)
                    """,
            job_config=job_config,
        ).result()
        self.sync_data()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of IDs to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def similarity_search_by_vectors(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        with_scores: bool = False,
        with_embeddings: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Core similarity search function.

        Handles a list of embedding vectors, optionally returning scores and embeddings.

        Args:
            embeddings: List of embedding vectors, where each vector is a list of
                floats.
            filter: Dictionary specifying filtering criteria for the documents.

                i.e. `{"title": "mytitle"}`
            k: Number of top-ranking similar documents to return per embedding.
            with_scores: If `True`, include similarity scores in the result for each
                matched document.
            with_embeddings: If `True`, include the matched document's embedding vector
                in the result.
        Returns:
            A list of `k` documents for each embedding in `embeddings`
        """
        results = self._similarity_search_by_vectors_with_scores_and_embeddings(
            embeddings=embeddings, k=k, filter=filter, **kwargs
        )

        # Process results based on options
        for i, query_results in enumerate(results):
            if not with_scores and not with_embeddings:
                # return only docs
                results[i] = [x[0] for x in query_results]
            elif not with_embeddings:
                # return only docs and score
                results[i] = [[x[0], x[1]] for x in query_results]
            elif not with_scores:
                # return only docs and embeddings
                results[i] = [[x[0], x[2]] for x in query_results]

        return results  # type: ignore[return-value]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: Dictionary specifying filtering criteria for the documents.

                i.e. `{"title": "mytitle"}`
            k: Number of top-ranking similar documents to return per embedding.

        Returns:
            Return docs most similar to embedding vector.
        """
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k, **kwargs)[
            0
        ]

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: Dictionary specifying filtering criteria for the documents.

                i.e. `{"title": "mytitle"}`
            k: The number of top-ranking similar documents to return per embedding.

        Returns:
            Return docs most similar to embedding vector.
        """
        return self.similarity_search_by_vectors(
            embeddings=[embedding], filter=filter, k=k, with_scores=True
        )[0]

    def similarity_search(
        self, query: str, k: int = 5, **kwargs: Any
    ) -> List[Document]:
        """Search for top `k` docs most similar to input query.

        Args:
            query: Search query to search documents with.
            filter: Dictionary specifying filtering criteria for the documents.

                i.e. `{"title": "mytitle"}`
            k: The number of top-ranking similar documents to return per embedding.

        Returns:
            Return docs most similar to input query.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k, **kwargs)[
            0
        ]

    def similarity_search_with_score(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for top `k` docs most similar to input query, returns both docs and
        scores.

        Args:
            query: Search query to search documents with.
            filter: Dictionary specifying filtering criteria for the documents.

                i.e. `{"title": "mytitle"}`
            k: The number of top-ranking similar documents to return per embedding.

        Returns:
            Return docs most similar to input query along with scores.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding=embedding, filter=filter, k=k
        )

    @classmethod
    def from_texts(
        cls: Type["BaseBigQueryVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BaseBigQueryVectorStore":
        raise NotImplementedError()

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 25,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            **kwargs:
            query: Search query text.
            filter: Filter on metadata properties, e.g.

                ```json
                {
                    "str_property": "foo",
                    "int_property": 123
                }
                ```
            k: Number of documents to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree of
                diversity among the results with 0 corresponding to maximum diversity
                and `1` to minimum diversity.

        Returns:
            List of documents selected by maximal marginal relevance.
        """
        embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        fetch_k: int = 25,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: Filter on metadata properties, e.g.

                ```json
                {
                    "str_property": "foo",
                    "int_property": 123
                }
                ```
            k: Number of documents to return.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            lambda_mult: Number between `0` and `1` that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and `1` to minimum diversity.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        doc_tuples = self.similarity_search_by_vectors(
            embeddings=[embedding],
            k=fetch_k,
            with_embeddings=True,
            with_scores=True,
            **kwargs,
        )[0]
        doc_embeddings = [d[2] for d in doc_tuples]  # type: ignore[index]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]  # type: ignore[index]
