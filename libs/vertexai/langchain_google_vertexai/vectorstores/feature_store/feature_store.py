"""Vector Store using Vertex AI Feature Store"""

from __future__ import annotations

import asyncio
import uuid
from functools import partial
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Type, Union

from google.cloud import bigquery
from google.cloud.aiplatform import base
from google.cloud.exceptions import NotFound
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, validate_call

from langchain_google_vertexai.vectorstores.feature_store.executors import (
    BigQueryExecutor,
    BruteForceExecutor,
    FeatureOnlineStoreExecutor,
)
from langchain_google_vertexai.vectorstores.feature_store.utils import (
    EnvConfig,
    validate_column_in_bq_schema,
)

logger = base.Logger(__name__)


class FeatureStore(VectorStore, BaseModel):
    """Google Cloud Feature Store vector store.
    The FeatureStore aims to facilitate similarity search using different
        methodologies on Google Cloud including Big Query, Feature Store and a
        local bruteforce search engine.
    Big Query is the data source of truth and also the default search
        methodology (or executor).
    When lower latency is required, it is possible to move to the feature store
        executor with one line: my_fs.set_executor({"type": "feature_online_store"}).
    The data can be synced from BQ to FS using my_fs.sync().
    Optionally a cron schedule can be passed for automatic sync of BQ data
    to fs: my_fs.set_executor({
        "type": "feature_online_store", "cron_schedule": "TZ=Europe/Rome 00 00 01 5 *"
        })

    Attributes:
        embedding (Any): An embedding model instance for text to vector transformations.
        project_id (str): Your Google Cloud Project ID.
        dataset_name (str): Name of the dataset within BigQuery.
        table_name (str): Name of the table within the dataset.
        location (str): Location of your BigQuery dataset (e.g., "europe-west2").
        executor (Union[BigQueryExecutor, BruteForceExecutor,
            FeatureOnlineStoreExecutor]): The executor to use for search
            (defaults to BigQueryExecutor).
        content_field (str): The field name in the Feature Store that stores the
            text content.
        text_embedding_field (str): The field name in the Feature Store that stores
            the text embeddings.
        doc_id_field (str): The field name in the Feature Store that stores the
            document IDs.
        credentials (Optional[Any]): Optional credentials for Google Cloud
            authentication.

    To use, you need the following packages installed:
        google-cloud-bigquery
    """

    embedding: Any
    project_id: str
    dataset_name: str
    table_name: str
    location: str
    executor: Union[
        BigQueryExecutor, BruteForceExecutor, FeatureOnlineStoreExecutor
    ] = BigQueryExecutor()
    content_field: str = "content"
    text_embedding_field: str = "text_embedding"
    doc_id_field: str = "doc_id"
    credentials: Optional[Any] = None
    _extra_fields: Union[Dict[str, str], None] = None
    _env_config: Optional[EnvConfig] = None
    _table_schema: Any = None

    def model_post_init(self, __context):
        """Constructor for FeatureStore."""
        try:
            import pandas as pd  # type: ignore[import-untyped]

            find_spec("pyarrow")
            find_spec("db_types")
            find_spec("langchain_community")
            self._pd = pd
        except ModuleNotFoundError as e:
            logger.error(e)
            raise ImportError(
                "Please, install feature store dependency group: "
                "`pip install langchain-google-vertexai[featurestore]`"
            )
        self._bq_client = bigquery.Client(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )
        self._embedding_dimension = len(self.embedding.embed_query("test"))
        self._full_table_id = (
            f"{self.project_id}." f"{self.dataset_name}." f"{self.table_name}"
        )
        self._initialize_bq_table()
        self._validate_bq_table()
        logger.info(
            f"BigQuery table {self._full_table_id} "
            f"initialized/validated as persistent storage. "
            f"Access via BigQuery console:\n "
            f"https://console.cloud.google.com/bigquery?project={self.project_id}"
            f"&ws=!1m5!1m4!4m3!1s{self.project_id}!2s{self.dataset_name}!3s{self.table_name}"
        )
        self._env_config = EnvConfig(
            bq_client=self._bq_client,
            project_id=self.project_id,
            dataset_name=self.dataset_name,
            table_name=self.table_name,
            location=self.location,
            extra_fields=self._extra_fields,
            content_field=self.content_field,
            table_schema=self._table_schema,
            text_embedding_field=self.text_embedding_field,
            doc_id_field=self.doc_id_field,
            full_table_id=self._full_table_id,
            embedding_dimension=self._embedding_dimension,
        )
        self.executor.set_env_config(env_config=self._env_config)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @property
    def full_table_id(self) -> str:
        return self._full_table_id

    def _validate_bq_table(self):
        table_ref = bigquery.TableReference.from_string(self._full_table_id)

        try:
            table = self._bq_client.get_table(
                self.full_table_id
            )  # Attempt to retrieve the table information
        except NotFound:
            logger.debug(
                f"Couldn't find table {self.full_table_id}. "
                f"Table will be created once documents are added"
            )
            return

        table = self._bq_client.get_table(table_ref)
        schema = table.schema.copy()
        if schema:  ## Check if table has a schema
            self._table_schema = {field.name: field.field_type for field in schema}
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
                column_name=self.text_embedding_field,
                columns=columns,
                expected_types=["FLOAT", "FLOAT64"],
                expected_modes=["REPEATED"],
            )
            if self._extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [
                        self.doc_id_field,
                        self.content_field,
                        self.text_embedding_field,
                    ]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. "
                                f"REPEATED fields are not supported in this context."
                            )
                        extra_fields[column.name] = column.field_type
                self._extra_fields = extra_fields
                if self._env_config:
                    self._env_config.extra_fields = extra_fields
                    self._env_config.table_schema = self._table_schema
            else:
                for field, type in self._extra_fields.items():
                    validate_column_in_bq_schema(
                        column_name=field,
                        columns=columns,
                        expected_types=[type],
                        expected_modes=["NULLABLE", "REQUIRED"],
                    )
            logger.debug(f"Table {self.full_table_id} validated")
        return table_ref

    def _initialize_bq_table(self) -> Any:
        """Validates or creates the BigQuery table."""
        self._bq_client.create_dataset(dataset=self.dataset_name, exists_ok=True)
        table_ref = bigquery.TableReference.from_string(self._full_table_id)
        self._bq_client.create_table(table_ref, exists_ok=True)
        return table_ref

    def sync(self):
        self.executor.sync()

    def add_texts(  # type: ignore[override]
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])

        Returns:
            List of ids from adding the texts into the vectorstore.
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
        **kwargs: Any,
    ) -> List[str]:
        """Add precomputed embeddings and relative texts / metadatas to the vectorstore.

        Args:
            ids: List of unique ids in string format
            texts: List of strings to add to the vectorstore.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: List[Dict[str, List[Any]]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.content_field: text,
                self.text_embedding_field: emb,
            }
            record.update(metadata_dict)
            values_dict.append(record)  # type: ignore[arg-type]

        table = self._bq_client.get_table(
            self.full_table_id
        )  # Attempt to retrieve the table information
        df = self._pd.DataFrame(values_dict)
        job = self._bq_client.load_table_from_dataframe(df, table)
        job.result()
        self._validate_bq_table()
        logger.debug(f"stored {len(ids)} records in BQ")
        self.sync()
        return ids

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        return self.executor.get_documents(ids=ids, filter=filter, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by record IDs

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
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
        self.sync()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
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
    ) -> List[List[Document]]:
        """Core similarity search function. Handles a list of embedding vectors,
            optionally returning scores and embeddings.

        Args:
            embeddings: A list of embedding vectors, where each vector is a list of
                floats.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents.
                Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
            with_scores: (Optional) If True, include similarity scores in the result
                for each matched document. Defaults to False.
            with_embeddings: (Optional) If True, include the matched document's
                embedding vector in the result. Defaults to False.
        Returns:
            A list of `k` documents for each embedding in `embeddings`
        """
        results = self.executor.similarity_search_by_vectors_with_scores_and_embeddings(
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
        **kwargs,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
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
    ):
        """Return docs most similar to embedding vector with scores.

        Args:
            embedding: Embedding to look up documents similar to.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to embedding vector.
        """
        return self.similarity_search_by_vectors(
            embeddings=[embedding], filter=filter, k=k, with_scores=True
        )[0]

    def similarity_search(self, query: str, k: int = 5, **kwargs):
        """Search for top `k` docs most similar to input query.

        Args:
            query: search query to search documents with.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to input query.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k, **kwargs)[
            0
        ]

    def similarity_search_with_score(
        self, query: str, filter: Optional[Dict[str, Any]] = None, k: int = 5, **kwargs
    ):
        """Search for top `k` docs most similar to input query, returns both docs and
            scores.

        Args:
            query: search query to search documents with.
            filter: (Optional) A dictionary specifying filtering criteria for the
                documents. Ie. {"title": "mytitle"}
            k: (Optional) The number of top-ranking similar documents to return per
                embedding. Defaults to 5.
        Returns:
            Return docs most similar to input query along with scores.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            embedding=embedding, filter=filter, k=k, **kwargs
        )

    def batch_search(
        self,
        embeddings: Optional[List[List[float]]] = None,
        queries: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        with_scores: bool = False,
        with_embeddings: bool = False,
        **kwargs,
    ):
        """Multi-purpose batch search function. Accepts either embeddings or queries
            but not both. Optionally returns similarity scores and/or matched embeddings
        Args:
        embeddings: A list of embeddings to search with. If provided, each
            embedding represents a query vector.
        queries: A list of text queries to search with.  If provided, each
            query represents a query text.
        filter: A dictionary of filters to apply to the search. The keys
            of the dictionary should be field names, and the values should be the
                values to filter on. (e.g., {"category": "news"})
        k: The number of top results to return per query. Defaults to 5.
        with_scores: If True, returns the relevance scores of the results along with
            the documents
        with_embeddings: If True, returns the embeddings of the results along with
            the documents
        """
        if not embeddings and not queries:
            raise ValueError(
                "At least one of 'embeddings' or 'queries' must be provided."
            )

        if embeddings is not None and queries is not None:
            raise ValueError(
                "Only one parameter between 'embeddings' or 'queries' must be provided"
            )

        if queries is not None:
            embeddings = self.embedding.embed_documents(queries)

        if embeddings is not None:
            return self.similarity_search_by_vectors(
                embeddings=embeddings,
                filter=filter,
                k=k,
                with_scores=with_scores,
                with_embeddings=with_embeddings,
                **kwargs,
            )

    @classmethod
    def from_texts(
        cls: Type["FeatureStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "FeatureStore":
        """Return VectorStore initialized from input texts

        Args:
            texts: List of strings to add to the vectorstore.
            embedding: An embedding model instance for text to vector transformations.
            metadatas: Optional list of metadata records associated with the texts.
                (ie [{"url": "www.myurl1.com", "title": "title1"},
                {"url": "www.myurl2.com", "title": "title2"}])
        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        vs_obj = FeatureStore(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj

    @validate_call
    def set_executor(
        self,
        executor: Union[
            BigQueryExecutor, FeatureOnlineStoreExecutor, BruteForceExecutor
        ],
    ):
        """Set a different executor to run similarity search.

        Args:
            executor: Any of [BigQueryExecutor, FeatureOnlineStoreExecutor,
                BruteForceExecutor]
                example usage:
                    1. my_fs.set_executor({"type": "big_query"})
                    2. my_fs.set_executor({"type": "feature_online_store",
                        "cron_schedule": "TZ=Europe/Rome 00 00 01 5 *"})
                    3. my_fs.set_executor({"type": "brute_force"})
        Returns:
            None
        """
        self.executor = executor
        if self._env_config is not None:
            self.executor.set_env_config(env_config=self._env_config)
