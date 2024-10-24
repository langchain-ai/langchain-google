import uuid
from datetime import datetime, timedelta
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

from google.api_core.exceptions import ClientError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import model_validator

if TYPE_CHECKING:
    from google.cloud.bigquery.table import Table

from typing_extensions import Self

from langchain_google_community.bq_storage_vectorstores._base import (
    BaseBigQueryVectorStore,
)

_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock

# Constants for index creation
MIN_INDEX_ROWS = 5000
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"


class BigQueryVectorStore(BaseBigQueryVectorStore):
    """
    A vector store implementation that utilizes BigQuery and BigQuery Vector Search.

    This class provides efficient storage and retrieval of documents with vector
    embeddings within BigQuery. It is particularly indicated for prototyping, due the
    serverless nature of BigQuery, and batch retrieval.
    It supports similarity search, filtering, and batch operations through
    `batch_search` method.
    Optionally, this class can leverage a Vertex AI Feature Store for online serving
    through the `to_vertex_fs_vector_store` method.

    Attributes:
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
        distance_type (Literal["COSINE", "EUCLIDEAN", "DOT_PRODUCT"]): The distance
            metric used for similarity search. Defaults to "EUCLIDEAN".
    """

    distance_type: Literal["COSINE", "EUCLIDEAN", "DOT_PRODUCT"] = "EUCLIDEAN"
    _creating_index: bool = False
    _have_index: bool = False
    _last_index_check: datetime = datetime.min

    def sync_data(self) -> None:
        pass

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs: Any,
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
        from google.cloud import bigquery  # type: ignore[attr-defined]

        if ids and len(ids) > 0:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"

        where_filter_expr = self._create_filters(filter)

        job = self._bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self.full_table_id}`
                    WHERE {id_expr} AND {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in row.keys():
                if field not in [
                    self.embedding_field,
                    self.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            docs.append(doc)
        return docs

    @model_validator(mode="after")
    def initialize_bq_vector_index(self) -> Self:
        """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """
        from google.cloud import bigquery  # type: ignore[attr-defined]

        self._creating_index = self._creating_index
        self._have_index = self._have_index
        self._last_index_check = self._last_index_check

        if self._have_index or self._creating_index:
            return self

        table = self._bq_client.get_table(self._full_table_id)  # type: ignore[union-attr]

        # Update existing table schema
        schema = table.schema.copy()
        if schema:  ## Check if table has a schema
            self.table_schema = {field.name: field.field_type for field in schema}

        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            self._logger.debug("Not enough rows to create a vector index.")
            return self

        if datetime.utcnow() - self._last_index_check < INDEX_CHECK_INTERVAL:
            return self

        with _vector_table_lock:
            self._last_index_check = datetime.utcnow()
            # Check if index exists, create if necessary
            check_query = (
                f"SELECT 1 FROM `{self.project_id}."
                f"{self.dataset_name}"
                ".INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE"
                f" table_name = '{self.table_name}'"
            )
            job = self._bq_client.query(  # type: ignore[union-attr]
                check_query, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            if job.result().total_rows == 0:
                # Need to create an index. Make it in a separate thread.
                self._logger.debug("Trying to create a vector index.")
                Thread(
                    target=_create_bq_index,
                    kwargs={
                        "bq_client": self._bq_client,
                        "table_name": self.table_name,
                        "full_table_id": self._full_table_id,
                        "embedding_field": self.embedding_field,
                        "distance_type": self.distance_type,
                        "logger": self._logger,
                    },
                    daemon=True,
                ).start()

            else:
                self._logger.debug("Vector index already exists.")
                self._have_index = True
            return self

    def _similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Union[Dict[str, Any], str]] = None,
        k: int = 5,
        batch_size: Union[int, None] = 100,
    ) -> List[List[List[Any]]]:
        """Performs a similarity search using vector embeddings

        This function takes a set of query embeddings and searches for similar documents
        It returns the top-k matching documents, along with their similarity scores
        and their corresponding embeddings.

        Args:
            embeddings: A list of lists, where each inner list represents a
                query embedding.
            filter: (Optional) A dictionary or a string specifying filter criteria.
                - If a dictionary is provided, it should map column names to their
                corresponding
                values. The method will generate SQL expressions based on the data
                types defined
                in `self.table_schema`:
                    - For columns of type "INTEGER" or "FLOAT", the value is used
                    directly.
                    - For other data types, the value is enclosed in single quotes.
                Example:
                        {
                            "str_property": "foo",
                            "int_property": 123
                        }
                - If a string is provided, it is assumed to be a valid SQL WHERE clause.
            k: The number of top results to return for each query.
            batch_size: The size of batches to process embeddings.

        Returns:
            A list of lists of lists. Each inner list represents the results for a
                single query, and contains elements of the form
                [Document, score, embedding], where:
                - Document: The matching document object.
                - score: The similarity score between the query and document.
                - embedding: The document's embedding.
        """

        search_results = []

        for start in range(0, len(embeddings), batch_size):  # type: ignore[arg-type]
            end = start + batch_size  # type: ignore[operator]
            embs_batch = embeddings[start:end]
            search_results.extend(
                self._search_embeddings(embeddings=embs_batch, filter=filter, k=k)
            )

        return self._create_langchain_documents(
            search_results=list(search_results),
            k=k,
            num_queries=len(embeddings),
            with_embeddings=True,
        )

    def _create_filters(
        self,
        filter: Optional[Union[Dict[str, Any], str]] = None,
    ) -> str:
        """Creates a SQL WHERE clause based on the provided filter criteria.

        This function generates a SQL WHERE clause from a given filter, which can either
        be a dictionary of column-value pairs or a pre-formatted SQL string.
        If no filter is provided, it returns a default clause that
        evaluates to TRUE.

        Args:
            filter: (Optional) A dictionary or a string specifying filter criteria.
                - If a dictionary is provided, it should map column names to
                their corresponding
                values. The method will generate SQL expressions based on the
                data types defined in `self.table_schema`:
                    - For columns of type "INTEGER" or "FLOAT", the value is
                    used directly.
                    - For other data types, the value is enclosed in single quotes.
                Example:
                    {
                        "str_property": "foo",
                        "int_property": 123
                    }
                - If a string is provided, it is assumed to be a valid SQL WHERE clause.

        Returns:
            A string representing the SQL WHERE clause. This clause can be directly
            used in SQL queries to filter results. If no filter is provided, it returns
            the string "TRUE" to indicate that no filtering should be applied.
        """
        if filter:
            # Pull BQ Vector Store information if not already done.
            if not self.table_schema:
                self._validate_bq_table()
            if isinstance(filter, Dict):  # If Dict filters is passed
                filter_expressions = []
                for column, value in filter.items():
                    if self.table_schema[column] in ["INTEGER", "FLOAT"]:  # type: ignore[index]
                        filter_expressions.append(f"{column} = {value}")
                    else:
                        filter_expressions.append(f"{column} = '{value}'")
                where_filter_expr = " AND ".join(filter_expressions)
            else:  # If SQL clauses filters is passed
                where_filter_expr = filter
        else:
            where_filter_expr = "TRUE"
        return where_filter_expr

    def _create_search_query(
        self,
        num_embeddings: int,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        k: int = 5,
        table_to_query: Any = None,
        fields_to_exclude: Optional[List[str]] = None,
    ) -> str:
        # Get where filter
        where_filter_expr = self._create_filters(filter)

        if table_to_query is not None:
            embeddings_query = f"""
            with embeddings as (
            SELECT {self.embedding_field}, ROW_NUMBER() OVER() as row_num
            from `{table_to_query}`
            )"""

        else:
            embeddings_query = "with embeddings as (\n"

            for i in range(num_embeddings):
                embeddings_query += (
                    f"SELECT {i} as row_num, @emb_{i} AS {self.embedding_field}"
                    if i == 0
                    else f"\nUNION ALL\n"
                    f"SELECT {i} as row_num, @emb_{i} AS {self.embedding_field}"
                )
            embeddings_query += "\n)\n"

        if fields_to_exclude is not None:
            select_clause = f"""SELECT
            base.* EXCEPT({','.join(fields_to_exclude)}),
            query.row_num,
            distance AS score
            """
        else:
            select_clause = """SELECT
            base.*,
            query.row_num,
            distance AS score
            """

        full_query = f"""{embeddings_query}
        {select_clause}
        FROM VECTOR_SEARCH(
            (SELECT * FROM `{self.full_table_id}` WHERE {where_filter_expr}),
            "{self.embedding_field}",
            (SELECT row_num, {self.embedding_field} FROM embeddings),
            distance_type => "{self.distance_type}",
            top_k => {k}
        )
        """
        # Wrap the Inner Query with an Outer SELECT to eliminate "base." column prefix
        full_query_wrapper = f"""
        SELECT *
        FROM (
            {full_query}
        ) AS result
        ORDER BY row_num, score
        """
        return full_query_wrapper

    def _search_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Union[Dict[str, Any], str]] = None,
        k: int = 5,
    ) -> list:
        from google.cloud import bigquery  # type: ignore[attr-defined]

        full_query = self._create_search_query(
            filter=filter, k=k, num_embeddings=len(embeddings)
        )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(f"emb_{i}", "FLOAT64", emb)
                for i, emb in enumerate(embeddings)
            ],
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )

        results = self._bq_client.query(  # type: ignore[union-attr]
            full_query,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )
        return list(results)

    def _create_temp_bq_table(
        self,
        embeddings: Optional[List[List[float]]],
        expire_hours_temp_table: int = 12,
    ) -> "Table":
        """Create temporary table to store query embeddings prior to batch search"""
        import pandas as pd
        from google.cloud import bigquery  # type: ignore[attr-defined]

        df = pd.DataFrame([])

        df[self.embedding_field] = embeddings
        table_id = (
            f"{self.project_id}."
            f"{self.temp_dataset_name}."
            f"{self.table_name}_{uuid.uuid4().hex}"
        )

        schema = [
            bigquery.SchemaField(self.embedding_field, "FLOAT64", mode="REPEATED")
        ]
        table_ref = bigquery.Table(table_id, schema=schema)
        table = self._bq_client.create_table(table_ref)
        table.expires = datetime.now() + timedelta(hours=expire_hours_temp_table)
        table = self._bq_client.update_table(table, ["expires"])
        job = self._bq_client.load_table_from_dataframe(df, table)
        job.result()
        return table_ref

    def _create_langchain_documents(
        self,
        search_results: List[List[Any]],
        k: int,
        num_queries: int,
        with_embeddings: bool = False,
    ) -> List[List[List[Any]]]:
        if len(search_results) == 0:
            return [[]]

        result_fields = list(search_results[0].keys())  # type: ignore[attr-defined]
        metadata_fields = [
            x
            for x in result_fields
            if x not in [self.embedding_field, self.content_field, "row_num"]
        ]
        documents = []
        for result in search_results:
            metadata = {
                metadata_field: result[metadata_field]
                for metadata_field in metadata_fields
            }
            document = Document(
                page_content=result[self.content_field],  # type: ignore
                metadata=metadata,
            )
            if with_embeddings:
                document_record = [
                    document,
                    metadata["score"],
                    result[self.embedding_field],  # type: ignore
                ]
            else:
                document_record = [document, metadata["score"]]
            documents.append(document_record)

        results_docs = [documents[i * k : (i + 1) * k] for i in range(num_queries)]
        return results_docs

    def batch_search(
        self,
        embeddings: Optional[List[List[float]]] = None,
        queries: Optional[List[str]] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        k: int = 5,
        expire_hours_temp_table: int = 12,
    ) -> List[List[List[Any]]]:
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
        from google.cloud import bigquery  # type: ignore[attr-defined]

        if not embeddings and not queries:
            raise ValueError(
                "At least one of 'embeddings' or 'queries' must be provided."
            )

        if embeddings is not None and queries is not None:
            raise ValueError(
                "Only one parameter between 'embeddings' or 'queries' must be provided."
            )

        if queries is not None:
            embeddings = self.embedding.embed_documents(queries)

        if embeddings is None:
            raise ValueError("Could not obtain embeddings - value is None.")

        table_ref = self._create_temp_bq_table(
            embeddings=embeddings, expire_hours_temp_table=expire_hours_temp_table
        )

        full_query = self._create_search_query(
            filter=filter,
            k=k,
            num_embeddings=len(embeddings),
            table_to_query=table_ref,
            fields_to_exclude=[self.embedding_field],
        )

        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )

        search_results = self._bq_client.query(  # type: ignore[union-attr]
            full_query,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )

        return self._create_langchain_documents(
            search_results=list(search_results),
            k=k,
            num_queries=len(embeddings),
            with_embeddings=False,
        )

    @classmethod
    def from_texts(
        cls: Type["BigQueryVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "BigQueryVectorStore":
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
        vs_obj = BigQueryVectorStore(embedding=embedding, **kwargs)
        vs_obj.add_texts(texts, metadatas)
        return vs_obj

    def to_vertex_fs_vector_store(self, **kwargs: Any) -> Any:
        """
        Creates and returns a VertexFSVectorStore instance based on configuration.

        This method merges the base BigQuery vector store configuration with provided
            keyword arguments,
        then uses the combined parameters to instantiate a VertexFSVectorStore.

        Args:
            **kwargs: Additional keyword arguments to override or extend the base
                configuration. These are directly passed to the VertexFSVectorStore
                constructor.

        Returns:
            VertexFSVectorStore: A fully initialized VertexFSVectorStore instance\
                ready for use.

        Raises:
            ImportError: If the required LangChain Google Community feature store
                module is not available.
        """
        from langchain_google_community.bq_storage_vectorstores.featurestore import (
            VertexFSVectorStore,
        )

        base_params = self.model_dump(
            include=set(BaseBigQueryVectorStore.model_fields.keys())
        )
        base_params["embedding"] = self.embedding
        all_params = {**base_params, **kwargs}
        fs_obj = VertexFSVectorStore(**all_params)
        return fs_obj

    def job_stats(self, job_id: str) -> Dict:
        """Return the statistics for a single job execution.

        Args:
            job_id: The BigQuery Job id.

        Returns:
            A dictionary of job statistics for a given job. You can check out more
            details at [BigQuery Jobs]
            (https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobStatistics2).
        """
        return self._bq_client.get_job(job_id)._properties["statistics"]


def _create_bq_index(
    bq_client: Any,
    table_name: str,
    full_table_id: str,
    embedding_field: str,
    distance_type: str,
    logger: Any,
) -> bool:
    """
    Create a BQ Vector Index if doesn't exist, if the number of rows is above
    MIN_INDEX_ROWS constant
    """
    table = bq_client.get_table(full_table_id)  # type: ignore[union-attr]
    if (table.num_rows or 0) < MIN_INDEX_ROWS:
        return False

    index_name = f"{table_name}_langchain_index"
    try:
        sql = f"""
            CREATE VECTOR INDEX IF NOT EXISTS
            `{index_name}`
            ON `{full_table_id}`
            ({embedding_field})
            OPTIONS(distance_type="{distance_type}", index_type="IVF")
        """
        bq_client.query(sql).result()  # type: ignore[union-attr]
        return True
    except ClientError as ex:
        logger.debug("Vector index creation failed (%s).", ex.args[0])
        return False
