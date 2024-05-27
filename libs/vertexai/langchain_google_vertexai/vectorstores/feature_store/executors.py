from __future__ import annotations

import time
from datetime import datetime, timedelta
from subprocess import TimeoutExpired
from threading import Lock, Thread
from typing import Any, Dict, List, Literal, MutableSequence, Optional, Union

import numpy as np
import proto  # type: ignore[import-untyped]
import vertexai  # type: ignore[import-untyped]
from google.api_core.exceptions import (
    ClientError,
    MethodNotImplemented,
    ServiceUnavailable,
)
from google.cloud import bigquery
from google.cloud.aiplatform import base, telemetry
from google.cloud.aiplatform_v1beta1 import (
    FeatureOnlineStoreAdminServiceClient,
    FeatureOnlineStoreServiceClient,
)
from google.cloud.aiplatform_v1beta1.types import (
    NearestNeighborQuery,
    feature_online_store_service,
)
from google.cloud.aiplatform_v1beta1.types import (
    feature_online_store as feature_online_store_pb2,
)
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict
from vertexai.resources.preview import (  # type: ignore[import-untyped]
    AlgorithmConfig,
    DistanceMeasureType,
    FeatureOnlineStore,
    FeatureView,
    FeatureViewBigQuerySource,
)
from vertexai.resources.preview.feature_store import (  # type: ignore[import-untyped]
    utils,
)

from langchain_google_vertexai._utils import get_client_info, get_user_agent
from langchain_google_vertexai.vectorstores.feature_store.utils import (
    EnvConfig,
    cast_proto_type,
    doc_match_filter,
)

_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock

logger = base.Logger(__name__)
# Constants for index creation
MIN_INDEX_ROWS = 5
INDEX_CHECK_INTERVAL = timedelta(seconds=60)
USER_AGENT_PREFIX = "FeatureStore"


class BaseExecutor(BaseModel):
    _env_config: EnvConfig = EnvConfig()
    _user_agent: str = get_user_agent(None)[1]

    def sync(self):
        raise NotImplementedError()

    def similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
    ) -> list[list[list[Any]]]:
        raise NotImplementedError()

    def set_env_config(self, env_config: EnvConfig):
        self._env_config = env_config


class BruteForceExecutor(BaseExecutor):
    type: Literal["brute_force"] = "brute_force"
    _df: Any = None
    _vectors: Any = None
    _vectors_transpose: Any = None
    _df_records: Any = None
    _env_config: EnvConfig = EnvConfig()

    def sync(self):
        """Sync the data from the Big Query source into the Executor source"""
        self._df = self._query_table_to_df()
        self._vectors = np.array(
            self._df[self._env_config.text_embedding_field].tolist()
        )
        self._vectors_transpose = self._vectors.T
        self._df_records = self._df.drop(
            columns=[self._env_config.text_embedding_field]
        ).to_dict("records")

    def similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
    ) -> list[list[list[Any]]]:
        """Performs a similarity search using vector embeddings

        This function takes a set of query embeddings and searches for similar documents
        It returns the top-k matching documents, along with their similarity scores
        and their corresponding embeddings.

        Args:
            embeddings: A list of lists, where each inner list represents a
                query embedding.
            filter: (Optional) A dictionary specifying filter criteria for document
                on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
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

        if self._df is None:
            raise ValueError(
                "Brute force executor was correctly initialized but not "
                "synced yet. Please run FeatureStore - sync() method"
                " sync the index in memory"
            )
        scores = embeddings @ self._vectors_transpose
        sorted_indices = np.argsort(-scores)[:, :k]

        results = [np.array(self._df_records)[x] for x in sorted_indices]
        top_scores = scores[np.arange(len(embeddings))[:, np.newaxis], sorted_indices]
        top_embeddings = self._vectors[sorted_indices]

        documents = []
        for query_results, query_scores, embeddings_results in zip(
            results, top_scores, top_embeddings
        ):
            query_docs = []
            for doc, doc_score, embedding in zip(
                query_results, query_scores, embeddings_results
            ):
                if filter is not None and not doc_match_filter(
                    document=doc, filter=filter
                ):
                    continue
                query_docs.append(
                    [
                        Document(
                            page_content=doc[self._env_config.content_field],
                            metadata=doc,
                        ),
                        doc_score,
                        embedding,
                    ]
                )

            documents.append(query_docs)
        return documents

    def get_documents(
        self,
        ids: Optional[List[str]],
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

        if self._df is None:
            raise ValueError(
                "Brute force executor was correctly initialized but not "
                "synced yet. Please run FeatureStore - sync() method"
                " sync the index in memory"
            )
        output = []
        df = self._df
        if ids:
            results = df.loc[df[self._env_config.doc_id_field].isin(ids)]
        else:
            results = df
        for i, row in results.iterrows():
            metadata = {}
            for field in row.keys():
                if field not in [
                    self._env_config.text_embedding_field,
                    self._env_config.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self._env_config.doc_id_field]
            if filter is not None and not doc_match_filter(
                document=metadata, filter=filter
            ):
                continue
            doc = Document(
                page_content=row[self._env_config.content_field], metadata=metadata
            )
            output.append(doc)
        return output

    def _query_table_to_df(self):
        client = self._env_config.bq_client
        extra_fields = self._env_config.extra_fields
        if extra_fields is None:
            extra_fields = {}
        metadata_fields = list(extra_fields.keys())
        metadata_fields_str = ", ".join(metadata_fields)

        table = (
            f"{self._env_config.project_id}.{self._env_config.dataset_name}"
            f".{self._env_config.table_name}"
        )
        fields = (
            f"{self._env_config.doc_id_field}, {self._env_config.content_field}, "
            f"{self._env_config.text_embedding_field}, {metadata_fields_str}"
        )
        query = f"""
        SELECT {fields}
        FROM {table}
        """
        # Create a query job to read the data
        logger.info(f"Reading data from {table}. It might take a few minutes...")
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True, priority=bigquery.QueryPriority.INTERACTIVE
        )
        query_job = client.query(query, job_config=job_config)  # type: ignore[union-attr]
        return query_job.to_dataframe()


class BigQueryExecutor(BaseExecutor):
    type: Literal["bigquery"] = "bigquery"
    distance_type: Literal["COSINE", "EUCLIDEAN"] = "EUCLIDEAN"
    _creating_index: bool = False
    _have_index: bool = False
    _last_index_check: datetime = datetime.min
    _env_config: EnvConfig = EnvConfig()

    def model_post_init(self, __context: Any) -> None:
        # Initialize attributes after model creation
        self._creating_index = False
        self._have_index = False
        self._last_index_check = datetime.min

    def sync(self):
        """Sync the data from the Big Query source into the Executor source"""
        self._initialize_bq_vector_index()

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
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

        if ids and len(ids) > 0:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self._env_config.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"
        if filter:
            filter_expressions = []
            for column, value in filter.items():
                filter_expressions.append(f"{column} = '{value}'")
            filter_expression_str = " AND ".join(filter_expressions)
            where_filter_expr = f" AND ({filter_expression_str})"
        else:
            where_filter_expr = ""

        job = self._env_config.bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self._env_config.full_table_id}` WHERE {id_expr}
                    {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in row.keys():
                if field not in [
                    self._env_config.text_embedding_field,
                    self._env_config.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self._env_config.doc_id_field]
            doc = Document(
                page_content=row[self._env_config.content_field], metadata=metadata
            )
            docs.append(doc)
        return docs

    def _initialize_bq_vector_index(self) -> Any:
        """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """
        if self._have_index or self._creating_index:
            return

        table = self._env_config.bq_client.get_table(self._env_config.full_table_id)  # type: ignore[union-attr]
        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            logger.debug("Not enough rows to create a vector index.")
            return

        if datetime.utcnow() - self._last_index_check < INDEX_CHECK_INTERVAL:
            return

        with _vector_table_lock:
            self._last_index_check = datetime.utcnow()
            # Check if index exists, create if necessary
            check_query = (
                f"SELECT 1 FROM `{self._env_config.project_id}."
                f"{self._env_config.dataset_name}"
                ".INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE"
                f" table_name = '{self._env_config.table_name}'"
            )
            job = self._env_config.bq_client.query(  # type: ignore[union-attr]
                check_query, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            if job.result().total_rows == 0:
                # Need to create an index. Make it in a separate thread.
                self._create_bq_index_in_background()
            else:
                logger.debug("Vector index already exists.")
                self._have_index = True

    def _create_bq_index_in_background(self):
        if self._have_index or self._creating_index:
            return

        self._creating_index = True
        logger.debug("Trying to create a vector index.")
        Thread(target=self._create_bq_index, daemon=True).start()

    def _create_bq_index(self):
        table = self._env_config.bq_client.get_table(self._env_config.full_table_id)  # type: ignore[union-attr]
        if (table.num_rows or 0) < MIN_INDEX_ROWS:
            return

        index_name = f"{self._env_config.table_name}_langchain_index"
        try:
            sql = f"""
                CREATE VECTOR INDEX IF NOT EXISTS
                `{index_name}`
                ON `{self._env_config.full_table_id}`
                ({self._env_config.text_embedding_field})
                OPTIONS(distance_type="{self.distance_type}", index_type="IVF")
            """
            self._env_config.bq_client.query(sql).result()  # type: ignore[union-attr]
            self._have_index = True
        except ClientError as ex:
            logger.debug("Vector index creation failed (%s).", ex.args[0])
        finally:
            self._creating_index = False

    def similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = 100,
    ) -> list[list[list[Any]]]:
        """Performs a similarity search using vector embeddings

        This function takes a set of query embeddings and searches for similar documents
        It returns the top-k matching documents, along with their similarity scores
        and their corresponding embeddings.

        Args:
            embeddings: A list of lists, where each inner list represents a
                query embedding.
            filter: (Optional) A dictionary specifying filter criteria for document
                on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
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

        final_results = []

        for start in range(0, len(embeddings), batch_size):  # type: ignore[arg-type]
            end = start + batch_size  # type: ignore[operator]
            embs_batch = embeddings[start:end]
            final_results.extend(
                self._search_embeddings(embeddings=embs_batch, filter=filter, k=k)
            )
        if len(final_results) == 0:
            return [[]]
        documents = []
        fields = [
            x
            for x in final_results[0].keys()
            if x
            not in [
                self._env_config.text_embedding_field,
                self._env_config.content_field,
            ]
        ]
        for result in final_results:
            metadata = {}
            for field in fields:
                metadata[field] = result[field]
            documents.append(
                [
                    Document(
                        page_content=result[self._env_config.content_field],
                        metadata=metadata,
                    ),
                    metadata["score"],
                    result[self._env_config.text_embedding_field],
                ]
            )
        results_chunks = [
            documents[i * k : (i + 1) * k] for i in range(len(embeddings))
        ]
        return results_chunks

    def _search_embeddings(
        self, embeddings, filter: Optional[Dict[str, Any]] = None, k=5
    ):
        if filter:
            filter_expressions = []
            for column, value in filter.items():
                if self._env_config.table_schema[column] in ["INTEGER", "FLOAT"]:  # type: ignore[index]
                    filter_expressions.append(f"base.{column} = {value}")
                else:
                    filter_expressions.append(f"base.{column} = '{value}'")
            where_filter_expr = " AND ".join(filter_expressions)
        else:
            where_filter_expr = "TRUE"

        embeddings_query = "with embeddings as (\n"
        for i, emb in enumerate(embeddings):
            embeddings_query += (
                f"SELECT {i} as row_num, @emb_{i} AS text_embedding"
                if i == 0
                else f"\nUNION ALL\nSELECT {i} as row_num, @emb_{i} AS text_embedding"
            )
        embeddings_query += "\n)\n"

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(f"emb_{i}", "FLOAT64", emb)
                for i, emb in enumerate(embeddings)
            ],
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )
        full_query = (
            embeddings_query
            + f"""
        SELECT
            base.*,
            query.row_num,
            distance AS score
        FROM VECTOR_SEARCH(
            TABLE `{self._env_config.full_table_id}`,
            "text_embedding",
            (SELECT row_num, {self._env_config.text_embedding_field} from embeddings),
            distance_type => "{self.distance_type}",
            top_k => {k}
        )
        WHERE {where_filter_expr}
        ORDER BY row_num, score
        """
        )
        results = self._env_config.bq_client.query(  # type: ignore[union-attr]
            full_query,
            job_config=job_config,
            api_method=bigquery.enums.QueryApiMethod.QUERY,
        )
        return list(results)


class FeatureOnlineStoreExecutor(BaseExecutor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["feature_online_store"] = "feature_online_store"
    online_store_name: Union[str, None] = None
    view_name: Union[str, None] = None
    online_store_type: Literal["bigtable", "optimized"] = "optimized"
    cron_schedule: Union[str, None] = None
    location: Union[str, None] = None
    min_node_count: int = 1
    max_node_count: int = 3
    cpu_utilization_target: int = 50
    algorithm_config: AlgorithmConfig = utils.TreeAhConfig()
    filter_columns: Optional[List[str]] = None
    crowding_column: Optional[str] = None
    distance_measure_type: Optional[
        DistanceMeasureType
    ] = utils.DistanceMeasureType.DOT_PRODUCT_DISTANCE
    _env_config: EnvConfig = EnvConfig()
    _user_agent: str = ""

    def model_post_init(self, __context: Any) -> None:
        _, self._user_agent = get_user_agent(
            f"{USER_AGENT_PREFIX}-{type(self).__name__}"
        )

    def set_env_config(self, env_config: Any):
        super().set_env_config(env_config)
        self.init_feature_store()

    def _validate_bq_existing_source(
        self,
        project_id_param,
        dataset_param,
        table_param,
    ):
        bq_uri_split = self._feature_view.gca_resource.big_query_source.uri.split(".")  # type: ignore[union-attr]
        project_id = bq_uri_split[0].replace("bq://", "")
        dataset = bq_uri_split[1]
        table = bq_uri_split[2]
        try:
            assert project_id == project_id_param
            assert dataset == dataset_param
            assert table == table_param
        except AssertionError:
            error_message = (
                "The BQ table passed in input is"
                f"bq://{project_id_param}.{dataset_param}.{table_param} "
                f"while the BQ table linked to the feature view is "
                "{self._feature_view.gca_resource.big_query_source.uri}."
                "Make sure you are using the same table for the feature "
                "view."
            )
            raise AssertionError(error_message)

    def init_feature_store(self):
        self.online_store_name = self.online_store_name or self._env_config.dataset_name
        self.view_name = self.view_name or self._env_config.table_name
        self.location = self.location or self._env_config.location
        vertexai.init(project=self._env_config.project_id, location=self.location)

        api_endpoint = f"{self._env_config.location}-aiplatform.googleapis.com"
        self._admin_client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": api_endpoint},
            client_info=get_client_info(module=self._user_agent),
        )
        self._online_store = self._create_online_store()
        self._search_client = self._get_search_client()
        self._feature_view = self._get_feature_view()

    def _get_search_client(self) -> FeatureOnlineStoreServiceClient:
        gca_resource = self._online_store.gca_resource
        endpoint = gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name
        return FeatureOnlineStoreServiceClient(
            client_options={"api_endpoint": endpoint}
        )

    def _wait_until_dummy_query_success(self, timeout_seconds: int = 1200):
        """
        Waits until a dummy query succeeds, indicating the system is ready.
        """
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutExpired(
                    "Timeout of {} seconds exceeded".format(timeout_seconds),
                    timeout=timeout_seconds,
                )
            try:
                return self._search_embedding(
                    embedding=[1] * self._env_config.embedding_dimension,  # type: ignore[operator]
                    k=1,
                )
            except ServiceUnavailable:
                logger.info(
                    "DNS certificates are being propagated,"
                    " waiting for 10 seconds.  "
                )
                time.sleep(10)
            except MethodNotImplemented as e:
                if e.args and "Received http2 header with status" in e.args[0]:
                    logger.info(
                        "DNS certificates are being propagated,"
                        " waiting for 10 seconds.  "
                    )
                    time.sleep(10)
                else:
                    raise

    def sync(self):
        """Sync the data from the Big Query source into the Executor source"""
        self._feature_view = self._create_feature_view()
        self._validate_bq_existing_source(
            project_id_param=self._env_config.project_id,
            dataset_param=self._env_config.dataset_name,
            table_param=self._env_config.table_name,
        )
        sync_response = self._admin_client.sync_feature_view(
            feature_view=(
                f"projects/{self._env_config.project_id}/"
                f"locations/{self._env_config.location}"
                f"/featureOnlineStores/{self.online_store_name}"
                f"/featureViews/{self.view_name}"
            )
        )
        while True:
            feature_view_sync = self._admin_client.get_feature_view_sync(
                name=sync_response.feature_view_sync
            )
            if feature_view_sync.run_time.end_time.seconds > 0:
                status = (
                    "Succeed" if feature_view_sync.final_status.code == 0 else "Failed"
                )
                logger.info(f"Sync {status} for {feature_view_sync.name}.")
                break
            else:
                logger.info("Sync ongoing, waiting for 30 seconds.")
            time.sleep(30)

        self._wait_until_dummy_query_success()

    def similarity_search_by_vectors_with_scores_and_embeddings(
        self,
        embeddings: List[List[float]],
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        batch_size: Union[int, None] = None,
        **kwargs,
    ) -> list[list[list[Any]]]:
        """Performs a similarity search using vector embeddings

        This function takes a set of query embeddings and searches for similar documents
        It returns the top-k matching documents, along with their similarity scores
        and their corresponding embeddings.

        Args:
            embeddings: A list of lists, where each inner list represents a
                query embedding.
            filter: (Optional) A dictionary specifying filter criteria for document
                on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
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
        output = []
        for query_embedding in embeddings:
            documents = []
            results = self._search_embedding(embedding=query_embedding, k=k, **kwargs)

            for result in results:
                metadata, embedding = {}, None

                for feature in result.entity_key_values.key_values.features:
                    if feature.name not in [
                        self._env_config.text_embedding_field,
                        self._env_config.content_field,
                    ]:
                        dict_values = proto.Message.to_dict(feature.value)
                        col_type, value = next(iter(dict_values.items()))
                        value = cast_proto_type(column=col_type, value=value)
                        metadata[feature.name] = value
                    if feature.name == self._env_config.text_embedding_field:
                        embedding = feature.value.double_array_value.values
                    if feature.name == self._env_config.content_field:
                        dict_values = proto.Message.to_dict(feature.value)
                        content = list(dict_values.values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                documents.append(
                    [
                        Document(
                            page_content=content,
                            metadata=metadata,
                        ),
                        result.distance,
                        embedding,
                    ]
                )
            output.append(documents)
        return output

    def get_documents(
        self,
        ids: Optional[List[str]],
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
        output = []
        if ids is None:
            raise ValueError(
                "Feature Store executor doesn't support search by filter " "only"
            )
        for id in ids:
            with telemetry.tool_context_manager(self._user_agent):
                result = self._feature_view.read(key=[id])  # type: ignore[union-attr]
                metadata, content = {}, None
                for feature in result.to_dict()["features"]:
                    if feature["name"] not in [
                        self._env_config.text_embedding_field,
                        self._env_config.content_field,
                    ]:
                        metadata[feature["name"]] = list(feature["value"].values())[0]
                    if feature["name"] == self._env_config.content_field:
                        content = list(feature["value"].values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                output.append(
                    Document(
                        page_content=str(content),
                        metadata=metadata,
                    )
                )
        return output

    def search_neighbors_by_ids(
        self, ids: List[str], filter: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """Searches for neighboring entities in a Vertex Feature Store based on
            their IDs and optional filter on metadata

        Args:
            ids: A list of string identifiers representing the entities to search for.
            filter: (Optional) A dictionary specifying filter criteria for document
                    on metadata properties, e.g.
                                {
                                    "str_property": "foo",
                                    "int_property": 123
                                }
        """
        output = []
        if ids is None:
            raise ValueError(
                "Feature Store executor doesn't support search by filter " "only"
            )
        for entity_id in ids:
            results = self._search_embedding(entity_id=entity_id, **kwargs)
            for result in results:
                metadata, embedding = {}, None
                for feature in result.entity_key_values.key_values.features:
                    if feature.name not in [
                        self._env_config.text_embedding_field,
                        self._env_config.content_field,
                    ]:
                        dict_values = proto.Message.to_dict(feature.value)
                        metadata[feature.name] = list(dict_values.values())[0]
                    if feature.name == self._env_config.text_embedding_field:
                        embedding = feature.value.double_array_value
                    if feature.name == self._env_config.content_field:
                        dict_values = proto.Message.to_dict(feature.value)
                        content = list(dict_values.values())[0]
                if filter is not None and not doc_match_filter(
                    document=metadata, filter=filter
                ):
                    continue
                output.append(
                    [
                        Document(
                            page_content=content,
                            metadata=metadata,
                        ),
                        result.distance,
                        embedding,
                    ]
                )

        return output  # type: ignore[return-value]

    def _search_embedding(
        self,
        embedding: Optional[Any] = None,
        entity_id: Optional[str] = None,
        k: int = 5,
        string_filters: Optional[List[NearestNeighborQuery.StringFilter]] = None,
        per_crowding_attribute_neighbor_count: Optional[int] = None,
        approximate_neighbor_candidates: Optional[int] = None,
        leaf_nodes_search_fraction: Optional[float] = None,
    ) -> MutableSequence[Any]:
        if embedding:
            embedding = NearestNeighborQuery.Embedding(value=embedding)
        query = NearestNeighborQuery(
            entity_id=entity_id,
            embedding=embedding,
            neighbor_count=k,
            string_filters=string_filters,
            per_crowding_attribute_neighbor_count=per_crowding_attribute_neighbor_count,
            parameters={
                "approximate_neighbor_candidates": approximate_neighbor_candidates,
                "leaf_nodes_search_fraction": leaf_nodes_search_fraction,
            },
        )
        with telemetry.tool_context_manager(self._user_agent):
            result = self._search_client.search_nearest_entities(
                request=feature_online_store_service.SearchNearestEntitiesRequest(
                    feature_view=self._feature_view.gca_resource.name,  # type: ignore[union-attr]
                    query=query,
                    return_full_entity=True,  # returning entities with metadata
                )
            )
        return result.nearest_neighbors.neighbors

    def _create_online_store(self) -> FeatureOnlineStore:
        # Search for existing Online store
        stores_list = FeatureOnlineStore.list(
            project=self._env_config.project_id, location=self._env_config.location
        )
        for store in stores_list:
            if store.name == self.online_store_name:
                return store

        # Create it otherwise
        if self.online_store_type == "bigtable":
            online_store_config = feature_online_store_pb2.FeatureOnlineStore(
                bigtable=feature_online_store_pb2.FeatureOnlineStore.Bigtable(
                    auto_scaling=feature_online_store_pb2.FeatureOnlineStore.Bigtable.AutoScaling(
                        min_node_count=self.min_node_count,
                        max_node_count=self.max_node_count,
                        cpu_utilization_target=self.cpu_utilization_target,
                    )
                ),
                embedding_management=feature_online_store_pb2.FeatureOnlineStore.EmbeddingManagement(
                    enabled=True
                ),
            )
            create_store_lro = self._admin_client.create_feature_online_store(
                parent=f"projects/{self._env_config.project_id}/locations/{self._env_config.location}",
                feature_online_store_id=self.online_store_name,
                feature_online_store=online_store_config,
            )
            logger.info(create_store_lro.result())
        elif self.online_store_type == "optimized":
            online_store_config = feature_online_store_pb2.FeatureOnlineStore(
                optimized=feature_online_store_pb2.FeatureOnlineStore.Optimized()
            )
            create_store_lro = self._admin_client.create_feature_online_store(
                parent=f"projects/{self._env_config.project_id}/locations/{self._env_config.location}",
                feature_online_store_id=self.online_store_name,
                feature_online_store=online_store_config,
            )
            logger.info(create_store_lro.result())
            logger.info(create_store_lro.result())

        else:
            raise ValueError(
                f"{self.online_store_type} not allowed. "
                f"Accepted values are 'bigtable' or 'optimized'."
            )
        stores_list = FeatureOnlineStore.list(
            project=self._env_config.project_id, location=self._env_config.location
        )
        for store in stores_list:
            if store.name == self.online_store_name:
                return store

    def _create_feature_view(self) -> FeatureView:
        fv = self._get_feature_view()
        if fv:
            return fv
        else:
            big_query_source = FeatureViewBigQuerySource(
                uri=f"bq://{self._env_config.full_table_id}",
                entity_id_columns=[self._env_config.doc_id_field],
            )
            index_config = utils.IndexConfig(
                embedding_column=self._env_config.text_embedding_field,
                crowding_column=self.crowding_column,
                filter_columns=self.filter_columns,
                dimensions=self._env_config.embedding_dimension,
                distance_measure_type=self.distance_measure_type,
                algorithm_config=self.algorithm_config,
            )
            return self._online_store.create_feature_view(
                name=self.view_name,
                source=big_query_source,
                sync_config=self.cron_schedule,
                index_config=index_config,
                project=self._env_config.project_id,
                location=self._env_config.location,
            )

    def _get_feature_view(self) -> FeatureView | None:
        # Search for existing Feature view
        fv_list = FeatureView.list(
            feature_online_store_id=self._online_store.gca_resource.name
        )
        for fv in fv_list:
            if fv.name == self.view_name:
                return fv
        return None
