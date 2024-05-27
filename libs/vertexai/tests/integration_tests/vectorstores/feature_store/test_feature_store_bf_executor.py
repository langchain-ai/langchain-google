"""
Test Vertex Feature Store Vector Search with BF local executor.
"""

import os
import uuid

import pytest

from langchain_google_vertexai.vectorstores.feature_store.executors import (
    BruteForceExecutor,
)
from langchain_google_vertexai.vectorstores.feature_store.feature_store import (
    FeatureStore,
)

TEST_TABLE_NAME = "langchain_test_table"


@pytest.fixture(scope="class")
def store_bf_executor(request: pytest.FixtureRequest) -> FeatureStore:
    """BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery

    from langchain_google_vertexai import VertexAIEmbeddings

    embedding_model = VertexAIEmbeddings(
        model_name="textembedding-gecko@latest",
        project=os.environ.get("PROJECT_ID", None),
    )
    TestFeatureStore_bf_executor.store_bf_executor = FeatureStore(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=embedding_model,
        location="us-central1",
        dataset_name=TestFeatureStore_bf_executor.dataset_name,
        table_name=TEST_TABLE_NAME,
        executor=BruteForceExecutor(),
    )
    TestFeatureStore_bf_executor.store_bf_executor.add_texts(
        TestFeatureStore_bf_executor.texts,
        TestFeatureStore_bf_executor.metadatas,
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TestFeatureStore_bf_executor.dataset_name,
            delete_contents=True,
            not_found_ok=True,
        )

    request.addfinalizer(teardown)
    return TestFeatureStore_bf_executor.store_bf_executor


class TestFeatureStore_bf_executor:
    """BigQueryVectorStore tests class."""

    dataset_name = uuid.uuid4().hex
    store_bf_executor: FeatureStore
    texts = ["apple", "ice cream", "Saturn", "candy", "banana"]
    metadatas = [
        {
            "kind": "fruit",
        },
        {
            "kind": "treat",
        },
        {
            "kind": "planet",
        },
        {
            "kind": "treat",
        },
        {
            "kind": "fruit",
        },
    ]

    @pytest.mark.extended
    def test_semantic_search(self, store_bf_executor: FeatureStore) -> None:
        """Test on semantic similarity."""
        docs = store_bf_executor.similarity_search("food", k=4)
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_semantic_search_filter_fruits(
        self, store_bf_executor: FeatureStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store_bf_executor.similarity_search("food", filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_doc_by_filter(self, store_bf_executor: FeatureStore) -> None:
        """Test on document retrieval with metadata filter."""
        docs = store_bf_executor.get_documents(filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds
