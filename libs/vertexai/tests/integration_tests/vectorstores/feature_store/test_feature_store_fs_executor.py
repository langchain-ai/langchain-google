"""
Test Vertex Feature Store Vector Search with Feature Store executor.
"""

import os
import random

import pytest

from langchain_google_vertexai.vectorstores.feature_store.executors import (
    FeatureOnlineStoreExecutor,
)
from langchain_google_vertexai.vectorstores.feature_store.feature_store import (
    FeatureStore,
)

# Feature Online store is static to avoid cold start setup time during testing
TEST_DATASET = "langchain_test_dataset"
TEST_TABLE_NAME = f"langchain_test_table{str(random.randint(1,100000))}"
TEST_FOS_NAME = "langchain_test_fos"
TEST_VIEW_NAME = f"test{str(random.randint(1,100000))}"


@pytest.fixture(scope="class")
def store_fs_executor(request: pytest.FixtureRequest) -> FeatureStore:
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
    TestFeatureStore_fs_executor.store_fs_executor = FeatureStore(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=embedding_model,
        location="us-central1",
        dataset_name=TEST_DATASET,
        table_name=TEST_TABLE_NAME,
        executor=FeatureOnlineStoreExecutor(
            online_store_name=TEST_FOS_NAME,
            view_name=TEST_VIEW_NAME,
        ),
    )
    TestFeatureStore_fs_executor.ids = (
        TestFeatureStore_fs_executor.store_fs_executor.add_texts(
            TestFeatureStore_fs_executor.texts,
            TestFeatureStore_fs_executor.metadatas,
        )
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TestFeatureStore_fs_executor.store_fs_executor.dataset_name,
            delete_contents=True,
            not_found_ok=True,
        )
        TestFeatureStore_fs_executor.store_fs_executor.executor._feature_view.delete()
    
    request.addfinalizer(teardown)
    return TestFeatureStore_fs_executor.store_fs_executor


class TestFeatureStore_fs_executor:
    """BigQueryVectorStore tests class."""

    ids: list = []
    store_fs_executor: FeatureStore
    texts = ["apple", "ice cream", "Saturn", "candy", "banana"]
    metadatas = [
        {"kind": "fruit", "chunk": 0},
        {"kind": "treat", "chunk": 1},
        {"kind": "planet", "chunk": 2},
        {"kind": "treat", "chunk": 3},
        {"kind": "fruit", "chunk": 4},
    ]

    @pytest.mark.extended
    def test_semantic_search(self, store_fs_executor: FeatureStore) -> None:
        """Test on semantic similarity."""
        docs = store_fs_executor.similarity_search("fruit", k=4)
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "planet" not in kinds
        chunks = [d.metadata["chunk"] for d in docs]
        assert 0 in chunks
        assert 4 in chunks

    @pytest.mark.extended
    def test_semantic_search_filter_fruits(
        self, store_fs_executor: FeatureStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store_fs_executor.similarity_search("apple", filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_add_texts_with_embeddings_with_error(
        self, store_fs_executor: FeatureStore
    ) -> None:
        new_texts = ["chocolate", "Mars"]
        new_embs = store_fs_executor.embedding.embed_documents(new_texts)
        new_metadatas = [{"kind": "treat"}, {"kind": "planet"}]
        new_executor = FeatureStore(
            project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
            embedding=store_fs_executor.embedding,
            location="us-central1",
            dataset_name=TEST_DATASET,
            table_name="new_table_name",
            executor={  # type: ignore[arg-type]
                "type": "feature_online_store",
                "online_store_name": TEST_FOS_NAME,
                "view_name": TEST_VIEW_NAME,
            },
        )
        with pytest.raises(AssertionError):
            _ = new_executor.add_texts_with_embeddings(
                texts=new_texts, embs=new_embs, metadatas=new_metadatas
            )

    @pytest.mark.extended
    def test_get_doc_by_ids(self, store_fs_executor: FeatureStore) -> None:
        ids = TestFeatureStore_fs_executor.ids[0:2]

        retrieved_docs = store_fs_executor.get_documents(ids=ids)
        assert len(retrieved_docs) == 2

    @pytest.mark.extended
    def test_get_doc_by_filter(self, store_fs_executor: FeatureStore) -> None:
        """Test on document retrieval with metadata filter."""
        with pytest.raises(ValueError):
            _ = store_fs_executor.get_documents(filter={"kind": "fruit"})

    @pytest.mark.extended
    def test_batch_search_with_embeddings(
        self, store_fs_executor: FeatureStore
    ) -> None:
        queries = ["red fruit", "planet with rings"]
        embeddings = store_fs_executor.embedding.embed_documents(queries)

        results = store_fs_executor.batch_search(embeddings=embeddings, k=2)
        assert len(results) == 2
        assert len(results[0]) == 2  # Ensure 2 results per query
        assert len(results[1]) == 2
