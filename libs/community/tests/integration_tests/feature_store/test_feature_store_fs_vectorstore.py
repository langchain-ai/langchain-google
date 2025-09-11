"""
Test Vertex Feature Store Vector Search with Feature Store vectorstore.
"""

import os
import random

import grpc
import pytest

from langchain_google_community import VertexFSVectorStore
from tests.integration_tests.fake import FakeEmbeddings

# Feature Online store is static to avoid cold start setup time during testing
TEST_DATASET = "langchain_test_dataset"
TEST_TABLE_NAME = f"langchain_test_table{str(random.randint(1,100000))}"
TEST_FOS_NAME = "langchain_test_fos"
TEST_VIEW_NAME = f"test{str(random.randint(1,100000))}"
EMBEDDING_SIZE = 768


@pytest.fixture(scope="class")
def store_fs_vectorstore(request: pytest.FixtureRequest) -> VertexFSVectorStore:
    """BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery

    embedding_model = FakeEmbeddings(size=EMBEDDING_SIZE)

    TestVertexFSVectorStore_fs_vectorstore.store_fs_vectorstore = VertexFSVectorStore(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=embedding_model,
        location="us-central1",
        dataset_name=TEST_DATASET,
        table_name=TEST_TABLE_NAME,
        online_store_name=TEST_FOS_NAME,
    )
    TestVertexFSVectorStore_fs_vectorstore.ids = (
        TestVertexFSVectorStore_fs_vectorstore.store_fs_vectorstore.add_texts(
            TestVertexFSVectorStore_fs_vectorstore.texts,
            TestVertexFSVectorStore_fs_vectorstore.metadatas,
        )
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TestVertexFSVectorStore_fs_vectorstore.store_fs_vectorstore.dataset_name,
            delete_contents=True,
            not_found_ok=True,
        )
        fs_vs = TestVertexFSVectorStore_fs_vectorstore.store_fs_vectorstore
        fs_vs.feature_view.delete()  # type: ignore[union-attr]

    request.addfinalizer(teardown)
    return TestVertexFSVectorStore_fs_vectorstore.store_fs_vectorstore


class TestVertexFSVectorStore_fs_vectorstore:
    """BigQueryVectorStore tests class."""

    ids: list = []
    store_fs_vectorstore: VertexFSVectorStore
    texts = ["apple", "ice cream", "Saturn", "candy", "banana"]
    metadatas = [
        {"kind": "fruit", "chunk": 0},
        {"kind": "treat", "chunk": 1},
        {"kind": "planet", "chunk": 2},
        {"kind": "treat", "chunk": 3},
        {"kind": "fruit", "chunk": 4},
    ]

    @pytest.mark.extended
    def test_semantic_search(self, store_fs_vectorstore: VertexFSVectorStore) -> None:
        """Test on semantic similarity."""
        docs = store_fs_vectorstore.similarity_search("fruit", k=2)
        kinds = [d.metadata["kind"] for d in docs]
        assert len(kinds) == 2

    @pytest.mark.extended
    def test_semantic_search_filter_fruits(
        self, store_fs_vectorstore: VertexFSVectorStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store_fs_vectorstore.similarity_search("apple", filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_add_texts_with_embeddings_with_error(
        self, store_fs_vectorstore: VertexFSVectorStore
    ) -> None:
        new_texts = ["chocolate", "Mars"]
        new_embs = store_fs_vectorstore.embedding.embed_documents(new_texts)
        new_metadatas = [{"kind": "treat"}, {"kind": "planet"}]
        new_vectorstore = VertexFSVectorStore(
            project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
            embedding=store_fs_vectorstore.embedding,
            location="us-central1",
            dataset_name=store_fs_vectorstore.dataset_name,
            table_name=f"error_table{str(random.randint(1,100000))}",
            online_store_name=store_fs_vectorstore.online_store_name,
            view_name=store_fs_vectorstore.view_name,
        )
        with pytest.raises(AssertionError):
            _ = new_vectorstore.add_texts_with_embeddings(
                texts=new_texts, embs=new_embs, metadatas=new_metadatas
            )

    @pytest.mark.extended
    def test_get_doc_by_ids(self, store_fs_vectorstore: VertexFSVectorStore) -> None:
        ids = TestVertexFSVectorStore_fs_vectorstore.ids[0:2]

        retrieved_docs = store_fs_vectorstore.get_documents(ids=ids)
        assert len(retrieved_docs) == 2

    @pytest.mark.extended
    def test_to_bq_vector_store(
        self, store_fs_vectorstore: VertexFSVectorStore
    ) -> None:
        """Test getter feature store vectorstore"""
        new_store = store_fs_vectorstore.to_bq_vector_store()
        assert new_store.dataset_name == TEST_DATASET


@pytest.mark.extended
def test_psc_feature_store() -> None:
    """Test creation of feature store with private service connect enabled"""
    # ruff: noqa: E501
    from google.cloud.aiplatform_v1.services.feature_online_store_service.transports.grpc import (
        FeatureOnlineStoreServiceGrpcTransport,
    )

    embedding_model = FakeEmbeddings(size=EMBEDDING_SIZE)
    project_id = os.environ.get("PROJECT_ID", None)

    transport = FeatureOnlineStoreServiceGrpcTransport(
        channel=grpc.insecure_channel("dummy:10002")
    )
    try:
        vertex_fs = VertexFSVectorStore(
            project_id=project_id,  # type: ignore[arg-type]
            location="us-central1",
            dataset_name=TEST_DATASET + f"_psc_{str(random.randint(1,100000))}",
            table_name=TEST_TABLE_NAME,
            embedding=embedding_model,
            enable_private_service_connect=True,
            project_allowlist=[project_id],  # type: ignore[list-item]
            transport=transport,
        )
    finally:
        # Clean up resources
        vertex_fs.online_store.delete()
