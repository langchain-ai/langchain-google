"""
Test Vertex Feature Store Vector Search with BQ Vector Search vectorstore.
"""

import os
import random

import pytest

from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryVectorStore,
)
from tests.integration_tests.fake import FakeEmbeddings

TEST_DATASET = "langchain_test_dataset"
TEST_TEMP_DATASET = "temp_langchain_test_dataset"
TEST_TABLE_NAME = f"langchain_test_table{str(random.randint(1, 100000))}"
TEST_FOS_NAME = "langchain_test_fos"
EMBEDDING_SIZE = 768


@pytest.fixture(scope="class")
def store_bq_vectorstore(request: pytest.FixtureRequest) -> BigQueryVectorStore:
    """BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery  # type: ignore[attr-defined]

    embedding_model = FakeEmbeddings(size=EMBEDDING_SIZE)
    TestBigQueryVectorStore_bq_vectorstore.store_bq_vectorstore = BigQueryVectorStore(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=embedding_model,
        location="us-central1",
        dataset_name=TEST_DATASET,
        temp_dataset_name=TEST_TEMP_DATASET,
        table_name=TEST_TABLE_NAME,
    )
    TestBigQueryVectorStore_bq_vectorstore.store_bq_vectorstore.add_texts(
        TestBigQueryVectorStore_bq_vectorstore.texts,
        TestBigQueryVectorStore_bq_vectorstore.metadatas,
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TEST_DATASET,
            delete_contents=True,
            not_found_ok=True,
        )

    request.addfinalizer(teardown)
    return TestBigQueryVectorStore_bq_vectorstore.store_bq_vectorstore


@pytest.fixture(scope="class")
def existing_store_bq_vectorstore(
    request: pytest.FixtureRequest,
) -> BigQueryVectorStore:
    """Existing BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery  # type: ignore[attr-defined]

    embedding_model = FakeEmbeddings(size=EMBEDDING_SIZE)
    TestBigQueryVectorStore_bq_vectorstore.existing_store_bq_vectorstore = (
        BigQueryVectorStore(
            project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
            embedding=embedding_model,
            location="us-central1",
            dataset_name=TEST_DATASET,
            temp_dataset_name=TEST_TEMP_DATASET,
            table_name=TEST_TABLE_NAME,
        )
    )

    def teardown() -> None:
        bigquery.Client(location="us-central1").delete_dataset(
            TEST_DATASET,
            delete_contents=True,
            not_found_ok=True,
        )

    request.addfinalizer(teardown)
    return TestBigQueryVectorStore_bq_vectorstore.existing_store_bq_vectorstore


class TestBigQueryVectorStore_bq_vectorstore:
    """BigQueryVectorStore tests class."""

    store_bq_vectorstore: BigQueryVectorStore
    existing_store_bq_vectorstore: BigQueryVectorStore
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
    def test_semantic_search_sql_filter_fruits(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on semantic similarity with sql filter."""
        docs = store_bq_vectorstore.similarity_search(
            "food", filter='kind="fruit"', k=10
        )
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_doc_by_sql_filter(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on document retrieval with sql filter."""
        docs = store_bq_vectorstore.get_documents(filter='kind="fruit"')
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_doc_by_complexe_sql_filter(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on document retrieval with sql filter."""
        docs = store_bq_vectorstore.get_documents(filter='kind="fruit" OR kind="treat"')
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_existing_store_semantic_search_sql_filter_fruits(
        self, existing_store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on semantic similarity with sql filter."""
        docs = existing_store_bq_vectorstore.similarity_search(
            "food", filter='kind="fruit"', k=10
        )
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_existing_store_get_doc_by_sql_filter(
        self, existing_store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on document retrieval with sql filter."""
        docs = existing_store_bq_vectorstore.get_documents(filter='kind="fruit"')
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_semantic_search(self, store_bq_vectorstore: BigQueryVectorStore) -> None:
        """Test on semantic similarity."""
        docs = store_bq_vectorstore.similarity_search("food", k=4)
        kinds = [d.metadata["kind"] for d in docs]
        assert len(kinds) == 4

    @pytest.mark.extended
    def test_semantic_search_filter_fruits(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store_bq_vectorstore.similarity_search(
            "food", filter={"kind": "fruit"}, k=10
        )
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_doc_by_filter(self, store_bq_vectorstore: BigQueryVectorStore) -> None:
        """Test on document retrieval with metadata filter."""
        docs = store_bq_vectorstore.get_documents(filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_existing_store_semantic_search_filter_fruits(
        self, existing_store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = existing_store_bq_vectorstore.similarity_search(
            "food", filter={"kind": "fruit"}, k=10
        )
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_existing_store_get_doc_by_filter(
        self, existing_store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test on document retrieval with metadata filter."""
        docs = existing_store_bq_vectorstore.get_documents(filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.extended
    def test_get_documents_by_ids(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test retrieving documents by their IDs."""
        # Get the first two documents
        first_two_docs = store_bq_vectorstore.get_documents()[:2]
        ids_to_retrieve = [doc.metadata["__id"] for doc in first_two_docs]
        # Retrieve them by their IDs
        retrieved_docs = store_bq_vectorstore.get_documents(ids_to_retrieve)
        assert len(retrieved_docs) == 2
        # Check that the content and metadata match
        for orig_doc, retrieved_doc in zip(first_two_docs, retrieved_docs):
            assert orig_doc.page_content == retrieved_doc.page_content
            assert orig_doc.metadata == retrieved_doc.metadata

    @pytest.mark.extended
    def test_add_texts_with_embeddings(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test adding texts with pre-computed embeddings."""
        new_texts = ["chocolate", "mars"]
        new_metadatas = [{"kind": "treat"}, {"kind": "planet"}]
        new_embeddings = store_bq_vectorstore.embedding.embed_documents(new_texts)
        ids = store_bq_vectorstore.add_texts_with_embeddings(
            new_texts, new_embeddings, new_metadatas
        )
        assert len(ids) == 2  # Ensure we got IDs back
        # Verify the documents were added correctly
        retrieved_docs = store_bq_vectorstore.get_documents(ids)
        assert retrieved_docs[0].page_content == "chocolate"
        assert retrieved_docs[1].page_content == "mars"
        assert retrieved_docs[0].metadata["kind"] == "treat"
        assert retrieved_docs[1].metadata["kind"] == "planet"

    @pytest.mark.extended
    def test_get_documents_by_ids_and_filters(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test retrieving documents by their IDs and with filters."""
        # Add new text for testing
        new_texts = ["cat", "pigeon", "dog"]
        new_metadatas = [{"kind": "mammal"}, {"kind": "bird"}, {"kind": "mammal"}]
        new_embeddings = store_bq_vectorstore.embedding.embed_documents(new_texts)
        ids = store_bq_vectorstore.add_texts_with_embeddings(
            new_texts, new_embeddings, new_metadatas
        )
        # Retrieve addeds documents and
        # retrieved documents them by their IDs and filters
        orig_docs = store_bq_vectorstore.get_documents(ids=ids)
        retrieved_docs = store_bq_vectorstore.get_documents(
            ids=ids, filter='kind="mammal" AND content="dog"'
        )
        assert len(retrieved_docs) == 1
        # Check that the content and metadata match
        for orig_doc, retrieved_doc in zip(orig_docs, retrieved_docs):
            if (
                retrieved_doc.metadata == "mammal"
                and retrieved_doc.page_content == "dog"
            ):
                assert orig_doc.page_content == retrieved_doc.page_content
                assert orig_doc.metadata == retrieved_doc.metadata
        # Check that the filters worked
        kinds = [d.metadata["kind"] for d in retrieved_docs]
        page_content = [d.page_content for d in retrieved_docs]
        assert "mammal" in kinds
        assert "bird" not in kinds
        assert "fruit" not in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds
        assert "dog" == page_content[0]

    @pytest.mark.extended
    def test_delete_documents(self, store_bq_vectorstore: BigQueryVectorStore) -> None:
        """Test deleting documents by their IDs."""
        doc_to_delete = store_bq_vectorstore.get_documents()[0]
        id_to_delete = doc_to_delete.metadata["__id"]
        # Delete the document
        delete_result = store_bq_vectorstore.delete([id_to_delete])
        assert delete_result is True  # Deletion should succeed
        # Try to retrieve the deleted document

        result = store_bq_vectorstore.get_documents([id_to_delete])
        assert result == []

    @pytest.mark.extended
    def test_batch_search(self, store_bq_vectorstore: BigQueryVectorStore) -> None:
        """Test batch search with queries and embeddings."""
        # Batch search with queries
        query_results = store_bq_vectorstore.batch_search(queries=["apple", "treat"])
        assert len(query_results) == 2  # 2 queries
        assert all(
            len(result) > 0 for result in query_results
        )  # Results for each query

        # Batch search with embeddings
        embeddings = store_bq_vectorstore.embedding.embed_documents(["apple", "treat"])
        embedding_results = store_bq_vectorstore.batch_search(embeddings=embeddings)
        assert len(embedding_results) == 2  # 2 embeddings
        assert all(len(result) > 0 for result in embedding_results)

    @pytest.mark.extended
    def test_to_vertex_fs_vector_store(
        self, store_bq_vectorstore: BigQueryVectorStore
    ) -> None:
        """Test getter feature store vectorstore"""
        new_store = store_bq_vectorstore.to_vertex_fs_vector_store(
            online_store_name=TEST_FOS_NAME
        )
        assert new_store.online_store_name == TEST_FOS_NAME
