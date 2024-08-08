"""Test BigQuery Vector Search.
In order to run this test, you need to install Google Cloud BigQuery SDK
pip install google-cloud-bigquery
Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
"""

import os
import uuid

import pytest

from langchain_google_community import BigQueryVectorSearch
from tests.integration_tests.fake import FakeEmbeddings

TEST_TABLE_NAME = "langchain_test_table"


@pytest.fixture(scope="class")
def store(request: pytest.FixtureRequest) -> BigQueryVectorSearch:
    """BigQueryVectorStore tests context.

    In order to run this test, you define PROJECT_ID environment variable
    with GCP project id.

    Example:
    export PROJECT_ID=...
    """
    from google.cloud import bigquery  # type: ignore[attr-defined]

    bigquery.Client(location="US").create_dataset(
        TestBigQueryVectorStore.dataset_name, exists_ok=True
    )
    TestBigQueryVectorStore.store = BigQueryVectorSearch(
        project_id=os.environ.get("PROJECT_ID", None),  # type: ignore[arg-type]
        embedding=FakeEmbeddings(),  # type: ignore[call-arg]
        dataset_name=TestBigQueryVectorStore.dataset_name,
        table_name=TEST_TABLE_NAME,
    )
    TestBigQueryVectorStore.store.add_texts(
        TestBigQueryVectorStore.texts, TestBigQueryVectorStore.metadatas
    )

    def teardown() -> None:
        bigquery.Client(location="US").delete_dataset(
            TestBigQueryVectorStore.dataset_name,
            delete_contents=True,
            not_found_ok=True,
        )

    request.addfinalizer(teardown)
    return TestBigQueryVectorStore.store


class TestBigQueryVectorStore:
    """BigQueryVectorStore tests class."""

    dataset_name = uuid.uuid4().hex
    store: BigQueryVectorSearch
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

    @pytest.mark.skip(reason="investigating")
    @pytest.mark.extended
    def test_semantic_search(self, store: BigQueryVectorSearch) -> None:
        """Test on semantic similarity."""
        docs = store.similarity_search("food", k=4)
        print(docs)  # noqa: T201
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" in kinds
        assert "planet" not in kinds

    @pytest.mark.skip(reason="investigating")
    @pytest.mark.extended
    def test_semantic_search_filter_fruits(self, store: BigQueryVectorSearch) -> None:
        """Test on semantic similarity with metadata filter."""
        docs = store.similarity_search("food", filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds

    @pytest.mark.skip(reason="investigating")
    @pytest.mark.extended
    def test_get_doc_by_filter(self, store: BigQueryVectorSearch) -> None:
        """Test on document retrieval with metadata filter."""
        docs = store.get_documents(filter={"kind": "fruit"})
        kinds = [d.metadata["kind"] for d in docs]
        assert "fruit" in kinds
        assert "treat" not in kinds
        assert "planet" not in kinds
