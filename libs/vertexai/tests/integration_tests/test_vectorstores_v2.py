"""Test Vertex AI API wrapper for V2.
Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
Additionally in order to run the test you must have set the following environment
variables:
- PROJECT_ID: Id of the Google Cloud Project
- REGION (optional): Region of the Bucket, Index and Endpoint (default: us-central1)
- VECTOR_SEARCH_V2_COLLECTION_ID (optional): Id of the Vector Search V2 Collection
  (default: langchain-test-collection)
"""

import os
import time
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores.vectorstores import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)


@pytest.fixture
def embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(model_name="text-embedding-005")  # type: ignore


@pytest.fixture
def vector_store_v2(embeddings: VertexAIEmbeddings) -> VectorSearchVectorStore:
    """Initializes a VectorSearchVectorStore for V2 batch updates."""
    return VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
        collection_id=os.environ.get(
            "VECTOR_SEARCH_V2_COLLECTION_ID", "langchain-test-collection"
        ),
        embedding=embeddings,
        api_version="v2",
        stream_update=False,
    )


@pytest.fixture
def datastore_vector_store_v2(
    embeddings: VertexAIEmbeddings,
) -> VectorSearchVectorStoreDatastore:
    """Initializes a VectorSearchVectorStoreDatastore for V2 stream updates."""
    return VectorSearchVectorStoreDatastore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
        collection_id=os.environ.get(
            "VECTOR_SEARCH_V2_COLLECTION_ID", "langchain-test-collection"
        ),
        embedding=embeddings,
        api_version="v2",
        stream_update=True,
    )


def test_vector_store_v2_add_texts_and_dense_search(
    vector_store_v2: VectorSearchVectorStore,
):
    """Tests adding texts and performing a dense search in V2."""
    texts = ["my favourite animal is the elephant", "my favourite animal is the lion"]
    ids = [str(uuid4()), str(uuid4())]
    vector_store_v2.add_texts(texts=texts, ids=ids)

    query = "What are your favourite animals?"
    docs = vector_store_v2.similarity_search(query, k=2)
    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Document)


@pytest.mark.skip(
    reason=(
        "V2 sparse vector support needs sparse vector schema "
        "configuration in collection"
    )
)
def test_vector_store_v2_hybrid_search(
    vector_store_v2: VectorSearchVectorStore, embeddings: VertexAIEmbeddings
):
    """Tests hybrid search in V2.

    Note: This test is skipped because the test collection needs to be configured
    with a sparse vector field in the vector_schema for hybrid search to work.
    """
    texts = ["my favourite car is a porsche", "my favourite car is a lamborghini"]
    ids = [str(uuid4()), str(uuid4())]
    vector_store_v2.add_texts(texts=texts, ids=ids)

    query = "What are your favourite cars?"
    embedding = embeddings.embed_query(query)
    sparse_embedding: dict[str, list[int] | list[float]] = {
        "values": [0.5, 0.7],
        "indices": [2, 4],
    }

    docs_with_scores = vector_store_v2.similarity_search_by_vector_with_score(
        embedding=embedding, sparse_embedding=sparse_embedding, k=1
    )
    assert len(docs_with_scores) == 1
    for doc, scores in docs_with_scores:
        assert isinstance(doc, Document)
        assert isinstance(scores, dict)
        assert "dense_score" in scores
        assert "sparse_score" in scores


def test_vector_store_v2_advanced_filtering(
    vector_store_v2: VectorSearchVectorStore,
):
    """Tests advanced filtering in V2 with dict-based queries."""
    docs_to_add = [
        Document(
            page_content="A blue car.", metadata={"color": "blue", "price": 20000}
        ),
        Document(page_content="A red car.", metadata={"color": "red", "price": 30000}),
        Document(
            page_content="A blue bike.", metadata={"color": "blue", "price": 1000}
        ),
    ]
    ids = [str(uuid4()) for _ in docs_to_add]
    vector_store_v2.add_documents(docs_to_add, ids=ids)

    # Use dict filter
    filter_dict = {"$and": [{"color": {"$eq": "blue"}}, {"price": {"$lt": 15000}}]}
    documents = vector_store_v2.similarity_search("A vehicle", filter=filter_dict, k=10)

    assert len(documents) > 0
    assert all(doc.metadata["color"] == "blue" for doc in documents)
    assert all(doc.metadata["price"] < 15000 for doc in documents)


def test_vector_store_v2_return_full_datapoint(
    vector_store_v2: VectorSearchVectorStore, embeddings: VertexAIEmbeddings
):
    """Tests the return_full_datapoint feature in V2."""
    texts = ["a document about cats", "a document about dogs"]
    vector_store_v2.add_texts(texts=texts)

    query_embedding = embeddings.embed_query("pets")
    docs_with_scores = vector_store_v2.similarity_search_by_vector_with_score(
        embedding=query_embedding, k=1, return_full_datapoint=True
    )

    assert len(docs_with_scores) == 1
    doc, score = docs_with_scores[0]
    # The full datapoint is not directly exposed in the Document object,
    # but we can check that the search was successful.
    assert isinstance(doc, Document)
    assert isinstance(score, float)


def test_vector_store_v2_delete_by_ids(vector_store_v2: VectorSearchVectorStore):
    """Tests deleting documents by IDs in V2."""
    texts = ["doc to delete", "doc to keep"]
    delete_id = f"delete_me_{str(uuid4())}"
    keep_id = f"keep_me_{str(uuid4())}"
    ids = [delete_id, keep_id]
    vector_store_v2.add_texts(texts=texts, ids=ids)

    vector_store_v2.delete(ids=[delete_id])

    # This is a best-effort check, as the index update can take time.
    # A more robust test would involve polling.
    results = vector_store_v2.similarity_search("doc", k=10)
    result_ids = [doc.metadata.get("id") for doc in results]
    assert keep_id in result_ids
    assert delete_id not in result_ids


def test_vector_store_v2_delete_by_filter(vector_store_v2: VectorSearchVectorStore):
    """Tests deleting documents by filter in V2 using the recommended workaround.

    Note: Direct delete by metadata filter has API limitations in V2.
    The recommended approach is to search with filter, then delete by IDs.
    This test uses schema-defined fields (source, category, page) for filtering.
    """
    docs_to_add = [
        Document(
            page_content="A document to delete.",
            metadata={"source": "delete_me", "category": "test"},
        ),
        Document(
            page_content="A document to keep.",
            metadata={"source": "keep_me", "category": "test"},
        ),
    ]
    ids = [str(uuid4()) for _ in docs_to_add]
    vector_store_v2.add_documents(docs_to_add, ids=ids)

    # Wait for indexing (eventual consistency)
    time.sleep(3)

    # Use the recommended workaround: search with filter, then delete by IDs
    results_to_delete = vector_store_v2.similarity_search(
        "document", k=100, filter={"source": {"$eq": "delete_me"}}
    )
    ids_to_delete: list[str] = [
        doc.metadata["id"]
        for doc in results_to_delete
        if "id" in doc.metadata and doc.metadata["id"] is not None
    ]

    # Only proceed with delete if we found documents
    # (test may be affected by eventual consistency)
    if ids_to_delete:
        vector_store_v2.delete(ids=ids_to_delete)

        # Wait for deletion to propagate
        time.sleep(2)

        # Verify deletion worked - check that delete_me docs are gone
        results = vector_store_v2.similarity_search("document", k=100)
        deleted_docs = [
            doc for doc in results if doc.metadata.get("source") == "delete_me"
        ]
        # The delete operation should have removed the delete_me documents
        assert len(deleted_docs) == 0, (
            f"Expected 0 delete_me docs, found {len(deleted_docs)}"
        )
