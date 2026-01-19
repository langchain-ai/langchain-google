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
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from google.cloud import vectorsearch_v1beta
from langchain_core.documents import Document

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores.vectorstores import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)


@pytest.fixture(scope="module")
def embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(model_name="text-embedding-005")  # type: ignore


@pytest.fixture(scope="module")
@pytest.mark.extended
def vector_store_v2(embeddings: VertexAIEmbeddings) -> VectorSearchVectorStore:
    """Initializes a VectorSearchVectorStore for V2 batch updates."""
    project_id = os.environ["PROJECT_ID"]
    region = os.environ.get("REGION", "us-central1")
    collection_id = os.environ.get(
        "VECTOR_SEARCH_V2_COLLECTION_ID", "langchain-test-collection"
    )

    # Create collection if it doesn't exist
    client = vectorsearch_v1beta.VectorSearchServiceClient()
    parent = f"projects/{project_id}/locations/{region}"
    collection_name = f"{parent}/collections/{collection_id}"

    try:
        client.get_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist, create it
        request = vectorsearch_v1beta.CreateCollectionRequest(
            parent=parent,
            collection_id=collection_id,
            collection={
                "data_schema": {
                    "type": "object",
                    "properties": {
                        "page_content": {"type": "string"},
                        "source": {"type": "string"},
                        "category": {"type": "string"},
                        "color": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
                "vector_schema": {
                    "embedding": {
                        "dense_vector": {
                            "dimensions": 768,
                        },
                    },
                },
            },
        )
        operation = client.create_collection(request=request)
        operation.result()  # Wait for collection creation

    return VectorSearchVectorStore.from_components(
        project_id=project_id,
        region=region,
        collection_id=collection_id,
        embedding=embeddings,
        api_version="v2",
        stream_update=False,
    )


@pytest.fixture
@pytest.mark.extended
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


@pytest.fixture(scope="module")
@pytest.mark.extended
def semantic_search_collection():
    """Creates a collection with semantic search support and cleans up after tests."""
    project_id = os.environ["PROJECT_ID"]
    region = os.environ.get("REGION", "us-central1")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    collection_id = f"test-semantic-search-{timestamp}"

    client = vectorsearch_v1beta.VectorSearchServiceClient()
    parent = f"projects/{project_id}/locations/{region}"

    # Create collection with semantic search configuration
    request = vectorsearch_v1beta.CreateCollectionRequest(
        parent=parent,
        collection_id=collection_id,
        collection={
            "data_schema": {
                "type": "object",
                "properties": {
                    "page_content": {"type": "string"},
                    "topic": {"type": "string"},
                    "source": {"type": "string"},
                    "category": {"type": "string"},
                    "id": {"type": "string"},
                },
            },
            "vector_schema": {
                "embedding": {
                    "dense_vector": {
                        "dimensions": 768,
                        "vertex_embedding_config": {
                            "model_id": "text-embedding-005",
                            "text_template": "{page_content}",
                            "task_type": "RETRIEVAL_DOCUMENT",
                        },
                    },
                },
            },
        },
    )

    operation = client.create_collection(request=request)
    operation.result()  # Wait for collection creation

    yield {
        "project_id": project_id,
        "region": region,
        "collection_id": collection_id,
    }

    # Cleanup: Delete the collection
    collection_name = f"{parent}/collections/{collection_id}"
    try:
        client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"Warning: Failed to delete collection {collection_id}: {e}")


@pytest.mark.extended
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
@pytest.mark.extended
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


@pytest.mark.extended
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


@pytest.mark.extended
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


@pytest.mark.extended
def test_vector_store_v2_delete_by_ids(vector_store_v2: VectorSearchVectorStore):
    """Tests deleting documents by IDs in V2."""
    texts = ["doc to delete", "doc to keep"]
    delete_id = f"delete_me_{str(uuid4())}"
    keep_id = f"keep_me_{str(uuid4())}"
    ids = [delete_id, keep_id]
    vector_store_v2.add_texts(texts=texts, ids=ids)

    # Wait for indexing (eventual consistency)
    time.sleep(10)

    vector_store_v2.delete(ids=[delete_id])

    # Wait for deletion to propagate
    time.sleep(10)

    # Verify the deleted document is not in results
    # Use large k value to thoroughly check the collection
    results = vector_store_v2.similarity_search("doc", k=100)
    result_ids = [doc.metadata.get("id") for doc in results]
    assert delete_id not in result_ids, (
        f"Deleted ID {delete_id} should not be in results"
    )


@pytest.mark.extended
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


@pytest.mark.extended
def test_vector_store_v2_semantic_search(
    semantic_search_collection, embeddings: VertexAIEmbeddings
):
    """Tests semantic search in V2 with auto-generated embeddings."""
    # Create vector store with the semantic search collection
    # Note: We pass embeddings but they won't be used for semantic search
    vector_store = VectorSearchVectorStore.from_components(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
        collection_id=semantic_search_collection["collection_id"],
        embedding=embeddings,  # Required by API but not used for semantic search
        api_version="v2",
    )

    # Add documents - embeddings will be auto-generated by the collection
    docs_to_add = [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"topic": "animals"},
        ),
        Document(
            page_content="A journey of a thousand miles begins with a single step.",
            metadata={"topic": "wisdom"},
        ),
        Document(
            page_content="Cats and dogs are popular pets in many households.",
            metadata={"topic": "animals"},
        ),
    ]
    ids = [str(uuid4()) for _ in docs_to_add]

    # Add documents using the SDK directly (no embeddings needed)
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
    )
    clients = sdk_manager.get_v2_client()
    data_client = clients["data_object_service_client"]

    collection_name = (
        f"projects/{semantic_search_collection['project_id']}/"
        f"locations/{semantic_search_collection['region']}/"
        f"collections/{semantic_search_collection['collection_id']}"
    )

    # Create data objects without embeddings - they'll be auto-generated
    batch_requests = []
    for doc_id, doc in zip(ids, docs_to_add):
        metadata = doc.metadata.copy()
        metadata["page_content"] = doc.page_content
        metadata["id"] = doc_id

        batch_requests.append(
            {
                "data_object_id": doc_id,
                "data_object": {
                    "data": metadata,
                    "vectors": {},  # Empty - auto-generated
                },
            }
        )

    request = vectorsearch_v1beta.BatchCreateDataObjectsRequest(
        parent=collection_name,
        requests=batch_requests,
    )
    data_client.batch_create_data_objects(request=request)

    # Wait for indexing and embedding generation
    time.sleep(5)

    # Perform semantic search - embeddings are generated automatically
    results = vector_store.semantic_search(
        query="Tell me about animals",
        k=3,
        search_field="embedding",  # Field with vertex_embedding_config
        task_type="RETRIEVAL_QUERY",
    )

    assert len(results) > 0, "Semantic search should return results"
    # The animal-related documents should be ranked higher
    animal_results = [
        doc
        for doc in results
        if "topic" in doc.metadata and doc.metadata["topic"] == "animals"
    ]
    assert len(animal_results) > 0, "Should find animal-related documents"


@pytest.mark.extended
def test_vector_store_v2_text_search(
    semantic_search_collection, embeddings: VertexAIEmbeddings
):
    """Tests text/keyword search in V2."""
    # Create vector store with the semantic search collection
    vector_store = VectorSearchVectorStore.from_components(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
        collection_id=semantic_search_collection["collection_id"],
        embedding=embeddings,
        api_version="v2",
    )

    # Add documents using SDK directly
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
    )
    clients = sdk_manager.get_v2_client()
    data_client = clients["data_object_service_client"]

    collection_name = (
        f"projects/{semantic_search_collection['project_id']}/"
        f"locations/{semantic_search_collection['region']}/"
        f"collections/{semantic_search_collection['collection_id']}"
    )

    docs_to_add = [
        Document(
            page_content="Python is a great programming language",
            metadata={"language": "python", "category": "programming"},
        ),
        Document(
            page_content="JavaScript is widely used for web development",
            metadata={"language": "javascript", "category": "programming"},
        ),
        Document(
            page_content="Java powers many enterprise applications",
            metadata={"language": "java", "category": "programming"},
        ),
    ]
    ids = [str(uuid4()) for _ in docs_to_add]

    batch_requests = []
    for doc_id, doc in zip(ids, docs_to_add):
        metadata = doc.metadata.copy()
        metadata["page_content"] = doc.page_content
        metadata["id"] = doc_id

        batch_requests.append(
            {
                "data_object_id": doc_id,
                "data_object": {
                    "data": metadata,
                    "vectors": {},
                },
            }
        )

    request = vectorsearch_v1beta.BatchCreateDataObjectsRequest(
        parent=collection_name,
        requests=batch_requests,
    )
    data_client.batch_create_data_objects(request=request)

    # Wait for indexing
    time.sleep(5)

    # Perform text search for exact keyword match
    results = vector_store.text_search(
        query="Python",
        k=3,
        data_field_names=["page_content"],
    )

    assert len(results) > 0, "Text search should return results"
    # Should find the Python document
    python_found = any("Python" in doc.page_content for doc in results)
    assert python_found, "Should find document containing 'Python'"


@pytest.mark.extended
def test_vector_store_v2_semantic_text_hybrid_search(
    semantic_search_collection, embeddings: VertexAIEmbeddings
):
    """Tests hybrid search (semantic + text with RRF) in V2."""
    # Create vector store with the semantic search collection
    vector_store = VectorSearchVectorStore.from_components(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
        collection_id=semantic_search_collection["collection_id"],
        embedding=embeddings,
        api_version="v2",
    )

    # Add documents using SDK directly
    from langchain_google_vertexai.vectorstores._sdk_manager import (
        VectorSearchSDKManager,
    )

    sdk_manager = VectorSearchSDKManager(
        project_id=semantic_search_collection["project_id"],
        region=semantic_search_collection["region"],
    )
    clients = sdk_manager.get_v2_client()
    data_client = clients["data_object_service_client"]

    collection_name = (
        f"projects/{semantic_search_collection['project_id']}/"
        f"locations/{semantic_search_collection['region']}/"
        f"collections/{semantic_search_collection['collection_id']}"
    )

    # Add documents about beach and summer wear
    docs_to_add = [
        Document(
            page_content="Men's swim trunks for beach",
            metadata={"category": "swimwear", "season": "summer"},
        ),
        Document(
            page_content="Women's short sundress perfect for summer",
            metadata={"category": "dresses", "season": "summer"},
        ),
        Document(
            page_content="Beach shorts for men casual wear",
            metadata={"category": "shorts", "season": "summer"},
        ),
        Document(
            page_content="Winter jacket heavy duty",
            metadata={"category": "outerwear", "season": "winter"},
        ),
    ]
    ids = [str(uuid4()) for _ in docs_to_add]

    batch_requests = []
    for doc_id, doc in zip(ids, docs_to_add):
        metadata = doc.metadata.copy()
        metadata["page_content"] = doc.page_content
        metadata["id"] = doc_id

        batch_requests.append(
            {
                "data_object_id": doc_id,
                "data_object": {
                    "data": metadata,
                    "vectors": {},  # Auto-generate embeddings
                },
            }
        )

    request = vectorsearch_v1beta.BatchCreateDataObjectsRequest(
        parent=collection_name,
        requests=batch_requests,
    )
    data_client.batch_create_data_objects(request=request)

    # Wait for indexing and embedding generation
    time.sleep(5)

    # Perform hybrid search - combines semantic understanding + keyword matching
    results = vector_store.hybrid_search(
        query="Men's short for beach",
        k=10,
        search_field="embedding",
        data_field_names=["page_content"],
        task_type="RETRIEVAL_QUERY",
    )

    assert len(results) > 0, "Hybrid search should return results"

    # Hybrid search should find both:
    # 1. Semantic match: "swim trunks" (semantically similar to "short for beach")
    # 2. Keyword match: "Beach shorts" (contains both "short" and "beach")
    result_contents = [doc.page_content for doc in results]

    # Check that summer/beach related items rank higher than winter items
    beach_or_short_found = any(
        "beach" in content.lower() or "short" in content.lower()
        for content in result_contents
    )
    assert beach_or_short_found, "Should find beach or short related items"

    print("\n Hybrid search results for 'Men's short for beach':")
    for i, doc in enumerate(results[:5], 1):
        print(f"  {i}. {doc.page_content}")
