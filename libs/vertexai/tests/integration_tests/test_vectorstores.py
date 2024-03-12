"""Test Vertex AI API wrapper.
Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
Additionally in order to run the test you must have set the following environment 
variables:
- PROJECT_ID: Id of the Google Cloud Project
- REGION: Region of the Bucket, Index and Endpoint
- GCS_BUCKET_NAME: Name of a Google Cloud Storage Bucket
- INDEX_ID: Id of the Vector Search index.
- ENDPOINT_ID: Id of the Vector Search endpoint.
If required to run slow tests, environment variable 'RUN_SLOW_TESTS' must be set
"""

import os
from uuid import uuid4

import pytest
from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from langchain_core.documents import Document

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores._document_storage import (
    DataStoreDocumentStorage,
    DocumentStorage,
    GCSDocumentStorage,
)
from langchain_google_vertexai.vectorstores._sdk_manager import VectorSearchSDKManager
from langchain_google_vertexai.vectorstores._searcher import (
    VectorSearchSearcher,
)
from langchain_google_vertexai.vectorstores.vectorstores import VectorSearchVectorStore


@pytest.fixture
def sdk_manager() -> VectorSearchSDKManager:
    sdk_manager = VectorSearchSDKManager(
        project_id=os.environ["PROJECT_ID"], region=os.environ["REGION"]
    )
    return sdk_manager


@pytest.fixture
def gcs_document_storage(sdk_manager: VectorSearchSDKManager) -> GCSDocumentStorage:
    bucket = sdk_manager.get_gcs_bucket(bucket_name=os.environ["GCS_BUCKET_NAME"])
    return GCSDocumentStorage(bucket=bucket, prefix="integration_tests")


@pytest.fixture
def datastore_document_storage(
    sdk_manager: VectorSearchSDKManager,
) -> DataStoreDocumentStorage:
    ds_client = sdk_manager.get_datastore_client(namespace="integration_tests")
    return DataStoreDocumentStorage(datastore_client=ds_client)


@pytest.mark.extended
def test_vector_search_sdk_manager(sdk_manager: VectorSearchSDKManager):
    gcs_client = sdk_manager.get_gcs_client()
    assert isinstance(gcs_client, storage.Client)

    gcs_bucket = sdk_manager.get_gcs_bucket(os.environ["GCS_BUCKET_NAME"])
    assert isinstance(gcs_bucket, storage.Bucket)

    index = sdk_manager.get_index(index_id=os.environ["INDEX_ID"])
    assert isinstance(index, MatchingEngineIndex)

    endpoint = sdk_manager.get_endpoint(endpoint_id=os.environ["ENDPOINT_ID"])
    assert isinstance(endpoint, MatchingEngineIndexEndpoint)


@pytest.mark.extended
@pytest.mark.parametrize(
    "storage_class", ["gcs_document_storage", "datastore_document_storage"]
)
def test_document_storage(
    sdk_manager: VectorSearchSDKManager,
    storage_class: str,
    request: pytest.FixtureRequest,
):
    document_storage: DocumentStorage = request.getfixturevalue(storage_class)

    N = 10
    documents = [
        Document(
            page_content=f"Text content of document {i}",
            metadata={"index": i, "nested": {"a": i, "b": str(uuid4())}},
        )
        for i in range(N)
    ]
    ids = [str(uuid4()) for i in range(N)]

    # Test individual retrieval
    for id, document in zip(ids, documents):
        document_storage.store_by_id(document_id=id, document=document)
        retrieved = document_storage.get_by_id(document_id=id)
        assert document == retrieved

    # Test batch regtrieval
    document_storage.batch_store_by_id(ids, documents)
    retrieved_documents = document_storage.batch_get_by_id(ids)

    for og_document, retrieved_document in zip(documents, retrieved_documents):
        assert og_document == retrieved_document


@pytest.mark.extended
def test_public_endpoint_vector_searcher(sdk_manager: VectorSearchSDKManager):
    index = sdk_manager.get_index(os.environ["INDEX_ID"])
    endpoint = sdk_manager.get_endpoint(os.environ["ENDPOINT_ID"])
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    searcher = VectorSearchSearcher(endpoint=endpoint, index=index)

    texts = ["What's your favourite animal", "What's your favourite city"]

    embeddings_vector = embeddings.embed_documents(texts=texts)

    matching_neighbors_list = searcher.find_neighbors(embeddings=embeddings_vector, k=4)

    assert len(matching_neighbors_list) == 2


@pytest.mark.extended
def test_vector_store():
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    vector_store = VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        index_id=os.environ["INDEX_ID"],
        endpoint_id=os.environ["ENDPOINT_ID"],
        embedding=embeddings,
    )

    assert isinstance(vector_store, VectorSearchVectorStore)

    query = "What are your favourite animals?"
    docs_with_scores = vector_store.similarity_search_with_score(query, k=1)
    assert len(docs_with_scores) == 1
    for doc, score in docs_with_scores:
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    docs = vector_store.similarity_search(query, k=2)
    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, Document)


@pytest.mark.extended
def test_vector_store_update_index():
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    vector_store = VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        index_id=os.environ["INDEX_ID"],
        endpoint_id=os.environ["ENDPOINT_ID"],
        embedding=embeddings,
    )

    vector_store.add_texts(
        texts=[
            "Lions are my favourite animals",
            "There are two apples on the table",
            "Today is raining a lot in Madrid",
        ]
    )
