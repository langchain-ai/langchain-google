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
"""

import os
from typing import List
from uuid import uuid4

import pytest
from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)
from langchain_core.documents import Document

from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores._sdk_manager import VectorSearchSDKManager
from langchain_google_vertexai.vectorstores._searcher import (
    VectorSearchSearcher,
)
from langchain_google_vertexai.vectorstores.document_storage import (
    DataStoreDocumentStorage,
    DocumentStorage,
    GCSDocumentStorage,
)
from langchain_google_vertexai.vectorstores.vectorstores import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)


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


@pytest.fixture
def vector_store() -> VectorSearchVectorStore:
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    vector_store = VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        index_id=os.environ["INDEX_ID"],
        endpoint_id=os.environ["ENDPOINT_ID"],
        embedding=embeddings,
    )

    return vector_store


@pytest.fixture
def vector_store_private() -> VectorSearchVectorStore:
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    vector_store_private = VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        index_id=os.environ["INDEX_ID"],
        endpoint_id=os.environ["ENDPOINT_ID"],
        private_service_connect_ip_address=os.environ[
            "PRIVATE_SERVICE_CONNECT_IP_ADDRESS"
        ],
        embedding=embeddings,
    )

    return vector_store_private


@pytest.fixture
def datastore_vector_store() -> VectorSearchVectorStoreDatastore:
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko-default")

    vector_store = VectorSearchVectorStoreDatastore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ["REGION"],
        index_id=os.environ["STREAM_INDEX_ID_DATASTORE"],
        endpoint_id=os.environ["STREAM_ENDPOINT_ID_DATASTORE"],
        embedding=embeddings,
        stream_update=True,
    )

    return vector_store


@pytest.mark.xfail(reason="investigating")
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


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
@pytest.mark.parametrize(
    "storage_class", ["gcs_document_storage", "datastore_document_storage"]
)
def test_document_storage(
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

    # Test batch storage and retrieval
    document_storage.mset(list(zip(ids, documents)))
    retrieved_documents = document_storage.mget(ids)

    for og_document, retrieved_document in zip(documents, retrieved_documents):
        assert og_document == retrieved_document

    # Test key yielding
    keys = list(document_storage.yield_keys())
    assert all(id in keys for id in ids)

    # Test deletion
    document_storage.mdelete(ids)
    assert all(item is None for item in document_storage.mget(ids))


@pytest.mark.xfail(reason="investigating")
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


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
@pytest.mark.parametrize(
    "vector_store_class", ["vector_store", "datastore_vector_store"]
)
def test_vector_store(vector_store_class: str, request: pytest.FixtureRequest):
    vector_store: VectorSearchVectorStore = request.getfixturevalue(vector_store_class)

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


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
@pytest.mark.parametrize(
    "vector_store_class",
    [
        "vector_store",
        # "datastore_vector_store" Waiting for the bug to be fixed as its stream
    ],
)
def test_vector_store_filtering(
    vector_store_class: str, request: pytest.FixtureRequest
):
    vector_store: VectorSearchVectorStore = request.getfixturevalue(vector_store_class)
    documents = vector_store.similarity_search(
        "I want some pants",
        filter=[Namespace(name="color", allow_tokens=["blue"])],
        numeric_filter=[NumericNamespace(name="price", value_float=20.0, op="LESS")],
    )

    assert len(documents) > 0
    assert all(document.metadata["color"] == "blue" for document in documents)
    assert all(document.metadata["price"] < 20.0 for document in documents)


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
def test_vector_store_update_index(
    vector_store: VectorSearchVectorStore, sample_documents: List[Document]
):
    vector_store.add_documents(documents=sample_documents, is_complete_overwrite=True)


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
def test_vector_store_stream_update_index(
    datastore_vector_store: VectorSearchVectorStoreDatastore,
    sample_documents: List[Document],
):
    datastore_vector_store.add_documents(
        documents=sample_documents, is_complete_overwrite=True
    )


@pytest.fixture
def sample_documents() -> List[Document]:
    record_data = [
        {
            "description": "A versatile pair of dark-wash denim jeans."
            "Made from durable cotton with a classic straight-leg cut, these jeans"
            " transition easily from casual days to dressier occasions.",
            "price": 65.00,
            "color": "blue",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A lightweight linen button-down shirt in a crisp white."
            " Perfect for keeping cool with breathable fabric and a relaxed fit.",
            "price": 34.99,
            "color": "white",
            "season": ["summer", "spring"],
        },
        {
            "description": "A soft, chunky knit sweater in a vibrant forest green. "
            "The oversized fit and cozy wool blend make this ideal for staying warm "
            "when the temperature drops.",
            "price": 89.99,
            "color": "green",
            "season": ["fall", "winter"],
        },
        {
            "description": "A classic crewneck t-shirt in a soft, heathered blue. "
            "Made from comfortable cotton jersey, this t-shirt is a wardrobe essential "
            "that works for every season.",
            "price": 19.99,
            "color": "blue",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A flowing midi-skirt in a delicate floral print. "
            "Lightweight and airy, this skirt adds a touch of feminine style "
            "to warmer days.",
            "price": 45.00,
            "color": "white",
            "season": ["spring", "summer"],
        },
        {
            "description": "A pair of tailored black trousers in a comfortable stretch "
            "fabric. Perfect for work or dressier events, these trousers provide a"
            " sleek, polished look.",
            "price": 59.99,
            "color": "black",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A cozy fleece hoodie in a neutral heather grey.  "
            "This relaxed sweatshirt is perfect for casual days or layering when the "
            "weather turns chilly.",
            "price": 39.99,
            "color": "grey",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A bright yellow raincoat with a playful polka dot pattern. "
            "This waterproof jacket will keep you dry and add a touch of cheer to "
            "rainy days.",
            "price": 75.00,
            "color": "yellow",
            "season": ["spring", "fall"],
        },
        {
            "description": "A pair of comfortable khaki chino shorts. These versatile "
            "shorts are a summer staple, perfect for outdoor adventures or relaxed"
            " weekends.",
            "price": 34.99,
            "color": "khaki",
            "season": ["summer"],
        },
        {
            "description": "A bold red cocktail dress with a flattering A-line "
            "silhouette. This statement piece is made from a luxurious satin fabric, "
            "ensuring a head-turning look.",
            "price": 125.00,
            "color": "red",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of classic white sneakers crafted from smooth "
            "leather. These timeless shoes offer a clean and polished look, perfect "
            "for everyday wear.",
            "price": 79.99,
            "color": "white",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A chunky cable-knit scarf in a rich burgundy color. "
            "Made from a soft wool blend, this scarf will provide warmth and a touch "
            "of classic style to cold-weather looks.",
            "price": 45.00,
            "color": "burgundy",
            "season": ["fall", "winter"],
        },
        {
            "description": "A lightweight puffer vest in a vibrant teal hue. "
            "This versatile piece adds a layer of warmth without bulk, transitioning"
            " perfectly between seasons.",
            "price": 65.00,
            "color": "teal",
            "season": ["fall", "spring"],
        },
        {
            "description": "A pair of high-waisted leggings in a sleek black."
            " Crafted from a moisture-wicking fabric with plenty of stretch, "
            "these leggings are perfect for workouts or comfortable athleisure style.",
            "price": 49.99,
            "color": "black",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A denim jacket with a faded wash and distressed details. "
            "This wardrobe staple adds a touch of effortless cool to any outfit.",
            "price": 79.99,
            "color": "blue",
            "season": ["fall", "spring", "summer"],
        },
        {
            "description": "A woven straw sunhat with a wide brim. This stylish "
            "accessory provides protection from the sun while adding a touch of "
            "summery elegance.",
            "price": 32.00,
            "color": "beige",
            "season": ["summer"],
        },
        {
            "description": "A graphic tee featuring a vintage band logo. "
            "Made from a soft cotton blend, this casual tee adds a touch of "
            "personal style to everyday looks.",
            "price": 24.99,
            "color": "white",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of well-tailored dress pants in a neutral grey. "
            "Made from a wrinkle-resistant blend, these pants look sharp and "
            "professional for workwear or formal occasions.",
            "price": 69.99,
            "color": "grey",
            "season": ["fall", "winter", "summer", "spring"],
        },
        {
            "description": "A pair of classic leather ankle boots in a rich brown hue."
            " Featuring a subtle stacked heel and sleek design, these boots are perfect"
            " for elevating outfits in cooler seasons.",
            "price": 120.00,
            "color": "brown",
            "season": ["fall", "winter", "spring"],
        },
        {
            "description": "A vibrant swimsuit with a bold geometric pattern. This fun "
            "and eye-catching piece is perfect for making a splash by the pool or at "
            "the beach.",
            "price": 55.00,
            "color": "multicolor",
            "season": ["summer"],
        },
    ]

    documents = []
    for record in record_data:
        record = record.copy()
        page_content = record.pop("description")
        if isinstance(page_content, str):
            metadata = {**record}
            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)

    return documents
