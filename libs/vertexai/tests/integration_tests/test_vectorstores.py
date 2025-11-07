"""Test Vertex AI API wrapper.
Your end-user credentials would be used to make the calls (make sure you've run
`gcloud auth login` first).
Additionally in order to run the test you must have set the following environment
variables:
- PROJECT_ID: Id of the Google Cloud Project
- REGION: Region of the Bucket, Index and Endpoint
- VECTOR_SEARCH_STAGING_BUCKET: Name of a Google Cloud Storage Bucket
- INDEX_ID: Id of the Vector Search index.
- ENDPOINT_ID: Id of the Vector Search endpoint.
"""

import os
from uuid import uuid4

import pytest
import vertexai
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
    return VectorSearchSDKManager(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
    )


@pytest.fixture
def gcs_document_storage(sdk_manager: VectorSearchSDKManager) -> GCSDocumentStorage:
    bucket = sdk_manager.get_gcs_bucket(
        bucket_name=os.environ["VECTOR_SEARCH_STAGING_BUCKET"]
    )
    return GCSDocumentStorage(bucket=bucket, prefix="integration_tests")


@pytest.fixture
def gcs_document_storage_unthreaded(
    sdk_manager: VectorSearchSDKManager,
) -> GCSDocumentStorage:
    bucket = sdk_manager.get_gcs_bucket(
        bucket_name=os.environ["VECTOR_SEARCH_STAGING_BUCKET"]
    )
    return GCSDocumentStorage(bucket=bucket, prefix="integration_tests", threaded=False)


@pytest.fixture
def datastore_document_storage(
    sdk_manager: VectorSearchSDKManager,
) -> DataStoreDocumentStorage:
    ds_client = sdk_manager.get_datastore_client(namespace="integration_tests")
    return DataStoreDocumentStorage(datastore_client=ds_client)


@pytest.fixture
def embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(model_name="text-embedding-005")  # type: ignore


@pytest.fixture
def vector_store(embeddings: VertexAIEmbeddings) -> VectorSearchVectorStore:
    return VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
        gcs_bucket_name=os.environ["VECTOR_SEARCH_STAGING_BUCKET"],
        index_id=os.environ["VECTOR_SEARCH_BATCH_INDEX_ID"],
        endpoint_id=os.environ["VECTOR_SEARCH_BATCH_ENDPOINT_ID"],
        embedding=embeddings,
    )


@pytest.fixture
def vector_store_private(embeddings: VertexAIEmbeddings) -> VectorSearchVectorStore:
    return VectorSearchVectorStore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
        gcs_bucket_name=os.environ["VECTOR_SEARCH_STAGING_BUCKET"],
        index_id=os.environ["VECTOR_SEARCH_BATCH_INDEX_ID"],
        endpoint_id=os.environ["VECTOR_SEARCH_BATCH_ENDPOINT_ID"],
        private_service_connect_ip_address=os.environ[
            "PRIVATE_SERVICE_CONNECT_IP_ADDRESS"
        ],
        embedding=embeddings,
    )


@pytest.fixture
def datastore_vector_store(
    embeddings: VertexAIEmbeddings,
) -> VectorSearchVectorStoreDatastore:
    return VectorSearchVectorStoreDatastore.from_components(
        project_id=os.environ["PROJECT_ID"],
        region=os.environ.get("REGION", "us-central1"),
        index_id=os.environ["VECTOR_SEARCH_STREAM_INDEX_ID"],
        endpoint_id=os.environ["VECTOR_SEARCH_STREAM_ENDPOINT_ID"],
        embedding=embeddings,
        stream_update=True,
    )


@pytest.mark.extended
def test_vector_search_sdk_manager(sdk_manager: VectorSearchSDKManager) -> None:
    gcs_client = sdk_manager.get_gcs_client()
    assert isinstance(gcs_client, storage.Client)

    gcs_bucket = sdk_manager.get_gcs_bucket(os.environ["VECTOR_SEARCH_STAGING_BUCKET"])
    assert isinstance(gcs_bucket, storage.Bucket)

    index = sdk_manager.get_index(index_id=os.environ["VECTOR_SEARCH_BATCH_INDEX_ID"])
    assert isinstance(index, MatchingEngineIndex)

    endpoint = sdk_manager.get_endpoint(
        endpoint_id=os.environ["VECTOR_SEARCH_BATCH_ENDPOINT_ID"]
    )
    assert isinstance(endpoint, MatchingEngineIndexEndpoint)


@pytest.mark.extended
@pytest.mark.parametrize("n_threads", ["-1", -1, 51, "100"])
def test_gcs_document_storage_invalid_user_input(
    sdk_manager: VectorSearchSDKManager, n_threads: int
) -> None:
    bucket = sdk_manager.get_gcs_bucket(os.environ["VECTOR_SEARCH_STAGING_BUCKET"])
    with pytest.raises(ValueError) as excinfo:
        GCSDocumentStorage(
            bucket=bucket,
            prefix="integration_tests",
            threaded=True,
            n_threads=n_threads,
        )
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.extended
@pytest.mark.parametrize("n_threads", ["1", 1, 50])
def test_gcs_document_storage_valid_user_input(
    sdk_manager: VectorSearchSDKManager, n_threads: int
) -> None:
    bucket = sdk_manager.get_gcs_bucket(os.environ["VECTOR_SEARCH_STAGING_BUCKET"])
    doc_store = GCSDocumentStorage(
        bucket=bucket, prefix="integration_tests", threaded=True, n_threads=n_threads
    )
    assert isinstance(doc_store, GCSDocumentStorage)


@pytest.mark.extended
@pytest.mark.parametrize(
    "storage_class",
    [
        "gcs_document_storage",
        "gcs_document_storage_unthreaded",
        "datastore_document_storage",
    ],
)
def test_document_storage(
    storage_class: str,
    request: pytest.FixtureRequest,
) -> None:
    document_storage: DocumentStorage = request.getfixturevalue(storage_class)

    weirdly_encoded_texts = [
        "ユーザー別サイト",
        "简体中文",
        "크로스 플랫폼으로",
        "מדורים מבוקשים",
        "أفضل البحوث",
        "Σὲ γνωρίζω ἀπὸ",
        "Десятую Международную",
        "แผ่นดินฮั่นเสื่อมโทรมแสนสังเวช",
        "∮ E⋅da = Q, n → ∞, ∑ f(i) = ∏ g(i)",
        "français langue étrangère",
        "mañana olé y vamos Messi!",
    ]

    N = 10
    documents = [
        Document(
            page_content=f"Text content of document {i}: {text}",
            metadata={"index": i, "nested": {"a": i, "b": str(uuid4())}},
        )
        for i, text in enumerate(weirdly_encoded_texts * N)
    ]
    ids = [str(uuid4()) for i in range(N * len(weirdly_encoded_texts))]

    # Test batch storage and retrieval
    document_storage.mset(list(zip(ids, documents, strict=False)))
    retrieved_documents = document_storage.mget(ids)

    for og_document, retrieved_document in zip(
        documents, retrieved_documents, strict=False
    ):
        assert og_document == retrieved_document

    # Test key yielding
    keys = list(document_storage.yield_keys())
    assert all(id in keys for id in ids)

    # Test deletion
    document_storage.mdelete(ids)
    assert all(item is None for item in document_storage.mget(ids))


@pytest.mark.extended
def test_public_endpoint_vector_searcher(
    embeddings: VertexAIEmbeddings, sdk_manager: VectorSearchSDKManager
) -> None:
    vertexai.init(api_transport="grpc")
    index = sdk_manager.get_index(os.environ["VECTOR_SEARCH_BATCH_INDEX_ID"])
    endpoint = sdk_manager.get_endpoint(os.environ["VECTOR_SEARCH_BATCH_ENDPOINT_ID"])

    searcher = VectorSearchSearcher(endpoint=endpoint, index=index)

    texts = ["What's your favourite animal", "What's your favourite city"]

    embeddings_vector = embeddings.embed_documents(texts=texts)

    matching_neighbors_list = searcher.find_neighbors(embeddings=embeddings_vector, k=4)

    assert len(matching_neighbors_list) == 2


@pytest.mark.extended
@pytest.mark.parametrize(
    "vector_store_class", ["vector_store", "datastore_vector_store"]
)
def test_vector_store(vector_store_class: str, request: pytest.FixtureRequest) -> None:
    vertexai.init(api_transport="grpc")
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


@pytest.mark.extended
@pytest.mark.parametrize(
    "vector_store_class", ["vector_store", "datastore_vector_store"]
)
def test_vector_store_hybrid_search(
    vector_store_class: str,
    request: pytest.FixtureRequest,
    embeddings: VertexAIEmbeddings,
) -> None:
    vertexai.init(api_transport="grpc")
    vector_store: VectorSearchVectorStore = request.getfixturevalue(vector_store_class)

    query = "What are your favourite animals?"
    embedding = embeddings.embed_query(query)
    sparse_embedding: dict[str, list[int] | list[float]] = {
        "values": [0.5, 0.7],
        "dimensions": [2, 4],
    }

    docs_with_scores = vector_store.similarity_search_by_vector_with_score(
        embedding=embedding, sparse_embedding=sparse_embedding, k=1
    )
    assert len(docs_with_scores) == 1
    for doc, scores in docs_with_scores:
        assert isinstance(doc, Document)
        assert isinstance(scores, dict)
        assert "dense_score" in scores
        assert "sparse_score" in scores
        assert isinstance(scores["dense_score"], float)
        assert isinstance(scores["sparse_score"], float)


@pytest.mark.extended
@pytest.mark.parametrize("vector_store_class", ["datastore_vector_store"])
def test_add_texts_with_embeddings(
    vector_store_class: str,
    request: pytest.FixtureRequest,
    embeddings: VertexAIEmbeddings,
) -> None:
    vector_store: VectorSearchVectorStore = request.getfixturevalue(vector_store_class)

    texts = ["my favourite animal is the elephant", "my favourite animal is the lion"]
    ids = ["idx1", "idx2"]
    embs = embeddings.embed_documents(texts)
    ids1 = vector_store.add_texts_with_embeddings(
        texts=texts, embeddings=embs, ids=ids, is_complete_overwrite=True
    )
    assert len(ids1) == 2

    sparse_embeddings: list[dict[str, list[int] | list[float]]] = [
        {"values": [0.5, 0.7], "dimensions": [2, 4]}
    ] * 2
    ids2 = vector_store.add_texts_with_embeddings(
        texts=texts,
        embeddings=embs,
        sparse_embeddings=sparse_embeddings,
        ids=ids,
        is_complete_overwrite=True,
    )
    assert ids == ids1 == ids2


@pytest.mark.extended
@pytest.mark.skip("rebuild the index with restricts")
@pytest.mark.parametrize(
    "vector_store_class",
    [
        "vector_store",
        # "datastore_vector_store" Waiting for the bug to be fixed as its stream
    ],
)
def test_vector_store_filtering(
    vector_store_class: str, request: pytest.FixtureRequest
) -> None:
    vector_store: VectorSearchVectorStore = request.getfixturevalue(vector_store_class)
    documents = vector_store.similarity_search(
        "I want some pants",
        filter=[Namespace(name="color", allow_tokens=["blue"])],
        numeric_filter=[NumericNamespace(name="price", value_float=20.0, op="LESS")],
    )

    assert len(documents) > 0
    assert all(document.metadata["color"] == "blue" for document in documents)
    assert all(document.metadata["price"] < 20.0 for document in documents)


@pytest.mark.long
def test_vector_store_update_index(
    vector_store: VectorSearchVectorStore, sample_documents: list[Document]
) -> None:
    vector_store.add_documents(documents=sample_documents, is_complete_overwrite=True)


@pytest.mark.extended
def test_vector_store_stream_update_index(
    datastore_vector_store: VectorSearchVectorStoreDatastore,
    sample_documents: list[Document],
) -> None:
    datastore_vector_store.add_documents(
        documents=sample_documents, is_complete_overwrite=True
    )


@pytest.fixture
def sample_documents() -> list[Document]:
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
