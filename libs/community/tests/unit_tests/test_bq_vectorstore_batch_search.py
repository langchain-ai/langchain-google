"""Unit tests for `BigQueryVectorStore.batch_search` result shaping."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryVectorStore,
)


def _store_with_rows(rows: list[dict[str, Any]]) -> tuple[BigQueryVectorStore, Any]:
    """A store whose validators never ran, with the BigQuery calls stubbed."""
    store = BigQueryVectorStore.model_construct(
        embedding=MagicMock(),
        project_id="p",
        dataset_name="d",
        table_name="t",
        location="us",
        content_field="content",
        embedding_field="embedding",
        embedding_dimension=2,
    )
    store._bq_client = MagicMock()
    store._bq_client.query.return_value = rows
    store._create_temp_bq_table = MagicMock(return_value="p.d_temp.tmp")  # type: ignore[method-assign]
    store._create_search_query = MagicMock(return_value="SELECT 1")  # type: ignore[method-assign]
    return store, store._create_search_query


ROWS = [
    {"content": "apple", "score": 0.9, "embedding": [1.0, 0.0], "row_num": 1},
    {"content": "pear", "score": 0.8, "embedding": [0.0, 1.0], "row_num": 2},
]


def test_batch_search_default_returns_docs_and_scores() -> None:
    store, _ = _store_with_rows(ROWS)

    results = store.batch_search(embeddings=[[1.0, 0.0]], k=2)

    assert results == [
        [
            [Document(page_content="apple", metadata={"score": 0.9}), 0.9],
            [Document(page_content="pear", metadata={"score": 0.8}), 0.8],
        ]
    ]


def test_batch_search_with_embeddings_includes_vectors_and_keeps_column() -> None:
    store, search_query = _store_with_rows(ROWS)

    results = store.batch_search(embeddings=[[1.0, 0.0]], k=2, with_embeddings=True)

    # The embedding column must not be excluded from the query in this mode.
    assert search_query.call_args.kwargs["fields_to_exclude"] == []
    assert results[0][0] == [
        Document(page_content="apple", metadata={"score": 0.9}),
        0.9,
        [1.0, 0.0],
    ]


def test_batch_search_docs_only() -> None:
    store, search_query = _store_with_rows(ROWS)

    results = store.batch_search(embeddings=[[1.0, 0.0]], k=2, with_scores=False)

    assert search_query.call_args.kwargs["fields_to_exclude"] == ["embedding"]
    assert results == [
        [
            Document(page_content="apple", metadata={"score": 0.9}),
            Document(page_content="pear", metadata={"score": 0.8}),
        ]
    ]


def test_batch_search_embeddings_without_scores() -> None:
    store, _ = _store_with_rows(ROWS)

    results = store.batch_search(
        embeddings=[[1.0, 0.0]], k=2, with_scores=False, with_embeddings=True
    )

    assert results[0][1] == [
        Document(page_content="pear", metadata={"score": 0.8}),
        [0.0, 1.0],
    ]


def test_batch_search_requires_exactly_one_input() -> None:
    store, _ = _store_with_rows(ROWS)

    with pytest.raises(ValueError, match="At least one"):
        store.batch_search()
    with pytest.raises(ValueError, match="Only one parameter"):
        store.batch_search(embeddings=[[1.0, 0.0]], queries=["apple"])
