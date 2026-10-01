"""Regression tests for BigQuery vector store filters."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Union
from unittest.mock import MagicMock

import pytest
from google.cloud import bigquery

from langchain_google_community.bq_storage_vectorstores.bigquery import (
    BigQueryVectorStore,
)


def _store() -> BigQueryVectorStore:
    store = BigQueryVectorStore.model_construct()
    store.table_schema = {
        "tenant_id": "STRING",
        "priority": "INTEGER",
        "created": "DATE",
        "start_time": "TIME",
        "updated": "TIMESTAMP",
        "amount": "NUMERIC",
    }
    store._bq_client = MagicMock()
    store.project_id = "project"
    store.dataset_name = "dataset"
    store.table_name = "vectors"
    return store


def test_filter_values_bound_in_get_documents() -> None:
    store = _store()
    payload = "legit' OR tenant_id = 'victim'; DROP TABLE x --"
    store.get_documents(ids=["one"], filter={"tenant_id": payload})
    query, kwargs = store._bq_client.query.call_args
    assert payload not in query[0]
    assert "`tenant_id` = @filter_0" in query[0]
    parameters = kwargs["job_config"].query_parameters
    assert isinstance(parameters[0], bigquery.ArrayQueryParameter)
    assert parameters[1].value == payload


def test_filter_values_bound_in_search() -> None:
    store = _store()
    store._search_embeddings([[0.1]], filter={"priority": 1})
    query, kwargs = store._bq_client.query.call_args
    assert "`priority` = @filter_0" in query[0]
    assert [parameter.name for parameter in kwargs["job_config"].query_parameters] == [
        "filter_0",
        "emb_0",
    ]


def test_raw_sql_requires_opt_in() -> None:
    store = _store()
    with pytest.raises(ValueError, match="allow_raw_sql_filters"):
        store.get_documents(filter='tenant_id="trusted"')
    store.allow_raw_sql_filters = True
    store.get_documents(filter='tenant_id="trusted"')
    query, kwargs = store._bq_client.query.call_args
    assert 'tenant_id="trusted"' in query[0]
    assert kwargs["job_config"].query_parameters == []


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("tenant_id", 2024, "2024"),
        ("created", "2024-01-01", "2024-01-01"),
        ("created", date(2024, 1, 1), "2024-01-01"),
        ("start_time", "09:30:00", "09:30:00"),
        ("start_time", time(9, 30), "09:30:00"),
        ("updated", datetime(2024, 1, 1), "2024-01-01 00:00:00+00:00"),
        ("amount", Decimal("1.25"), "1.25"),
    ],
)
def test_filter_type_serialization(column: str, value: Any, expected: str) -> None:
    store = _store()
    store.get_documents(filter={column: value})
    _, kwargs = store._bq_client.query.call_args
    parameter = kwargs["job_config"].query_parameters[0]
    assert parameter.to_api_repr()["parameterValue"]["value"] == expected


def test_integral_scalar_filter() -> None:
    import numpy as np

    store = _store()
    store.get_documents(filter={"priority": np.int64(1)})
    _, kwargs = store._bq_client.query.call_args
    assert kwargs["job_config"].query_parameters[0].value == 1


@pytest.mark.parametrize(
    "filter_value",
    ["1=1", {"priority": "1 OR 1=1 --"}, {"tenant_id`: OR TRUE --": "x"}],
)
def test_unsafe_filters_rejected(filter_value: Union[dict[str, Any], str]) -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.get_documents(filter=filter_value)
    store._bq_client.query.assert_not_called()
