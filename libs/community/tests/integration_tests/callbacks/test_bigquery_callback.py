"""Integration tests for `BigQueryCallbackHandler`."""

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List, Tuple, cast

import google.auth
import pytest
from google.cloud import bigquery  # type: ignore[attr-defined]
from google.cloud.exceptions import NotFound
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import Generation, LLMResult

from langchain_google_community.callbacks import (
    AsyncBigQueryCallbackHandler,
    BigQueryCallbackHandler,
)


def _get_unique_table_name(prefix: str) -> str:
    """Get a unique table name that doesn't exist."""
    _, project_id = google.auth.default()
    dataset_id = os.environ.get("BIGQUERY_LOGGING_DATASET_ID")
    if not dataset_id:
        pytest.skip("BIGQUERY_LOGGING_DATASET_ID not set")

    client = bigquery.Client(project=project_id)
    for _ in range(10):
        table_id = f"{prefix}_{str(uuid.uuid4()).replace('-', '_')}"
        try:
            client.get_table(f"{project_id}.{dataset_id}.{table_id}")
        except NotFound:
            return table_id
    raise ValueError("Could not generate a unique table name.")


def _check_bq_logs(
    client: bigquery.Client,
    table_id: str,
    expected_rows: List[Dict[str, Any]],
) -> None:
    """Check if the logs were written to BigQuery correctly."""
    # Give BQ time to stream the data.
    time.sleep(15)

    # Query all relevant fields, excluding timestamp
    query = f"""
        SELECT
            event_type,
            agent,
            session_id,
            invocation_id,
            user_id,
            trace_id,
            span_id,
            parent_span_id,
            content,
            attributes,
            status,
            error_message,
            is_truncated
        FROM `{table_id}`
    """
    try:
        query_job = client.query(query)
        rows = [dict(row) for row in query_job.result()]
    except Exception as e:
        pytest.fail(f"BigQuery query failed: {e}")

    # Parse JSON strings in 'content' for robust comparison
    for row in rows:
        if row.get("content") and "summary" in row["content"]:
            try:
                row["content"]["summary"] = json.loads(row["content"]["summary"])
            except (json.JSONDecodeError, TypeError):
                pass
    for row in expected_rows:
        if row.get("content") and "summary" in row["content"]:
            try:
                row["content"]["summary"] = json.loads(row["content"]["summary"])
            except (json.JSONDecodeError, TypeError):
                pass

    sorted_actual = sorted(rows, key=lambda x: x["invocation_id"])
    sorted_expected = sorted(expected_rows, key=lambda x: x["invocation_id"])
    assert sorted_actual == sorted_expected


@pytest.fixture(scope="module")
def expected_data() -> Tuple[List[Tuple[uuid.UUID, uuid.UUID]], List[Dict[str, Any]]]:
    """Provides test data and expected rows for the callback handlers."""
    ids = [(uuid.UUID(int=i), uuid.UUID(int=i + 1000)) for i in range(16)]
    tags = ["test-tag"]

    expected_rows: List[Dict[str, Any]] = [
        {
            "event_type": "LLM_REQUEST",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[0][0]),
            "user_id": None,
            "trace_id": str(ids[0][0]),
            "span_id": str(ids[0][0]),
            "parent_span_id": str(ids[0][1]),
            "content": {"summary": "Hello world"},
            "attributes": {"tags": tags, "model": "TestModel"},
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "LLM_RESPONSE",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[1][0]),
            "user_id": None,
            "trace_id": str(ids[1][0]),
            "span_id": str(ids[1][0]),
            "parent_span_id": str(ids[1][1]),
            "content": {"summary": "Hello there!"},
            "attributes": {"usage": cast(Dict[str, Any], {})},
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "LLM_REQUEST",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[2][0]),
            "user_id": None,
            "trace_id": str(ids[2][0]),
            "span_id": str(ids[2][0]),
            "parent_span_id": str(ids[2][1]),
            "content": {"summary": "Hello"},
            "attributes": {"tags": tags, "model": "TestModel"},
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "LLM_ERROR",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[3][0]),
            "user_id": None,
            "trace_id": str(ids[3][0]),
            "span_id": str(ids[3][0]),
            "parent_span_id": str(ids[3][1]),
            "content": {"summary": "None"},
            "attributes": None,
            "status": "ERROR",
            "error_message": "LLM Error",
            "is_truncated": False,
        },
        {
            "event_type": "CHAIN_START",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[4][0]),
            "user_id": None,
            "trace_id": str(ids[4][0]),
            "span_id": str(ids[4][0]),
            "parent_span_id": str(ids[4][1]),
            "content": {"summary": '{"input": "test"}'},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "CHAIN_END",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[5][0]),
            "user_id": None,
            "trace_id": str(ids[5][0]),
            "span_id": str(ids[5][0]),
            "parent_span_id": str(ids[5][1]),
            "content": {"summary": '{"output": "test"}'},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "TOOL_STARTING",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[6][0]),
            "user_id": None,
            "trace_id": str(ids[6][0]),
            "span_id": str(ids[6][0]),
            "parent_span_id": str(ids[6][1]),
            "content": {"summary": "search"},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "AGENT_ACTION",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[7][0]),
            "user_id": None,
            "trace_id": str(ids[7][0]),
            "span_id": str(ids[7][0]),
            "parent_span_id": str(ids[7][1]),
            "content": {"summary": '{"tool": "test_tool", "input": "test_input"}'},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "AGENT_FINISH",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[8][0]),
            "user_id": None,
            "trace_id": str(ids[8][0]),
            "span_id": str(ids[8][0]),
            "parent_span_id": str(ids[8][1]),
            "content": {"summary": '{"output": {"output": "test"}}'},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "TOOL_COMPLETED",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[9][0]),
            "user_id": None,
            "trace_id": str(ids[9][0]),
            "span_id": str(ids[9][0]),
            "parent_span_id": str(ids[9][1]),
            "content": {"summary": "tool output"},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "TOOL_ERROR",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[10][0]),
            "user_id": None,
            "trace_id": str(ids[10][0]),
            "span_id": str(ids[10][0]),
            "parent_span_id": str(ids[10][1]),
            "content": {"summary": "None"},
            "attributes": None,
            "status": "ERROR",
            "error_message": "Tool Error",
            "is_truncated": False,
        },
        {
            "event_type": "CHAIN_ERROR",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[11][0]),
            "user_id": None,
            "trace_id": str(ids[11][0]),
            "span_id": str(ids[11][0]),
            "parent_span_id": str(ids[11][1]),
            "content": {"summary": "None"},
            "attributes": None,
            "status": "ERROR",
            "error_message": "Chain Error",
            "is_truncated": False,
        },
        {
            "event_type": "RETRIEVER_START",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[12][0]),
            "user_id": None,
            "trace_id": str(ids[12][0]),
            "span_id": str(ids[12][0]),
            "parent_span_id": str(ids[12][1]),
            "content": {"summary": "retriever query"},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "RETRIEVER_END",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[13][0]),
            "user_id": None,
            "trace_id": str(ids[13][0]),
            "span_id": str(ids[13][0]),
            "parent_span_id": str(ids[13][1]),
            "content": {
                "summary": (
                    '[{"page_content": "doc content", "metadata": {}, '
                    '"type": "Document", "id": null}]'
                )
            },
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
        {
            "event_type": "RETRIEVER_ERROR",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[14][0]),
            "user_id": None,
            "trace_id": str(ids[14][0]),
            "span_id": str(ids[14][0]),
            "parent_span_id": str(ids[14][1]),
            "content": {"summary": "None"},
            "attributes": None,
            "status": "ERROR",
            "error_message": "Retriever Error",
            "is_truncated": False,
        },
        {
            "event_type": "TEXT",
            "agent": None,
            "session_id": None,
            "invocation_id": str(ids[15][0]),
            "user_id": None,
            "trace_id": str(ids[15][0]),
            "span_id": str(ids[15][0]),
            "parent_span_id": str(ids[15][1]),
            "content": {"summary": "some text"},
            "attributes": None,
            "status": "OK",
            "error_message": None,
            "is_truncated": False,
        },
    ]
    return ids, expected_rows


@pytest.mark.extended
def test_bigquery_callback_handler(expected_data: tuple) -> None:
    """Test `BigQueryCallbackHandler`."""  # type: ignore
    project_id = os.environ.get("PROJECT_ID")
    dataset_id = os.environ.get("BIGQUERY_LOGGING_DATASET_ID")
    if not project_id or not dataset_id:
        pytest.skip(
            "PROJECT_ID or BIGQUERY_LOGGING_DATASET_ID environment variables not set"
        )

    table_prefix = os.environ.get("BIGQUERY_LOGGING_TABLE_PREFIX", "langchain")
    table_id = _get_unique_table_name(table_prefix)

    handler = BigQueryCallbackHandler(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
    )

    client = bigquery.Client(project=project_id)
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    ids, expected_rows_partial = expected_data

    try:
        serialized = {"name": "TestModel"}
        tags = ["test-tag"]

        expected_rows = list(expected_rows_partial)

        # 1. on_llm_start
        run_id, parent_run_id = ids[0]
        prompts = ["Hello world"]
        handler.on_llm_start(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 2. on_llm_end
        run_id, parent_run_id = ids[1]
        generations = [[Generation(text="Hello there!")]]
        result = LLMResult(generations=generations, llm_output={"token_usage": {}})
        handler.on_llm_end(
            response=result,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 3. on_chat_model_start
        run_id, parent_run_id = ids[2]
        messages: List[List[BaseMessage]] = [[HumanMessage(content="Hello")]]
        handler.on_chat_model_start(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 4. on_llm_error
        run_id, parent_run_id = ids[3]
        handler.on_llm_error(
            ValueError("LLM Error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 5. on_chain_start
        run_id, parent_run_id = ids[4]
        handler.on_chain_start(
            serialized=serialized,
            inputs={"input": "test"},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 6. on_chain_end
        run_id, parent_run_id = ids[5]
        handler.on_chain_end(
            outputs={"output": "test"},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 7. on_tool_start
        run_id, parent_run_id = ids[6]
        handler.on_tool_start(
            serialized=serialized,
            input_str="search",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 8. on_agent_action
        run_id, parent_run_id = ids[7]
        action = AgentAction(tool="test_tool", tool_input="test_input", log="")
        handler.on_agent_action(
            action=action, run_id=run_id, parent_run_id=parent_run_id, tags=tags
        )

        # 9. on_agent_finish
        run_id, parent_run_id = ids[8]
        finish = AgentFinish(return_values={"output": "test"}, log="")
        handler.on_agent_finish(
            finish=finish, run_id=run_id, parent_run_id=parent_run_id, tags=tags
        )

        # 10. on_tool_end
        run_id, parent_run_id = ids[9]
        handler.on_tool_end(
            output="tool output",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 11. on_tool_error
        run_id, parent_run_id = ids[10]
        handler.on_tool_error(
            ValueError("Tool Error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 12. on_chain_error
        run_id, parent_run_id = ids[11]
        handler.on_chain_error(
            ValueError("Chain Error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 13. on_retriever_start
        run_id, parent_run_id = ids[12]
        handler.on_retriever_start(
            serialized=serialized,
            query="retriever query",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 14. on_retriever_end
        run_id, parent_run_id = ids[13]
        documents = [Document(page_content="doc content")]
        handler.on_retriever_end(
            documents=documents, run_id=run_id, parent_run_id=parent_run_id, tags=tags
        )

        # 15. on_retriever_error
        run_id, parent_run_id = ids[14]
        handler.on_retriever_error(
            ValueError("Retriever Error"),
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )
        handler.on_text(
            "some text", run_id=ids[15][0], parent_run_id=ids[15][1], tags=tags
        )

        # Close the handler to flush all logs
        handler.close()

        # Check the logs in BigQuery
        _check_bq_logs(client, full_table_id, expected_rows)

    finally:
        # Clean up the table
        try:
            client.delete_table(full_table_id)
        except Exception:
            pass


@pytest.mark.extended
@pytest.mark.asyncio
async def test_async_bigquery_callback_handler(expected_data: tuple) -> None:  # type: ignore
    """Test `AsyncBigQueryCallbackHandler`."""
    project_id = os.environ.get("PROJECT_ID")
    dataset_id = os.environ.get("BIGQUERY_LOGGING_DATASET_ID")
    if not project_id or not dataset_id:
        pytest.skip(
            "PROJECT_ID or BIGQUERY_LOGGING_DATASET_ID environment variables not set"
        )

    table_prefix = os.environ.get("BIGQUERY_LOGGING_TABLE_PREFIX", "langchain")
    table_id = _get_unique_table_name(table_prefix)

    handler = AsyncBigQueryCallbackHandler(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
    )

    client = bigquery.Client(project=project_id)
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    ids, expected_rows_partial = expected_data

    try:
        serialized = {"name": "TestModel"}
        tags = ["test-tag"]

        expected_rows = list(expected_rows_partial)

        # 1. on_llm_start
        run_id, parent_run_id = ids[0]
        prompts = ["Hello world"]
        asyncio.create_task(
            handler.on_llm_start(
                serialized=serialized,
                prompts=prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 2. on_llm_end)
        run_id, parent_run_id = ids[1]
        generations = [[Generation(text="Hello there!")]]
        result = LLMResult(generations=generations, llm_output={"token_usage": {}})
        asyncio.create_task(
            handler.on_llm_end(
                response=result, run_id=run_id, parent_run_id=parent_run_id, tags=tags
            )
        )

        # 3. on_chat_model_start
        run_id, parent_run_id = ids[2]
        messages: List[List[BaseMessage]] = [[HumanMessage(content="Hello")]]
        asyncio.create_task(
            handler.on_chat_model_start(
                serialized=serialized,
                messages=messages,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 4. on_llm_error)
        run_id, parent_run_id = ids[3]
        asyncio.create_task(
            handler.on_llm_error(
                ValueError("LLM Error"),
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 5. on_chain_start)
        run_id, parent_run_id = ids[4]
        # This `await` acts as a sync point, ensuring the handler's lazy
        # initialization (e.g., table creation) completes on the event.
        # This helps verify that all subsequent events, scheduled as background
        # tasks, are correctly enqueued and fully written on `close()`,
        # and no rows are lost before/after initialization.
        await handler.on_chain_start(
            serialized=serialized,
            inputs={"input": "test"},
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
        )

        # 6. on_chain_end
        run_id, parent_run_id = ids[5]
        asyncio.create_task(
            handler.on_chain_end(
                outputs={"output": "test"},
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 7. on_tool_start
        run_id, parent_run_id = ids[6]
        asyncio.create_task(
            handler.on_tool_start(
                serialized=serialized,
                input_str="search",
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )
        # 8. on_agent_action
        run_id, parent_run_id = ids[7]
        action = AgentAction(tool="test_tool", tool_input="test_input", log="")
        asyncio.create_task(
            handler.on_agent_action(
                action=action, run_id=run_id, parent_run_id=parent_run_id, tags=tags
            )
        )

        # 9. on_agent_finish
        run_id, parent_run_id = ids[8]
        finish = AgentFinish(return_values={"output": "test"}, log="")
        asyncio.create_task(
            handler.on_agent_finish(
                finish=finish, run_id=run_id, parent_run_id=parent_run_id, tags=tags
            )
        )

        # 10. on_tool_end
        run_id, parent_run_id = ids[9]
        asyncio.create_task(
            handler.on_tool_end(
                output="tool output",
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 11. on_tool_error
        run_id, parent_run_id = ids[10]
        asyncio.create_task(
            handler.on_tool_error(
                ValueError("Tool Error"),
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 12. on_chain_error
        run_id, parent_run_id = ids[11]
        asyncio.create_task(
            handler.on_chain_error(
                ValueError("Chain Error"),
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 13. on_retriever_start
        run_id, parent_run_id = ids[12]
        asyncio.create_task(
            handler.on_retriever_start(
                serialized=serialized,
                query="retriever query",
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )
        # 14. on_retriever_end
        run_id, parent_run_id = ids[13]
        documents = [Document(page_content="doc content")]
        asyncio.create_task(
            handler.on_retriever_end(
                documents=documents,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 15. on_retriever_error
        run_id, parent_run_id = ids[14]
        asyncio.create_task(
            handler.on_retriever_error(
                ValueError("Retriever Error"),
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
            )
        )

        # 16. on_text
        asyncio.create_task(
            handler.on_text(
                "some text", run_id=ids[15][0], parent_run_id=ids[15][1], tags=tags
            )
        )

        # Close the handler to flush all logs
        await handler.close()

        # Check the logs in BigQuery
        _check_bq_logs(client, full_table_id, expected_rows)

    finally:
        # Clean up the table
        try:
            client.delete_table(full_table_id)
        except Exception:
            pass
