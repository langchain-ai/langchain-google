"""Unit tests for `BigQueryCallbackHandler`."""

import sys
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.outputs import Generation, LLMResult

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryCallbackHandler,
)


@pytest.fixture(autouse=True)
def mock_bigquery_clients() -> Generator[Dict[str, Any], None, None]:
    """Mocks the BigQuery clients and dependencies."""
    mock_bigquery = MagicMock()
    mock_bq_client = MagicMock()
    mock_bigquery.Client.return_value = mock_bq_client

    mock_async_write_client_cls = MagicMock()
    mock_async_write_client = AsyncMock()
    mock_async_write_client_cls.BigQueryWriteAsyncClient.return_value = (
        mock_async_write_client
    )

    mock_sync_write_client_cls = MagicMock()
    mock_sync_write_client = MagicMock()
    mock_sync_write_client_cls.BigQueryWriteClient.return_value = mock_sync_write_client

    mock_storage = MagicMock()
    mock_storage_client = MagicMock()
    mock_storage.Client.return_value = mock_storage_client

    mock_pyarrow = MagicMock()
    mock_pyarrow.schema().serialize().to_pybytes.return_value = b"schema"
    mock_pyarrow.RecordBatch.from_pydict()
    mock_pyarrow.RecordBatch.from_pydict().serialize().to_pybytes.return_value = (
        b"batch"
    )

    mock_google_auth = MagicMock()
    mock_google_auth.default.return_value = (MagicMock(), "test-project")

    mock_cloud_exceptions = MagicMock()
    mock_cloud_exceptions.NotFound = type("NotFound", (Exception,), {})

    mock_api_core = MagicMock()
    mock_api_core.exceptions.NotFound = type("NotFound", (Exception,), {})

    def guard_import_side_effect(module_name: str, pip_name: str = "") -> Any:
        if module_name == "google.cloud.bigquery":
            return mock_bigquery
        if (
            module_name
            == "google.cloud.bigquery_storage_v1.services.big_query_write.async_client"
        ):
            return mock_async_write_client_cls
        if (
            module_name
            == "google.cloud.bigquery_storage_v1.services.big_query_write.client"
        ):
            return mock_sync_write_client_cls
        if module_name == "google.cloud.storage":
            return mock_storage
        if module_name == "pyarrow":
            return mock_pyarrow
        if module_name == "google.auth":
            return mock_google_auth
        if module_name == "google.api_core":
            return mock_api_core
        if module_name == "google.cloud.exceptions":
            return mock_cloud_exceptions
        return MagicMock()

    with (
        patch(
            "langchain_google_community.callbacks.bigquery_callback.guard_import",
            side_effect=guard_import_side_effect,
        ),
        patch(
            "sys.modules",
            {"pyarrow": mock_pyarrow, "google.api_core": mock_api_core, **sys.modules},
        ),
    ):
        # The `pyarrow` module is imported directly in some places, so we
        # need to patch it in `sys.modules`.
        yield {
            "mock_bq_client": mock_bq_client,
            "mock_async_write_client": mock_async_write_client,
            "mock_sync_write_client": mock_sync_write_client,
            "mock_storage_client": mock_storage_client,
            "mock_cloud_exceptions": mock_cloud_exceptions,
        }


@pytest.fixture
async def handler() -> AsyncBigQueryCallbackHandler:
    """Returns an initialized `AsyncBigQueryCallbackHandler` with mocked clients."""
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    await handler._ensure_started()
    return handler


@pytest.fixture
def sync_handler() -> BigQueryCallbackHandler:
    """
    Returns an initialized `BigQueryCallbackHandler` with mocked clients.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    handler._ensure_started()
    return handler


@pytest.mark.asyncio
async def test_async_on_llm_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_llm_start` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    run_id = uuid4()
    parent_run_id = uuid4()
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["test prompt"],
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_llm_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_llm_start` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    run_id = uuid4()
    parent_run_id = uuid4()
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["test prompt"],
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_llm_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_llm_end` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    response = LLMResult(
        generations=[[Generation(text="test generation")]],
        llm_output={"model_name": "test_model"},
    )
    await handler.on_llm_end(response, run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_llm_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_llm_end` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    response = LLMResult(
        generations=[[Generation(text="test generation")]],
        llm_output={"model_name": "test_model"},
    )
    sync_handler.on_llm_end(response, run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_chain_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_chain_start` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_chain_start(
        serialized={"name": "test_chain"}, inputs={"input": "test"}, run_id=uuid4()
    )
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_chain_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_chain_start` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_chain_start(
        serialized={"name": "test_chain"}, inputs={"input": "test"}, run_id=uuid4()
    )
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_chain_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_chain_end` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_chain_end(outputs={"output": "test"}, run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_chain_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_chain_end` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_chain_end(outputs={"output": "test"}, run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_tool_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_tool_start` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_tool_start(
        serialized={"name": "test_tool"}, input_str="test", run_id=uuid4()
    )
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_tool_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_tool_start` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_tool_start(
        serialized={"name": "test_tool"}, input_str="test", run_id=uuid4()
    )
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_agent_action(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_agent_action` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    action = AgentAction(tool="test_tool", tool_input="test", log="test log")
    await handler.on_agent_action(action, run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_agent_action(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_agent_action` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    action = AgentAction(tool="test_tool", tool_input="test", log="test log")
    sync_handler.on_agent_action(action, run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_agent_finish(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_agent_finish` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    finish = AgentFinish(return_values={"output": "test"}, log="test log")
    await handler.on_agent_finish(finish, run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_agent_finish(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_agent_finish` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    finish = AgentFinish(return_values={"output": "test"}, log="test log")
    sync_handler.on_agent_finish(finish, run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_llm_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_llm_error` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_llm_error(Exception("test error"), run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_llm_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_llm_error` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_llm_error(Exception("test error"), run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_chat_model_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_chat_model_start` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_chat_model_start(
        serialized={"name": "test_chat_model"},
        messages=[[HumanMessage(content="test")]],
        run_id=uuid4(),
    )
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_chat_model_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_chat_model_start` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_chat_model_start(
        serialized={"name": "test_chat_model"},
        messages=[[HumanMessage(content="test")]],
        run_id=uuid4(),
    )
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_retriever_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_retriever_end` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    documents = [Document(page_content="test document")]
    await handler.on_retriever_end(documents, run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_retriever_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_retriever_end` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    documents = [Document(page_content="test document")]
    sync_handler.on_retriever_end(documents, run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


def test_sync_ensure_init_creates_dataset_and_table(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that sync `_ensure_init` creates dataset and table if they don't exist."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_cloud_exceptions = mock_bigquery_clients["mock_cloud_exceptions"]

    mock_bq_client.get_table.side_effect = mock_cloud_exceptions.NotFound(
        "Table not found"
    )
    handler._ensure_started()

    mock_bq_client.get_table.assert_called_once()
    mock_bq_client.create_table.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_init_creates_dataset_and_table(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that `_ensure_init` creates dataset and table if they don't exist."""
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project", dataset_id="test_dataset", table_id="test_table"
    )
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_cloud_exceptions = mock_bigquery_clients["mock_cloud_exceptions"]

    mock_bq_client.get_table.side_effect = mock_cloud_exceptions.NotFound(
        "Table not found"
    )

    await handler._ensure_started()

    mock_bq_client.get_table.assert_called_once()
    mock_bq_client.create_table.assert_called_once()


@pytest.mark.asyncio
async def test_async_close(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that the shutdown method closes clients."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.shutdown = AsyncMock()  # type: ignore[method-assign]
    await handler.shutdown()
    if handler.async_batch_processor:
        handler.async_batch_processor.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_tool_end(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_tool_end` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_tool_end(output="test output", run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_tool_end(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_tool_end` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_tool_end(output="test output", run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_tool_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_tool_error` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_tool_error(Exception("tool error"), run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_tool_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_tool_error` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_tool_error(Exception("tool error"), run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_chain_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_chain_error` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_chain_error(Exception("chain error"), run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_chain_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_chain_error` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_chain_error(Exception("chain error"), run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_retriever_start(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_retriever_start` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_retriever_start(
        serialized={"name": "test_retriever"}, query="test query", run_id=uuid4()
    )
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_retriever_start(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_retriever_start` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_retriever_start(
        serialized={"name": "test_retriever"}, query="test query", run_id=uuid4()
    )
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_retriever_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_retriever_error` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_retriever_error(Exception("retriever error"), run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_retriever_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_retriever_error` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_retriever_error(Exception("retriever error"), run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


@pytest.mark.asyncio
async def test_async_on_text(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that `on_text` logs the correct event."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]
    await handler.on_text("some text", run_id=uuid4())
    handler.async_batch_processor.append.assert_called_once()


def test_sync_on_text(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that sync `on_text` logs the correct event."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]
    sync_handler.on_text("some text", run_id=uuid4())
    sync_handler.batch_processor.append.assert_called_once()


def test_sync_close(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that the sync shutdown method closes clients."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.shutdown = MagicMock()  # type: ignore[method-assign]
    sync_handler.shutdown()
    if sync_handler.batch_processor:
        sync_handler.batch_processor.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_async_log_parsing_error(
    handler: AsyncBigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that a parsing error is handled gracefully in async handler."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    run_id = uuid4()
    error_message = "Test parsing error"

    with (
        patch(
            "langchain_google_community.callbacks.bigquery_callback._LangChainContentParser.parse_message_content",
            side_effect=Exception(error_message),
        ),
        patch(
            "langchain_google_community.callbacks.bigquery_callback.logger.warning"
        ) as mock_warning,
    ):
        await handler.on_text("some text that will fail parsing", run_id=run_id)

        mock_warning.assert_called_once()
        handler.async_batch_processor.append.assert_called_once()
        logged_row = handler.async_batch_processor.append.call_args[0][0]
        assert logged_row["status"] == "ERROR"
        assert (
            f"Failed to parse content: {error_message}" in logged_row["error_message"]
        )


def test_sync_log_parsing_error(
    sync_handler: BigQueryCallbackHandler, mock_bigquery_clients: Dict[str, Any]
) -> None:
    """Test that a parsing error is handled gracefully in sync handler."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    error_message = "Test parsing error"

    with (
        patch(
            "langchain_google_community.callbacks.bigquery_callback._SyncLangChainContentParser.parse_message_content",
            side_effect=Exception(error_message),
        ),
        patch(
            "langchain_google_community.callbacks.bigquery_callback.logger.warning"
        ) as mock_warning,
    ):
        sync_handler.on_text("some text that will fail parsing", run_id=run_id)
        mock_warning.assert_called_once()
        sync_handler.batch_processor.append.assert_called_once()
        logged_row = sync_handler.batch_processor.append.call_args[0][0]
        assert logged_row["status"] == "ERROR"
        assert (
            f"Failed to parse content: {error_message}" in logged_row["error_message"]
        )
