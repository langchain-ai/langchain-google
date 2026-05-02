"""Unit tests for `BigQueryCallbackHandler`."""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    AsyncTraceIdRegistry,
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
    TraceIdRegistry,
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
            {
                "pyarrow": mock_pyarrow,
                "google.api_core": mock_api_core,
                **sys.modules,
            },
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
async def handler() -> AsyncGenerator[AsyncBigQueryCallbackHandler, None]:
    """Returns an initialized `AsyncBigQueryCallbackHandler` with mocked clients."""
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    await handler._ensure_started()

    # Hand the handler to the test
    yield handler

    # Clean up the background tasks after the test completes!
    await handler.shutdown()


@pytest.fixture
def sync_handler() -> Generator[BigQueryCallbackHandler, None, None]:
    """Returns an initialized `BigQueryCallbackHandler` with mocked clients."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    handler._ensure_started()

    # Hand the handler to the test
    yield handler

    # Clean up the background threads after the test completes!
    handler.shutdown()


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
        serialized={"name": "test_chain"},
        inputs={"input": "test"},
        run_id=uuid4(),
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
        serialized={"name": "test_chain"},
        inputs={"input": "test"},
        run_id=uuid4(),
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
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
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


def test_sync_init_raises_if_dataset_missing(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that sync init raises ValueError if the dataset does not exist."""
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_cloud_exceptions = mock_bigquery_clients["mock_cloud_exceptions"]

    # Simulate dataset not found
    mock_bq_client.get_dataset.side_effect = mock_cloud_exceptions.NotFound(
        "Dataset not found"
    )

    with pytest.raises(ValueError, match="Dataset 'test_dataset' does not exist"):
        BigQueryCallbackHandler(
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
        )

    mock_bq_client.get_dataset.assert_called_with("test_dataset")


def test_async_init_raises_if_dataset_missing(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that async init raises ValueError if the dataset does not exist."""
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_cloud_exceptions = mock_bigquery_clients["mock_cloud_exceptions"]

    # Simulate dataset not found
    mock_bq_client.get_dataset.side_effect = mock_cloud_exceptions.NotFound(
        "Dataset not found"
    )

    with pytest.raises(ValueError, match="Dataset 'test_dataset' does not exist"):
        AsyncBigQueryCallbackHandler(
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="test_table",
        )

    mock_bq_client.get_dataset.assert_called_with("test_dataset")


def test_trace_id_registry_root_run() -> None:
    """Verify that a root run gets its own ID as the trace ID."""
    registry = TraceIdRegistry()
    run_id = uuid4()

    # Register a root run (no parent)
    trace_id = registry.register_run(run_id)

    # Trace ID should match the run ID for root runs
    assert trace_id == str(run_id)

    # Cleanup
    registry.end_run(run_id)
    assert run_id not in registry._run_map


def test_trace_id_registry_child_run_propagation() -> None:
    """Verify that a root run gets its own ID as the trace ID."""
    registry = TraceIdRegistry()
    root_run_id = uuid4()
    child_run_id = uuid4()
    grandchild_run_id = uuid4()

    # 1. Start Root
    root_trace_id = registry.register_run(root_run_id)
    assert root_trace_id == str(root_run_id)

    # 2. Start Child (linked to Root)
    child_trace_id = registry.register_run(child_run_id, parent_run_id=root_run_id)
    assert child_trace_id == str(root_run_id)

    # 3. Start Grandchild (linked to Child)
    grandchild_trace_id = registry.register_run(
        grandchild_run_id, parent_run_id=child_run_id
    )
    assert grandchild_trace_id == str(root_run_id)

    # 4. End Root Run (should clean up all descendants in the map)
    registry.end_run(root_run_id)

    assert root_run_id not in registry._run_map
    assert child_run_id not in registry._run_map
    assert grandchild_run_id not in registry._run_map


def test_trace_id_registry_missing_parent_behavior() -> None:
    """If parent is unknown, it should be treated as a new root."""
    registry = TraceIdRegistry()
    run_id = uuid4()
    unknown_parent_id = uuid4()

    trace_id = registry.register_run(run_id, parent_run_id=unknown_parent_id)

    # Should adopt the parent ID as trace ID even if parent wasn't explicitly registered
    assert trace_id == str(unknown_parent_id)


def test_trace_id_registry_concurrency() -> None:
    """Verify thread safety."""
    registry = TraceIdRegistry()
    root_run_id = uuid4()
    registry.register_run(root_run_id)

    def register_child(_: int) -> str:
        child_id = uuid4()
        return registry.register_run(child_id, parent_run_id=root_run_id)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(register_child, range(100)))

    for tid in results:
        assert tid == str(root_run_id)


@pytest.mark.asyncio
async def test_async_trace_id_registry_root_run() -> None:
    """Verify that a root run gets its own ID as the trace ID."""
    registry = AsyncTraceIdRegistry()
    root_run_id = uuid4()

    # Register a root run (no parent)
    trace_id = await registry.register_run(root_run_id)

    # Trace ID should match the run ID for root runs
    assert trace_id == str(root_run_id)

    # Cleanup
    await registry.end_run(root_run_id)
    assert root_run_id not in registry._run_map


@pytest.mark.asyncio
async def test_async_trace_id_registry_child_run_propagation() -> None:
    """Verify that child runs inherit the trace ID from the root."""
    registry = AsyncTraceIdRegistry()
    root_run_id = uuid4()
    child_run_id = uuid4()
    grandchild_run_id = uuid4()

    # 1. Start Root
    root_trace_id = await registry.register_run(root_run_id)
    assert root_trace_id == str(root_run_id)

    # 2. Start Child (linked to Root)
    child_trace_id = await registry.register_run(
        child_run_id, parent_run_id=root_run_id
    )
    assert child_trace_id == str(root_run_id)

    # 3. Start Grandchild (linked to Child)
    grandchild_trace_id = await registry.register_run(
        grandchild_run_id, parent_run_id=child_run_id
    )
    assert grandchild_trace_id == str(root_run_id)

    # 4. End Root Run (should clean up all descendants in the map)
    await registry.end_run(root_run_id)

    assert root_run_id not in registry._run_map
    assert child_run_id not in registry._run_map
    assert grandchild_run_id not in registry._run_map


@pytest.mark.asyncio
async def test_async_trace_id_registry_missing_parent_behavior() -> None:
    """If parent is unknown, it should be treated as a new root."""
    registry = AsyncTraceIdRegistry()
    run_id = uuid4()
    unknown_parent_id = uuid4()

    trace_id = await registry.register_run(run_id, parent_run_id=unknown_parent_id)

    # Should adopt the parent ID as trace ID even if parent wasn't explicitly registered
    assert trace_id == str(unknown_parent_id)


@pytest.mark.asyncio
async def test_async_trace_id_registry_concurrency() -> None:
    """Verify async concurrency safety."""
    registry = AsyncTraceIdRegistry()
    root_run_id = uuid4()
    await registry.register_run(root_run_id)

    async def register_child(_: int) -> str:
        child_id = uuid4()
        return await registry.register_run(child_id, parent_run_id=root_run_id)

    tasks = [register_child(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    for tid in results:
        assert tid == str(root_run_id)


# ==============================================================================
# LATENCY TRACKER TESTS
# ==============================================================================


def test_latency_tracker_basic_timing() -> None:
    """Test basic start/end timing functionality."""
    from langchain_google_community.callbacks.bigquery_callback import LatencyTracker

    tracker = LatencyTracker()
    run_id = uuid4()

    tracker.start(run_id)
    import time

    time.sleep(0.1)  # Sleep for 100ms
    result = tracker.end(run_id)

    assert result is not None
    assert result.total_ms >= 100  # Should be at least 100ms


def test_latency_tracker_component_timing() -> None:
    """Test component timing functionality."""
    from langchain_google_community.callbacks.bigquery_callback import LatencyTracker

    tracker = LatencyTracker()
    run_id = uuid4()

    tracker.start(run_id)
    tracker.start_component(run_id, "parsing")
    import time

    time.sleep(0.05)
    tracker.end_component(run_id, "parsing")
    result = tracker.end(run_id)

    assert result is not None
    assert result.component_ms is not None
    assert "parsing" in result.component_ms
    assert result.component_ms["parsing"] >= 50


def test_latency_tracker_missing_start() -> None:
    """Test that ending a run without starting returns None."""
    from langchain_google_community.callbacks.bigquery_callback import LatencyTracker

    tracker = LatencyTracker()
    run_id = uuid4()

    result = tracker.end(run_id)
    assert result is None


def test_latency_tracker_stale_cleanup() -> None:
    """Test that stale entries are cleaned up."""
    from langchain_google_community.callbacks.bigquery_callback import LatencyTracker

    # Use a very short stale threshold for testing
    tracker = LatencyTracker(stale_threshold_ms=1)
    run_id1 = uuid4()
    run_id2 = uuid4()

    tracker.start(run_id1)
    import time

    time.sleep(0.01)  # Wait for run_id1 to become stale

    # Starting a new run should clean up stale entries
    tracker.start(run_id2)

    # run_id1 should have been cleaned up
    assert tracker.end(run_id1) is None
    assert tracker.end(run_id2) is not None


@pytest.mark.asyncio
async def test_async_latency_tracker_basic_timing() -> None:
    """Test basic async start/end timing functionality."""
    from langchain_google_community.callbacks.bigquery_callback import (
        AsyncLatencyTracker,
    )

    tracker = AsyncLatencyTracker()
    run_id = uuid4()

    await tracker.start(run_id)
    await asyncio.sleep(0.1)  # Sleep for 100ms
    result = await tracker.end(run_id)

    assert result is not None
    assert result.total_ms >= 100


@pytest.mark.asyncio
async def test_async_latency_tracker_component_timing() -> None:
    """Test async component timing functionality."""
    from langchain_google_community.callbacks.bigquery_callback import (
        AsyncLatencyTracker,
    )

    tracker = AsyncLatencyTracker()
    run_id = uuid4()

    await tracker.start(run_id)
    await tracker.start_component(run_id, "llm_call")
    await asyncio.sleep(0.05)
    await tracker.end_component(run_id, "llm_call")
    result = await tracker.end(run_id)

    assert result is not None
    assert result.component_ms is not None
    assert "llm_call" in result.component_ms


@pytest.mark.asyncio
async def test_async_latency_tracker_missing_start() -> None:
    """Test that ending a run without starting returns None."""
    from langchain_google_community.callbacks.bigquery_callback import (
        AsyncLatencyTracker,
    )

    tracker = AsyncLatencyTracker()
    run_id = uuid4()

    result = await tracker.end(run_id)
    assert result is None


# ==============================================================================
# RUN CONTEXT REGISTRY TESTS
# ==============================================================================


def test_run_context_registry_register_get_pop() -> None:
    """Test register, get, and pop operations."""
    from langchain_google_community.callbacks.bigquery_callback import (
        RunContextRegistry,
    )

    registry = RunContextRegistry()
    run_id = uuid4()

    # Register
    context = registry.register(run_id, "my_tool", metadata={"key": "value"})
    assert context.name == "my_tool"
    assert context.run_id == str(run_id)

    # Get (should not remove)
    got = registry.get(run_id)
    assert got is not None
    assert got.name == "my_tool"

    # Get again (should still exist)
    got2 = registry.get(run_id)
    assert got2 is not None

    # Pop (should remove)
    popped = registry.pop(run_id)
    assert popped is not None
    assert popped.name == "my_tool"

    # Get after pop (should be None)
    assert registry.get(run_id) is None


def test_run_context_registry_update_metadata() -> None:
    """Test updating metadata for a run context."""
    from langchain_google_community.callbacks.bigquery_callback import (
        RunContextRegistry,
    )

    registry = RunContextRegistry()
    run_id = uuid4()

    registry.register(run_id, "tool", metadata={"a": 1})
    updated = registry.update_metadata(run_id, {"b": 2})

    assert updated is not None
    assert updated.metadata["a"] == 1
    assert updated.metadata["b"] == 2


@pytest.mark.asyncio
async def test_async_run_context_registry_register_pop() -> None:
    """Test async register and pop operations."""
    from langchain_google_community.callbacks.bigquery_callback import (
        AsyncRunContextRegistry,
    )

    registry = AsyncRunContextRegistry()
    run_id = uuid4()

    # Register
    context = await registry.register(run_id, "search_tool")
    assert context.name == "search_tool"

    # Pop
    popped = await registry.pop(run_id)
    assert popped is not None
    assert popped.name == "search_tool"

    # Pop again (should be None)
    assert await registry.pop(run_id) is None


# ==============================================================================


# ==============================================================================
# EVENT FILTERING TESTS
# ==============================================================================


def test_event_filtering_denylist(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that events in the denylist are not logged."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
        BigQueryLoggerConfig,
    )

    config = BigQueryLoggerConfig(event_denylist=["CHAIN_START", "CHAIN_END"])
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=config,
    )

    assert handler._should_log_event("LLM_REQUEST") is True
    assert handler._should_log_event("CHAIN_START") is False
    assert handler._should_log_event("CHAIN_END") is False


def test_event_filtering_allowlist(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that only events in the allowlist are logged."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
        BigQueryLoggerConfig,
    )

    config = BigQueryLoggerConfig(event_allowlist=["LLM_REQUEST", "LLM_RESPONSE"])
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=config,
    )

    assert handler._should_log_event("LLM_REQUEST") is True
    assert handler._should_log_event("LLM_RESPONSE") is True
    assert handler._should_log_event("CHAIN_START") is False
    assert handler._should_log_event("TOOL_STARTING") is False


def test_event_filtering_no_filters(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that all events are logged when no filters are set."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
    )

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )

    assert handler._should_log_event("LLM_REQUEST") is True
    assert handler._should_log_event("CHAIN_START") is True
    assert handler._should_log_event("ANY_EVENT") is True


# ==============================================================================
# LANGGRAPH DETECTION TESTS
# ==============================================================================


def test_langgraph_node_detection(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test detection of LangGraph nodes via metadata."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
    )

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        graph_name="test_graph",
    )

    # Regular chain (not a LangGraph node)
    serialized = {"name": "RunnableSequence"}
    metadata: Dict[str, Any] = {}
    assert handler._is_langgraph_root_invocation(serialized, uuid4(), metadata) is False

    # LangGraph root invocation
    serialized = {"name": "CompiledGraph"}
    metadata = {"langgraph_step": 0}
    assert handler._is_langgraph_root_invocation(serialized, None, metadata) is True


def test_langgraph_attributes_building(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test building LangGraph attributes."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
    )

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        graph_name="my_graph",
    )

    metadata = {
        "langgraph_node": "process_node",
        "langgraph_step": 2,
        "langgraph_triggers": ["start"],
    }

    attrs = handler._build_langgraph_attributes(
        node_name="process_node", metadata=metadata
    )

    assert "langgraph" in attrs
    assert attrs["langgraph"]["graph_name"] == "my_graph"
    assert attrs["langgraph"]["node_name"] == "process_node"
    assert attrs["langgraph"]["step"] == 2
    assert attrs["langgraph"]["triggers"] == ["start"]


def test_execution_order_tracking(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test execution order tracking."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
    )

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )

    assert handler._get_execution_order() == 0
    handler._increment_execution_order()
    assert handler._get_execution_order() == 1
    handler._increment_execution_order()
    assert handler._get_execution_order() == 2
    handler._reset_execution_order()
    assert handler._get_execution_order() == 0


def test_execution_order_isolation(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that execution order is isolated between instances."""
    from langchain_google_community.callbacks.bigquery_callback import (
        BigQueryCallbackHandler,
    )

    handler1 = BigQueryCallbackHandler(
        project_id="test-project", dataset_id="test_dataset", table_id="t1"
    )
    handler2 = BigQueryCallbackHandler(
        project_id="test-project", dataset_id="test_dataset", table_id="t2"
    )

    # Increment handler1
    handler1._increment_execution_order()
    assert handler1._get_execution_order() == 1
    assert handler2._get_execution_order() == 0

    # Increment handler2
    handler2._increment_execution_order()
    handler2._increment_execution_order()
    assert handler1._get_execution_order() == 1
    assert handler2._get_execution_order() == 2


# ==============================================================================
# GRAPH EXECUTION CONTEXT TESTS
# ==============================================================================


def test_context_manager_emits_events(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Context manager emits INVOCATION_STARTING and INVOCATION_COMPLETED."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    with sync_handler.graph_context("test_graph"):
        pass  # Graph execution would happen here

    # Should have two calls: INVOCATION_STARTING and INVOCATION_COMPLETED
    assert sync_handler.batch_processor.append.call_count == 2

    calls = sync_handler.batch_processor.append.call_args_list
    assert calls[0][0][0]["event_type"] == "INVOCATION_STARTING"
    assert calls[1][0][0]["event_type"] == "INVOCATION_COMPLETED"


def test_context_manager_handles_errors(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that the context manager emits INVOCATION_ERROR on exception."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    try:
        with sync_handler.graph_context("test_graph"):
            raise ValueError("Test error")
    except ValueError:
        pass

    # Should have two calls: INVOCATION_STARTING and INVOCATION_ERROR
    assert sync_handler.batch_processor.append.call_count == 2

    calls = sync_handler.batch_processor.append.call_args_list
    assert calls[0][0][0]["event_type"] == "INVOCATION_STARTING"
    assert calls[1][0][0]["event_type"] == "INVOCATION_ERROR"
    assert "Test error" in calls[1][0][0]["error_message"]


@pytest.mark.asyncio
async def test_async_context_manager_emits_events(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Async context manager emits INVOCATION_STARTING and INVOCATION_COMPLETED."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    async with handler.graph_context("test_graph"):
        pass  # Graph execution would happen here

    # Should have two calls: INVOCATION_STARTING and INVOCATION_COMPLETED
    assert handler.async_batch_processor.append.call_count == 2

    calls = handler.async_batch_processor.append.call_args_list
    assert calls[0][0][0]["event_type"] == "INVOCATION_STARTING"
    assert calls[1][0][0]["event_type"] == "INVOCATION_COMPLETED"


@pytest.mark.asyncio
async def test_async_context_manager_handles_errors(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that the async context manager emits INVOCATION_ERROR on exception."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    try:
        async with handler.graph_context("test_graph"):
            raise RuntimeError("Async test error")
    except RuntimeError:
        pass

    # Should have two calls: INVOCATION_STARTING and INVOCATION_ERROR
    assert handler.async_batch_processor.append.call_count == 2

    calls = handler.async_batch_processor.append.call_args_list
    assert calls[0][0][0]["event_type"] == "INVOCATION_STARTING"
    assert calls[1][0][0]["event_type"] == "INVOCATION_ERROR"


# ==============================================================================
# TOOL NAME TRACKING TESTS
# ==============================================================================


def test_tool_name_tracking_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that tool names are properly tracked through start/end."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()

    # Simulate tool_start
    sync_handler.on_tool_start(
        serialized={"name": "search_tool"},
        input_str="search query",
        run_id=run_id,
    )

    # Simulate tool_end
    sync_handler.on_tool_end(
        output="search results",
        run_id=run_id,
    )

    calls = sync_handler.batch_processor.append.call_args_list
    assert len(calls) == 2

    # Check tool name is in the end event
    end_call = calls[1][0][0]
    assert end_call["event_type"] == "TOOL_COMPLETED"
    assert end_call["attributes"]["tool_name"] == "search_tool"


@pytest.mark.asyncio
async def test_tool_name_tracking_async(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Test that tool names are properly tracked through start/end asynchronously."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    run_id = uuid4()

    # Simulate tool_start
    await handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2 + 2",
        run_id=run_id,
    )

    # Simulate tool_end
    await handler.on_tool_end(
        output="4",
        run_id=run_id,
    )

    calls = handler.async_batch_processor.append.call_args_list
    assert len(calls) == 2

    # Check tool name is in the end event
    end_call = calls[1][0][0]
    assert end_call["event_type"] == "TOOL_COMPLETED"
    assert end_call["attributes"]["tool_name"] == "calculator"


# ==============================================================================
# REGRESSION TESTS FOR ISSUE #1690
# (Metadata persistence across CHAIN_END / CHAIN_ERROR events)
# ==============================================================================


def test_chain_end_preserves_metadata_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """CHAIN_END must carry agent / user_id / session_id from the start call.

    langchain-core does not forward `metadata` to `on_chain_end`; without
    the start-time registry we'd lose it. See issue #1690.
    """
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {
        "session_id": "test-session",
        "user_id": "test-user",
        "agent": "test-agent",
    }

    sync_handler.on_chain_start(
        serialized={"name": "test_chain"},
        inputs={"input": "hello"},
        run_id=run_id,
        metadata=metadata,
    )
    # langchain-core deliberately omits metadata from on_chain_end kwargs.
    sync_handler.on_chain_end(outputs={"output": "ok"}, run_id=run_id)

    calls = sync_handler.batch_processor.append.call_args_list
    assert len(calls) == 2
    start_row, end_row = calls[0][0][0], calls[1][0][0]

    for row in (start_row, end_row):
        assert row["session_id"] == "test-session"
        assert row["user_id"] == "test-user"
        assert row["agent"] == "test-agent"


def test_chain_error_preserves_metadata_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """CHAIN_ERROR must also carry metadata captured at start (issue #1690)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {
        "session_id": "sess",
        "user_id": "alice",
        "agent": "react-agent",
    }

    sync_handler.on_chain_start(
        serialized={"name": "test_chain"},
        inputs={"input": "boom"},
        run_id=run_id,
        metadata=metadata,
    )
    sync_handler.on_chain_error(error=RuntimeError("nope"), run_id=run_id)

    calls = sync_handler.batch_processor.append.call_args_list
    assert len(calls) == 2
    err_row = calls[1][0][0]
    assert err_row["event_type"] == "CHAIN_ERROR"
    assert err_row["session_id"] == "sess"
    assert err_row["user_id"] == "alice"
    assert err_row["agent"] == "react-agent"


@pytest.mark.asyncio
async def test_chain_end_preserves_metadata_async(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {
        "session_id": "s1",
        "user_id": "u1",
        "agent": "a1",
    }

    await handler.on_chain_start(
        serialized={"name": "chain_async"},
        inputs={"q": "hi"},
        run_id=run_id,
        metadata=metadata,
    )
    await handler.on_chain_end(outputs={"o": "yo"}, run_id=run_id)

    calls = handler.async_batch_processor.append.call_args_list
    assert len(calls) == 2
    for row in (calls[0][0][0], calls[1][0][0]):
        assert row["session_id"] == "s1"
        assert row["user_id"] == "u1"
        assert row["agent"] == "a1"


def test_tool_end_preserves_metadata_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """TOOL_COMPLETED must keep session_id/user_id/agent (issue #1690)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {"session_id": "s2", "user_id": "u2", "agent": "a2"}
    sync_handler.on_tool_start(
        serialized={"name": "calc"},
        input_str="1+1",
        run_id=run_id,
        metadata=metadata,
    )
    sync_handler.on_tool_end(output="2", run_id=run_id)

    end_row = sync_handler.batch_processor.append.call_args_list[1][0][0]
    assert end_row["session_id"] == "s2"
    assert end_row["user_id"] == "u2"
    assert end_row["agent"] == "a2"


def test_llm_end_preserves_metadata_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """LLM_RESPONSE must keep metadata captured at on_llm_start (issue #1690)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {"session_id": "s3", "user_id": "u3", "agent": "a3"}
    sync_handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["hi"],
        run_id=run_id,
        metadata=metadata,
    )
    sync_handler.on_llm_end(
        LLMResult(generations=[[Generation(text="hello")]], llm_output={}),
        run_id=run_id,
    )

    end_row = sync_handler.batch_processor.append.call_args_list[1][0][0]
    assert end_row["session_id"] == "s3"
    assert end_row["user_id"] == "u3"
    assert end_row["agent"] == "a3"


# ==============================================================================
# REGRESSION TESTS FOR ISSUE #1720
# (Token tracking, sub-agent attribution, event-noise filtering)
# ==============================================================================


def test_token_usage_extracted_from_legacy_llm_output(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Legacy `llm_output['token_usage']` is still picked up (issue #1720)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    response = LLMResult(
        generations=[[Generation(text="ok")]],
        llm_output={"token_usage": {"total_tokens": 42, "prompt_tokens": 30}},
    )
    sync_handler.on_llm_end(response, run_id=uuid4())

    row = sync_handler.batch_processor.append.call_args_list[0][0][0]
    assert row["attributes"]["usage"] == {
        "total_tokens": 42,
        "prompt_tokens": 30,
    }


def test_token_usage_extracted_from_chat_message_metadata(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Modern Chat models attach `usage_metadata` to the AIMessage; the
    handler must surface it when `llm_output` is empty (issue #1720)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    msg = AIMessage(
        content="hello there",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
        },
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output=None,
    )
    sync_handler.on_llm_end(response, run_id=uuid4())

    row = sync_handler.batch_processor.append.call_args_list[0][0][0]
    usage = row["attributes"]["usage"]
    assert usage is not None
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 125


def test_sub_agent_attribution_from_langgraph_node_sync(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """When `agent` isn't set explicitly, the active LangGraph node fills the
    `agent` column so multi-agent telemetry can be filtered per sub-agent
    (issue #1720)."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    metadata = {
        "session_id": "s",
        "user_id": "u",
        "langgraph_node": "TheCritic",
        "langgraph_step": 2,
    }
    sync_handler.on_chain_start(
        serialized={"name": "TheCritic"},
        inputs={"x": 1},
        run_id=run_id,
        metadata=metadata,
    )
    sync_handler.on_chain_end(outputs={"y": 2}, run_id=run_id)

    calls = sync_handler.batch_processor.append.call_args_list
    assert len(calls) == 2
    for row in (calls[0][0][0], calls[1][0][0]):
        assert row["agent"] == "TheCritic"


def test_explicit_agent_metadata_overrides_langgraph_node(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Explicit `agent` always wins over the LangGraph-node fallback."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    sync_handler.on_chain_start(
        serialized={"name": "x"},
        inputs={},
        run_id=uuid4(),
        metadata={"agent": "PrimaryAgent", "langgraph_node": "TheCritic"},
    )
    row = sync_handler.batch_processor.append.call_args_list[0][0][0]
    assert row["agent"] == "PrimaryAgent"


def test_skip_internal_chain_events_drops_framework_chains(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`skip_internal_chain_events=True` removes noisy framework chain events
    (ChannelWrite, RunnableLambda, ...) from telemetry (issue #1720)."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(skip_internal_chain_events=True),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    parent_run_id = uuid4()
    internal_run_id = uuid4()

    # An internal LangGraph chain must be silently dropped on both ends.
    handler.on_chain_start(
        serialized={"name": "ChannelWrite<messages>"},
        inputs={"x": 1},
        run_id=internal_run_id,
        parent_run_id=parent_run_id,
        metadata={},
    )
    handler.on_chain_end(outputs={"x": 2}, run_id=internal_run_id)

    # A user-defined chain must still be emitted.
    user_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "MyCustomChain"},
        inputs={"x": 1},
        run_id=user_run_id,
        parent_run_id=parent_run_id,
        metadata={},
    )
    handler.on_chain_end(outputs={"x": 2}, run_id=user_run_id)

    emitted = handler.batch_processor.append.call_args_list
    assert len(emitted) == 2
    for row in emitted:
        # Only the user chain's events should have been recorded.
        assert "ChannelWrite" not in (row[0][0]["content"] or {}).get("data", "")
        assert row[0][0]["event_type"] in ("CHAIN_START", "CHAIN_END")


def test_skip_internal_chain_events_preserves_langgraph_nodes(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """LangGraph node chains must NOT be skipped, even when their name matches
    the internal pattern."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(skip_internal_chain_events=True),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "RunnableLambda"},  # matches internal pattern
        inputs={},
        run_id=run_id,
        metadata={"langgraph_node": "TheMeteo"},  # ...but it's a node
    )
    handler.on_chain_end(outputs={}, run_id=run_id)

    emitted = handler.batch_processor.append.call_args_list
    assert len(emitted) == 2
    assert emitted[0][0][0]["event_type"] == "AGENT_STARTING"
    assert emitted[1][0][0]["event_type"] == "AGENT_COMPLETED"
    assert emitted[1][0][0]["agent"] == "TheMeteo"


def test_extract_token_usage_returns_none_for_empty_response() -> None:
    """No usage anywhere → returns None (no spurious zeros)."""
    response = LLMResult(generations=[[Generation(text="x")]], llm_output=None)
    assert BigQueryCallbackHandler._extract_token_usage(response) is None


def test_extract_token_usage_prefers_legacy_when_present() -> None:
    """If `llm_output['token_usage']` is set, use it verbatim."""
    msg = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"token_usage": {"total_tokens": 99}},
    )
    assert BigQueryCallbackHandler._extract_token_usage(response) == {
        "total_tokens": 99,
    }


def test_extract_token_usage_preserves_empty_legacy_dict() -> None:
    """`llm_output={'token_usage': {}}` must round-trip as `{}`.

    Some providers explicitly emit an empty dict to signal "I checked and
    there is no usage info" — meaningfully different from "the field isn't
    here at all". The integration test
    `tests/integration_tests/callbacks/test_bigquery_callback.py` asserts
    `attributes['usage'] == {}` on this path, so the extractor must use a
    presence check (not a truthiness check) on the legacy slot.
    """
    response = LLMResult(
        generations=[[Generation(text="x")]],
        llm_output={"token_usage": {}},
    )
    assert BigQueryCallbackHandler._extract_token_usage(response) == {}


def test_skipped_internal_chain_preserves_trace_continuity_for_children(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Trace continuity must survive `skip_internal_chain_events=True`.

    When an internal chain (ChannelWrite, RunnableLambda, …) is skipped, any
    LLM/tool child whose `parent_run_id` points at that skipped chain must
    still resolve to the real graph root in the BigQuery `trace_id` column.
    Otherwise the child becomes its own root and we lose the ability to join
    rows for the same end-to-end invocation.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(skip_internal_chain_events=True),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    root_run_id = uuid4()
    skipped_run_id = uuid4()
    child_llm_run_id = uuid4()

    handler.on_chain_start(
        serialized={"name": "MyGraph"},
        inputs={},
        run_id=root_run_id,
        parent_run_id=None,
        metadata={"langgraph_step": 1},
    )
    handler.on_chain_start(
        serialized={"name": "ChannelWrite<messages>"},
        inputs={},
        run_id=skipped_run_id,
        parent_run_id=root_run_id,
        metadata={},
    )
    handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["hi"],
        run_id=child_llm_run_id,
        parent_run_id=skipped_run_id,
    )

    emitted_event_types = [
        call.args[0]["event_type"]
        for call in handler.batch_processor.append.call_args_list
    ]
    assert "LLM_REQUEST" in emitted_event_types
    assert "CHAIN_START" not in emitted_event_types  # internal chain dropped

    llm_row = next(
        call.args[0]
        for call in handler.batch_processor.append.call_args_list
        if call.args[0]["event_type"] == "LLM_REQUEST"
    )
    # The bug being guarded against: without registering the skipped run in
    # trace_registry, the child's trace_id collapses to skipped_run_id.
    assert llm_row["trace_id"] != str(skipped_run_id), (
        "trace_id collapsed onto the skipped internal chain — children no "
        "longer share a trace with the real graph root"
    )
    assert llm_row["trace_id"] == str(root_run_id)
    assert llm_row["parent_span_id"] == str(skipped_run_id)


@pytest.mark.asyncio
async def test_skipped_internal_chain_preserves_trace_continuity_async(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Async equivalent of the trace-continuity guard above."""
    handler = AsyncBigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(skip_internal_chain_events=True),
    )
    await handler._ensure_started()
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.async_batch_processor.append = AsyncMock()  # type: ignore[method-assign]

    root_run_id = uuid4()
    skipped_run_id = uuid4()
    child_llm_run_id = uuid4()

    await handler.on_chain_start(
        serialized={"name": "MyGraph"},
        inputs={},
        run_id=root_run_id,
        parent_run_id=None,
        metadata={"langgraph_step": 1},
    )
    await handler.on_chain_start(
        serialized={"name": "ChannelWrite<messages>"},
        inputs={},
        run_id=skipped_run_id,
        parent_run_id=root_run_id,
        metadata={},
    )
    await handler.on_llm_start(
        serialized={"name": "test_llm"},
        prompts=["hi"],
        run_id=child_llm_run_id,
        parent_run_id=skipped_run_id,
    )

    llm_row = next(
        call.args[0]
        for call in handler.async_batch_processor.append.call_args_list
        if call.args[0]["event_type"] == "LLM_REQUEST"
    )
    assert llm_row["trace_id"] == str(root_run_id)
    assert llm_row["trace_id"] != str(skipped_run_id)


# ==============================================================================
# ADK PARITY: ATTRIBUTE ENRICHMENT
# ==============================================================================


def test_attributes_enriched_with_root_agent_name(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`attributes.root_agent_name` mirrors the handler's `graph_name`.

    Matches ADK's `_enrich_attributes` so dashboards can group by top-level
    agent without users having to set `metadata['agent']` themselves.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        graph_name="MyTopAgent",
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    handler.on_chain_start(serialized={"name": "x"}, inputs={}, run_id=uuid4())
    row = handler.batch_processor.append.call_args_list[0].args[0]
    assert row["attributes"]["root_agent_name"] == "MyTopAgent"


def test_attributes_enriched_with_custom_tags(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Static `custom_tags` from config land on every event row."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(
            custom_tags={"env": "staging", "agent_role": "sales"},
        ),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    handler.on_chain_start(serialized={"name": "x"}, inputs={}, run_id=uuid4())
    row = handler.batch_processor.append.call_args_list[0].args[0]
    assert row["attributes"]["custom_tags"] == {
        "env": "staging",
        "agent_role": "sales",
    }


def test_attributes_session_metadata_can_be_disabled(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`log_session_metadata=False` suppresses the passthrough dump."""
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(log_session_metadata=False),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    handler.on_chain_start(
        serialized={"name": "x"},
        inputs={},
        run_id=uuid4(),
        metadata={"thread_id": "t1", "extra": "y"},
    )
    row = handler.batch_processor.append.call_args_list[0].args[0]
    assert "session_metadata" not in (row["attributes"] or {})


def test_attributes_session_metadata_excludes_promoted_keys(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`session_metadata` only carries keys we don't already promote.

    `session_id` / `user_id` / `langgraph_node` are surfaced as
    first-class columns (or in the langgraph attribute block), so the
    session_metadata dump must not duplicate them.
    """
    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    handler.on_chain_start(
        serialized={"name": "x"},
        inputs={},
        run_id=uuid4(),
        metadata={
            "session_id": "s1",
            "user_id": "u1",
            "langgraph_node": "TheCritic",
            "thread_id": "t1",
            "customer_id": "c-42",
        },
    )
    row = handler.batch_processor.append.call_args_list[0].args[0]
    session_meta = row["attributes"]["session_metadata"]
    assert session_meta == {"thread_id": "t1", "customer_id": "c-42"}


def test_llm_request_attributes_capture_llm_config_and_tools(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Mirrors ADK's per-LLM-request capture so dashboards can slice by
    temperature / available tools."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    sync_handler.on_llm_start(
        serialized={
            "name": "gemini-flash",
            "kwargs": {
                "model": "gemini-2.5-flash",
                "temperature": 0.2,
                "top_p": 0.9,
                "tools": [
                    {"name": "get_weather"},
                    {"function": {"name": "critique"}},
                ],
            },
        },
        prompts=["hi"],
        run_id=uuid4(),
    )
    row = sync_handler.batch_processor.append.call_args_list[0].args[0]
    assert row["attributes"]["llm_config"] == {"temperature": 0.2, "top_p": 0.9}
    assert row["attributes"]["tools"] == ["get_weather", "critique"]


def test_llm_response_attributes_capture_model_version_and_usage(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """ADK parity: `model_version` + raw `usage_metadata` (incl. cached
    tokens for context_cache_hit_rate) land on LLM_RESPONSE attributes."""
    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    sync_handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    msg = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "total_tokens": 125,
            "cached_content_token_count": 30,
        },
        response_metadata={
            "model_version": "gemini-2.5-flash-001",
            "cache_metadata": {"cached": True},
        },
    )
    response = LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output=None,
    )
    sync_handler.on_llm_end(response, run_id=uuid4())
    attrs = sync_handler.batch_processor.append.call_args_list[0].args[0]["attributes"]
    assert attrs["model_version"] == "gemini-2.5-flash-001"
    assert attrs["usage_metadata"]["cached_content_token_count"] == 30
    assert attrs["cache_metadata"] == {"cached": True}


def test_content_formatter_hook_runs_before_parsing(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`config.content_formatter` lets users redact / coerce content
    before the parser sees it. Mirrors ADK's content_formatter hook."""
    seen: list[tuple[Any, str]] = []

    def redact(content: Any, event_type: str) -> Any:
        seen.append((content, event_type))
        if isinstance(content, str):
            return "[REDACTED]"
        return content

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(content_formatter=redact),
    )
    handler._ensure_started()
    if not handler.batch_processor:
        raise ValueError("Batch processor not initialized")
    handler.batch_processor.append = MagicMock()  # type: ignore[method-assign]

    handler.on_text("hello world", run_id=uuid4())

    assert seen and seen[0][1] == "TEXT"
    row = handler.batch_processor.append.call_args_list[0].args[0]
    assert "[REDACTED]" in (row["content"] or {}).get("summary", "")


def test_flush_method_exists_on_both_handlers(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`flush()` lets callers ensure durability between requests without
    tearing the handler down."""
    assert callable(getattr(sync_handler, "flush", None))
    sync_handler.flush(timeout=0.1)  # No queued rows; should be a quick no-op.


@pytest.mark.asyncio
async def test_async_flush_method_exists(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    assert callable(getattr(handler, "flush", None))
    await handler.flush(timeout=0.1)


# ==============================================================================
# ADK PARITY: AUTO SCHEMA UPGRADE
# ==============================================================================


def test_new_table_gets_schema_version_label(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Tables created by the handler carry the schema-version label so the
    auto-upgrade path can short-circuit on subsequent runs."""
    from langchain_google_community.callbacks.bigquery_callback import (
        _SCHEMA_VERSION,
        _SCHEMA_VERSION_LABEL_KEY,
    )

    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.get_table.side_effect = mock_bigquery_clients[
        "mock_cloud_exceptions"
    ].NotFound("missing")

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    handler._ensure_started()

    create_call = mock_bq_client.create_table.call_args
    assert create_call is not None
    created_table = create_call.args[0]
    assert created_table.labels == {_SCHEMA_VERSION_LABEL_KEY: _SCHEMA_VERSION}


def test_auto_schema_upgrade_skipped_when_label_matches(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Existing tables already on the current schema version don't get
    ALTER TABLE'd a second time (idempotent fast-path)."""
    from langchain_google_community.callbacks.bigquery_callback import (
        _SCHEMA_VERSION,
        _SCHEMA_VERSION_LABEL_KEY,
    )

    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    existing = MagicMock()
    existing.labels = {_SCHEMA_VERSION_LABEL_KEY: _SCHEMA_VERSION}
    existing.schema = []
    mock_bq_client.get_table.return_value = existing

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(create_views=False),
    )
    handler._ensure_started()

    mock_bq_client.update_table.assert_not_called()


def test_auto_schema_upgrade_disabled_skips_alter(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`auto_schema_upgrade=False` opts out entirely."""
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    existing = MagicMock()
    existing.labels = {}
    existing.schema = []
    mock_bq_client.get_table.return_value = existing

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(auto_schema_upgrade=False, create_views=False),
    )
    handler._ensure_started()

    mock_bq_client.update_table.assert_not_called()


# ==============================================================================
# ADK PARITY: AUTO-CREATE ANALYTICS VIEWS
# ==============================================================================


def test_auto_create_views_emits_one_query_per_event_type(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`create_views=True` issues one CREATE OR REPLACE VIEW per event
    type, prefixed with `view_prefix`."""
    from langchain_google_community.callbacks.bigquery_callback import (
        _EVENT_VIEW_DEFS,
    )

    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.get_table.side_effect = mock_bigquery_clients[
        "mock_cloud_exceptions"
    ].NotFound("missing")

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(view_prefix="vstaging"),
    )
    handler._ensure_started()

    sql_calls = [c.args[0] for c in mock_bq_client.query.call_args_list]
    assert len(sql_calls) == len(_EVENT_VIEW_DEFS)
    assert all("CREATE OR REPLACE VIEW" in s for s in sql_calls)
    assert any("vstaging_llm_request" in s for s in sql_calls)
    assert any("vstaging_tool_completed" in s for s in sql_calls)


def test_create_views_disabled_skips_query_calls(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`create_views=False` opts out cleanly — no SQL is issued."""
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.get_table.side_effect = mock_bigquery_clients[
        "mock_cloud_exceptions"
    ].NotFound("missing")

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
        config=BigQueryLoggerConfig(create_views=False),
    )
    handler._ensure_started()

    mock_bq_client.query.assert_not_called()


def test_create_view_failure_does_not_raise(
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """If `CREATE OR REPLACE VIEW` fails (permissions, syntax, …) the
    handler logs and continues — analytics must never break the agent."""
    mock_bq_client = mock_bigquery_clients["mock_bq_client"]
    mock_bq_client.get_table.side_effect = mock_bigquery_clients[
        "mock_cloud_exceptions"
    ].NotFound("missing")
    call_count = {"n": 0}

    def failing_query(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom — insufficient permissions")
        return MagicMock()

    mock_bq_client.query.side_effect = failing_query

    handler = BigQueryCallbackHandler(
        project_id="test-project",
        dataset_id="test_dataset",
        table_id="test_table",
    )
    handler._ensure_started()


# ==============================================================================
# FLUSH() DURABILITY GUARANTEES
# ==============================================================================


def test_sync_flush_waits_for_in_flight_write_to_complete(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`flush()` must not return while a batch is still being written.

    Regression for the bug where `task_done()` fired immediately after
    `get()` (before `_write_rows_with_retry`), so `_queue.join()`
    returned while the in-flight batch was still in the middle of its write.
    """
    import threading
    import time as _time

    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")

    write_started = threading.Event()
    write_completed = threading.Event()

    def slow_write(rows: Any) -> None:
        write_started.set()
        # Simulate a real BigQuery RTT — non-trivial enough that a buggy
        # flush() would return well before this completes.
        _time.sleep(0.3)
        write_completed.set()

    sync_handler.batch_processor._write_rows_with_retry = slow_write  # type: ignore[method-assign]

    sync_handler.batch_processor.append({"event_type": "TEST"})

    assert write_started.wait(timeout=2.0), "writer never picked up the row"
    assert not write_completed.is_set(), "test arrived too late; write already finished"

    sync_handler.flush(timeout=5.0)
    assert write_completed.is_set(), "flush() returned while write was still in flight"


def test_sync_flush_honors_timeout(
    sync_handler: BigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """`flush(timeout)` must return after at most `timeout` seconds even
    if the write never completes. Previously the timeout argument was
    accepted but ignored (`Queue.join()` blocks unconditionally)."""
    import threading
    import time as _time

    if not sync_handler.batch_processor:
        raise ValueError("Batch processor not initialized")

    block = threading.Event()  # never set — write hangs forever

    def hanging_write(rows: Any) -> None:
        block.wait()

    sync_handler.batch_processor._write_rows_with_retry = hanging_write  # type: ignore[method-assign]
    sync_handler.batch_processor.append({"event_type": "TEST"})

    start = _time.monotonic()
    sync_handler.flush(timeout=0.3)
    elapsed = _time.monotonic() - start

    # Generous upper bound to account for thread scheduling on busy CI.
    assert elapsed < 2.0, f"flush ignored timeout (took {elapsed:.2f}s)"
    block.set()  # let the worker exit cleanly for fixture teardown


@pytest.mark.asyncio
async def test_async_flush_waits_for_in_flight_write_to_complete(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Async equivalent of the durability guard — flush() must wait for the
    real `_write_rows_with_retry` coroutine to finish, not just for the
    queue to be drained."""
    if not handler.async_batch_processor:
        raise ValueError("Batch processor not initialized")

    write_completed = asyncio.Event()

    async def slow_write(rows: Any) -> None:
        await asyncio.sleep(0.3)
        write_completed.set()

    handler.async_batch_processor._write_rows_with_retry = slow_write  # type: ignore[method-assign]
    await handler.async_batch_processor.append({"event_type": "TEST"})

    await handler.flush(timeout=5.0)
    assert write_completed.is_set(), (
        "async flush() returned while write was still in flight"
    )


@pytest.mark.asyncio
async def test_async_cancellation_during_write_does_not_double_ack(
    handler: AsyncBigQueryCallbackHandler,
    mock_bigquery_clients: Dict[str, Any],
) -> None:
    """Cancellation mid-write must not corrupt queue accounting.

    Regression for the bug where the inner finally acked the in-flight
    batch and then the outer `except CancelledError` re-acked the same
    rows. The duplicate ack only raised `ValueError` if
    `unfinished_tasks` was already 0 — otherwise it silently decremented
    a different queued row's accounting, leaving `unfinished_tasks` and
    `qsize()` out of sync.

    Setup: 2 rows enqueued (default `batch_size=1`). The worker dequeues
    row 1 and starts a slow write while row 2 sits in the queue. We cancel
    the worker mid-write and assert the queue accounting is consistent
    afterwards.
    """
    bp = handler.async_batch_processor
    if bp is None:
        raise ValueError("Batch processor not initialized")

    write_started = asyncio.Event()
    block = asyncio.Event()  # never set — write hangs until cancelled

    async def slow_write(rows: Any) -> None:
        write_started.set()
        await block.wait()

    bp._write_rows_with_retry = slow_write  # type: ignore[method-assign]

    await bp.append({"event_type": "ROW1"})
    await bp.append({"event_type": "ROW2"})

    await asyncio.wait_for(write_started.wait(), timeout=2.0)

    # Pre-cancellation snapshot: row 2 is still queued, both rows
    # accounted for as unfinished. `_unfinished_tasks` is the CPython
    # internal counter `asyncio.Queue.join` itself reads — de facto
    # stable but not in the type stubs.
    unfinished = lambda q: q._unfinished_tasks  # type: ignore[attr-defined]  # noqa: E731
    assert bp._queue.qsize() == 1
    assert unfinished(bp._queue) == 2

    assert bp._worker_task is not None
    bp._worker_task.cancel()
    try:
        await bp._worker_task
    except asyncio.CancelledError:
        pass

    # Post-cancellation: row 1's ack happened in the inner finally; row 2
    # was never dequeued, so it must still be accounted for. The buggy
    # version would have left unfinished_tasks at 0 here.
    assert bp._queue.qsize() == 1, "row 2 should still be enqueued"
    assert unfinished(bp._queue) == 1, (
        "queue accounting corrupted: unfinished_tasks="
        f"{unfinished(bp._queue)}, expected 1 "
        "(row 2 should still be unfinished)"
    )
