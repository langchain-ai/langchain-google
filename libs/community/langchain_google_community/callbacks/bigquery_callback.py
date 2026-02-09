from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import json
import logging
import mimetypes
from queue import Empty, Full, Queue
import random
import threading
import time
from types import MappingProxyType
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
import uuid

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.utils import guard_import


def import_google_cloud_bigquery() -> Any:
  """Import `google-cloud-bigquery` and its dependencies."""
  return (
      guard_import("google.cloud.bigquery"),
      guard_import("google.auth", pip_name="google-auth"),
      guard_import("google.api_core.gapic_v1.client_info"),
      guard_import(
          "google.cloud.bigquery_storage_v1.services.big_query_write.async_client"
      ),
      guard_import("google.cloud.exceptions"),
      guard_import(
          "google.cloud.bigquery_storage_v1.services.big_query_write.client"
      ),
      guard_import("google.cloud.storage"),
      guard_import("google.cloud.bigquery.schema"),
      guard_import("google.cloud.bigquery_storage_v1.types"),
      guard_import("google.api_core.exceptions"),
      guard_import("pyarrow"),
  )


logger = logging.getLogger(__name__)

_GRPC_DEADLINE_EXCEEDED = 4
_GRPC_INTERNAL = 13
_GRPC_UNAVAILABLE = 14
_DEFAULT_TRACE_ID = "langchain-bq-agent-analytics"


def _recursive_smart_truncate(obj: Any, max_len: int) -> tuple[Any, bool]:
  """Recursively truncates string values within a dict or list."""
  if isinstance(obj, str):
    if max_len != -1 and len(obj) > max_len:
      return obj[:max_len] + "...[TRUNCATED]", True
    return obj, False
  elif isinstance(obj, dict):
    truncated_any = False
    new_dict = {}
    for k, v in obj.items():
      val, trunc = _recursive_smart_truncate(v, max_len)
      if trunc:
        truncated_any = True
      new_dict[k] = val
    return new_dict, truncated_any
  elif isinstance(obj, (list, tuple)):
    truncated_any = False
    new_list = []
    for i in obj:
      val, trunc = _recursive_smart_truncate(i, max_len)
      if trunc:
        truncated_any = True
      new_list.append(val)
    return type(obj)(new_list), truncated_any
  return obj, False


def _bigquery_schema_to_arrow_schema(
    bq_schema_list: list[Any], bq_schema_cls: Any, pa_module: Any
) -> Any:
  """Converts a BigQuery schema to a PyArrow schema."""

  # --- PyArrow Helper Functions ---
  def _pyarrow_datetime() -> Any:
    return pa_module.timestamp("us", tz=None)

  def _pyarrow_numeric() -> Any:
    return pa_module.decimal128(38, 9)

  def _pyarrow_bignumeric() -> Any:
    return pa_module.decimal256(76, 38)

  def _pyarrow_time() -> Any:
    return pa_module.time64("us")

  def _pyarrow_timestamp() -> Any:
    return pa_module.timestamp("us", tz="UTC")

  _BQ_TO_ARROW_SCALARS = MappingProxyType({
      "BOOL": pa_module.bool_,
      "BOOLEAN": pa_module.bool_,
      "BYTES": pa_module.binary,
      "DATE": pa_module.date32,
      "DATETIME": _pyarrow_datetime,
      "FLOAT": pa_module.float64,
      "FLOAT64": pa_module.float64,
      "GEOGRAPHY": pa_module.string,
      "INT64": pa_module.int64,
      "INTEGER": pa_module.int64,
      "JSON": pa_module.string,
      "NUMERIC": _pyarrow_numeric,
      "BIGNUMERIC": _pyarrow_bignumeric,
      "STRING": pa_module.string,
      "TIME": _pyarrow_time,
      "TIMESTAMP": _pyarrow_timestamp,
  })

  _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
      "GEOGRAPHY": {
          b"ARROW:extension:name": b"google:sqlType:geography",
          b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
      },
      "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
      "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
  }
  _STRUCT_TYPES = ("RECORD", "STRUCT")

  def _bigquery_to_arrow_scalars(
      bigquery_scalar: str,
  ) -> Callable[[], Any] | None:
    return _BQ_TO_ARROW_SCALARS.get(bigquery_scalar)

  def _bigquery_to_arrow_field(bigquery_field: Any) -> Any:
    arrow_type: Any = _bigquery_to_arrow_data_type(bigquery_field)
    if arrow_type:
      metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
          bigquery_field.field_type.upper() if bigquery_field.field_type else ""
      )
      nullable = bigquery_field.mode.upper() != "REQUIRED"
      return pa_module.field(
          bigquery_field.name, arrow_type, nullable=nullable, metadata=metadata
      )
    logger.warning(
        "Could not determine Arrow type for field '%s' with type '%s'.",
        bigquery_field.name,
        bigquery_field.field_type,
    )
    return None

  def _bigquery_to_arrow_struct_data_type(field: Any) -> Any:
    arrow_fields = []
    for subfield in field.fields:
      arrow_subfield = _bigquery_to_arrow_field(subfield)
      if arrow_subfield:
        arrow_fields.append(arrow_subfield)
      else:
        logger.warning(
            "Failed to convert STRUCT/RECORD field '%s' due to subfield '%s'.",
            field.name,
            subfield.name,
        )
        return None
    return pa_module.struct(arrow_fields)

  def _bigquery_to_arrow_data_type(field: Any) -> Any:
    if field.mode == "REPEATED":
      inner = _bigquery_to_arrow_data_type(
          bq_schema_cls.SchemaField(
              field.name, field.field_type, fields=field.fields
          )
      )
      return pa_module.list_(inner) if inner else None
    field_type_upper = field.field_type.upper() if field.field_type else ""
    if field_type_upper in _STRUCT_TYPES:
      return _bigquery_to_arrow_struct_data_type(field)
    constructor = _bigquery_to_arrow_scalars(field_type_upper)
    if constructor:
      return constructor()
    return None

  arrow_fields = []
  for bigquery_field in bq_schema_list:
    field = _bigquery_to_arrow_field(bigquery_field)
    if field:
      arrow_fields.append(field)

  if len(arrow_fields) != len(bq_schema_list):
    logger.error("Failed to convert schema due to one or more fields.")
    return None
  return pa_module.schema(arrow_fields)


def _get_bigquery_events_schema(bigquery_module: Any) -> list[Any]:
  """Returns the BigQuery schema for the events table."""
  return [
      bigquery_module.SchemaField(
          "timestamp",
          "TIMESTAMP",
          mode="REQUIRED",
          description="The UTC timestamp when the event occurred.",
      ),
      bigquery_module.SchemaField(
          "event_type",
          "STRING",
          mode="NULLABLE",
          description="The category of the event.",
      ),
      bigquery_module.SchemaField(
          "agent",
          "STRING",
          mode="NULLABLE",
          description="The name of the agent.",
      ),
      bigquery_module.SchemaField(
          "session_id",
          "STRING",
          mode="NULLABLE",
          description="A unique identifier for the conversation session.",
      ),
      bigquery_module.SchemaField(
          "invocation_id",
          "STRING",
          mode="NULLABLE",
          description="A unique identifier for a single turn.",
      ),
      bigquery_module.SchemaField(
          "user_id",
          "STRING",
          mode="NULLABLE",
          description="The identifier of the end-user.",
      ),
      bigquery_module.SchemaField(
          "trace_id",
          "STRING",
          mode="NULLABLE",
          description="OpenTelemetry trace ID.",
      ),
      bigquery_module.SchemaField(
          "span_id",
          "STRING",
          mode="NULLABLE",
          description="OpenTelemetry span ID.",
      ),
      bigquery_module.SchemaField(
          "parent_span_id",
          "STRING",
          mode="NULLABLE",
          description="OpenTelemetry parent span ID.",
      ),
      bigquery_module.SchemaField(
          "content",
          "JSON",
          mode="NULLABLE",
          description="The primary payload of the event.",
      ),
      bigquery_module.SchemaField(
          "content_parts",
          "RECORD",
          mode="REPEATED",
          fields=[
              bigquery_module.SchemaField(
                  "mime_type", "STRING", mode="NULLABLE"
              ),
              bigquery_module.SchemaField("uri", "STRING", mode="NULLABLE"),
              bigquery_module.SchemaField(
                  "object_ref",
                  "RECORD",
                  mode="NULLABLE",
                  fields=[
                      bigquery_module.SchemaField(
                          "uri", "STRING", mode="NULLABLE"
                      ),
                      bigquery_module.SchemaField(
                          "version", "STRING", mode="NULLABLE"
                      ),
                      bigquery_module.SchemaField(
                          "authorizer", "STRING", mode="NULLABLE"
                      ),
                      bigquery_module.SchemaField(
                          "details", "JSON", mode="NULLABLE"
                      ),
                  ],
              ),
              bigquery_module.SchemaField("text", "STRING", mode="NULLABLE"),
              bigquery_module.SchemaField(
                  "part_index", "INTEGER", mode="NULLABLE"
              ),
              bigquery_module.SchemaField(
                  "part_attributes", "STRING", mode="NULLABLE"
              ),
              bigquery_module.SchemaField(
                  "storage_mode", "STRING", mode="NULLABLE"
              ),
          ],
          description=(
              "For multi-modal events, contains a list of content parts."
          ),
      ),
      bigquery_module.SchemaField(
          "attributes",
          "JSON",
          mode="NULLABLE",
          description="Arbitrary key-value pairs.",
      ),
      bigquery_module.SchemaField(
          "latency_ms",
          "JSON",
          mode="NULLABLE",
          description="Latency measurements.",
      ),
      bigquery_module.SchemaField(
          "status",
          "STRING",
          mode="NULLABLE",
          description="The outcome of the event.",
      ),
      bigquery_module.SchemaField(
          "error_message",
          "STRING",
          mode="NULLABLE",
          description="Detailed error message.",
      ),
      bigquery_module.SchemaField(
          "is_truncated",
          "BOOLEAN",
          mode="NULLABLE",
          description="Flag indicating if content was truncated.",
      ),
  ]


def _ensure_dataset_exists(
    bigquery: Any, project_id: str, dataset_id: str, cloud_exceptions: Any
) -> None:
  client = bigquery.Client(project=project_id)
  try:
    client.get_dataset(dataset_id)
  except cloud_exceptions.NotFound:
    raise ValueError(
        f"Dataset '{dataset_id}' does not exist in project '{project_id}'. "
        "Please create it before initializing the callback handler."
    )


# ==============================================================================
# CONFIGURATION
# ==============================================================================


@dataclass
class RetryConfig:
  max_retries: int = 3
  initial_delay: float = 1.0
  multiplier: float = 2.0
  max_delay: float = 10.0


@dataclass
class LatencyMeasurement:
  """Represents a latency measurement with optional component breakdown."""

  total_ms: int
  component_ms: Optional[Dict[str, int]] = None


@dataclass
class SpanContext:
  """Represents an OpenTelemetry-style span context."""

  trace_id: str
  span_id: str
  parent_span_id: Optional[str] = None


@dataclass
class RunContext:
  """Context information for a single run/operation."""

  run_id: str
  name: str
  parent_run_id: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BigQueryLoggerConfig:
  enabled: bool = True
  event_allowlist: list[str] | None = None
  event_denylist: list[str] | None = None
  max_content_length: int = 500 * 1024
  table_id: str = "agent_events_v2"
  clustering_fields: list[str] = field(
      default_factory=lambda: ["event_type", "agent", "user_id"]
  )
  log_multi_modal_content: bool = True
  retry_config: RetryConfig = field(default_factory=RetryConfig)
  batch_size: int = 1
  batch_flush_interval: float = 1.0
  shutdown_timeout: float = 10.0
  queue_max_size: int = 10000
  gcs_bucket_name: str | None = None
  connection_id: str | None = None


# ==============================================================================
# LATENCY TRACKING
# ==============================================================================


class LatencyTracker:
  """Thread-safe latency tracker for synchronous operations."""

  def __init__(self, stale_threshold_ms: int = 300000) -> None:
    """Initialize the latency tracker.

    Args:
        stale_threshold_ms: Time in ms after which unfinished entries are
          considered stale and cleaned up. Defaults to 5 minutes.
    """
    self._start_times: Dict[uuid.UUID, float] = {}
    self._component_times: Dict[uuid.UUID, Dict[str, float]] = {}
    self._lock = threading.Lock()
    self._stale_threshold_ms = stale_threshold_ms

  def start(self, run_id: uuid.UUID) -> None:
    """Start timing for a run.

    Args:
        run_id: Unique identifier for the run.
    """
    with self._lock:
      self._cleanup_stale()
      self._start_times[run_id] = time.time()
      self._component_times[run_id] = {}

  def start_component(self, run_id: uuid.UUID, component_name: str) -> None:
    """Start timing a specific component within a run.

    Args:
        run_id: Unique identifier for the run.
        component_name: Name of the component to track.
    """
    with self._lock:
      if run_id in self._component_times:
        self._component_times[run_id][f"{component_name}_start"] = time.time()
      else:
        logger.debug(
            "LatencyTracker: start_component failed, run_id %s not found",
            run_id,
        )

  def end_component(self, run_id: uuid.UUID, component_name: str) -> None:
    """End timing a specific component within a run.

    Args:
        run_id: Unique identifier for the run.
        component_name: Name of the component to track.
    """
    with self._lock:
      if run_id in self._component_times:
        start_key = f"{component_name}_start"
        if start_key in self._component_times[run_id]:
          start_time = self._component_times[run_id].pop(start_key)
          elapsed_ms = int((time.time() - start_time) * 1000)
          self._component_times[run_id][component_name] = elapsed_ms
      else:
        logger.debug(
            "LatencyTracker: end_component failed, run_id %s not found", run_id
        )

  def end(self, run_id: uuid.UUID) -> Optional[LatencyMeasurement]:
    """End timing for a run and return the measurement.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        LatencyMeasurement with total_ms and component_ms, or None if
        the run_id was not found.
    """

    with self._lock:
      if run_id not in self._start_times:
        logger.debug(
            "LatencyTracker: run_id %s not found in start_times", run_id
        )
        return None

      start_time = self._start_times.pop(run_id)
      total_ms = int((time.time() - start_time) * 1000)

      component_ms: Optional[Dict[str, int]] = None
      if run_id in self._component_times:
        components = self._component_times.pop(run_id)
        # Filter out any unfinished component start markers
        component_ms = {
            k: int(v) for k, v in components.items() if not k.endswith("_start")
        }
        if not component_ms:
          component_ms = None

      return LatencyMeasurement(total_ms=total_ms, component_ms=component_ms)

  def _cleanup_stale(self) -> None:
    """Remove entries older than the stale threshold."""
    current_time = time.time()
    stale_threshold_s = self._stale_threshold_ms / 1000.0
    stale_ids = [
        run_id
        for run_id, start_time in self._start_times.items()
        if current_time - start_time > stale_threshold_s
    ]
    for run_id in stale_ids:
      self._start_times.pop(run_id, None)
      self._component_times.pop(run_id, None)


class AsyncLatencyTracker:
  """Async-safe latency tracker for asynchronous operations."""

  def __init__(self, stale_threshold_ms: int = 300000) -> None:
    """Initialize the async latency tracker.

    Args:
        stale_threshold_ms: Time in ms after which unfinished entries are
          considered stale and cleaned up. Defaults to 5 minutes.
    """
    self._start_times: Dict[uuid.UUID, float] = {}
    self._component_times: Dict[uuid.UUID, Dict[str, float]] = {}
    self._lock = asyncio.Lock()
    self._stale_threshold_ms = stale_threshold_ms

  async def start(self, run_id: uuid.UUID) -> None:
    """Start timing for a run.

    Args:
        run_id: Unique identifier for the run.
    """
    async with self._lock:
      self._cleanup_stale()
      self._start_times[run_id] = time.time()
      self._component_times[run_id] = {}

  async def start_component(
      self, run_id: uuid.UUID, component_name: str
  ) -> None:
    """Start timing a specific component within a run.

    Args:
        run_id: Unique identifier for the run.
        component_name: Name of the component to track.
    """
    async with self._lock:
      if run_id in self._component_times:
        self._component_times[run_id][f"{component_name}_start"] = time.time()
      else:
        logger.debug(
            "AsyncLatencyTracker: start_component failed, run_id %s not found",
            run_id,
        )

  async def end_component(self, run_id: uuid.UUID, component_name: str) -> None:
    """End timing a specific component within a run.

    Args:
        run_id: Unique identifier for the run.
        component_name: Name of the component to track.
    """
    async with self._lock:
      if run_id in self._component_times:
        start_key = f"{component_name}_start"
        if start_key in self._component_times[run_id]:
          start_time = self._component_times[run_id].pop(start_key)
          elapsed_ms = int((time.time() - start_time) * 1000)
          self._component_times[run_id][component_name] = elapsed_ms
      else:
        logger.debug(
            "AsyncLatencyTracker: end_component failed, run_id %s not found",
            run_id,
        )

  async def end(self, run_id: uuid.UUID) -> Optional[LatencyMeasurement]:
    """End timing for a run and return the measurement.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        LatencyMeasurement with total_ms and component_ms, or None if
        the run_id was not found.
    """

    async with self._lock:
      if run_id not in self._start_times:
        logger.debug(
            "AsyncLatencyTracker: run_id %s not found in start_times", run_id
        )
        return None

      start_time = self._start_times.pop(run_id)
      total_ms = int((time.time() - start_time) * 1000)

      component_ms: Optional[Dict[str, int]] = None
      if run_id in self._component_times:
        components = self._component_times.pop(run_id)
        # Filter out any unfinished component start markers
        component_ms = {
            k: int(v) for k, v in components.items() if not k.endswith("_start")
        }
        if not component_ms:
          component_ms = None

      return LatencyMeasurement(total_ms=total_ms, component_ms=component_ms)

  def _cleanup_stale(self) -> None:
    """Remove entries older than the stale threshold."""
    current_time = time.time()
    stale_threshold_s = self._stale_threshold_ms / 1000.0
    stale_ids = [
        run_id
        for run_id, start_time in self._start_times.items()
        if current_time - start_time > stale_threshold_s
    ]
    for run_id in stale_ids:
      self._start_times.pop(run_id, None)
      self._component_times.pop(run_id, None)


# ==============================================================================
# RUN CONTEXT REGISTRY
# ==============================================================================


class RunContextRegistry:
  """Thread-safe registry for tracking run context (e.g., tool names)."""

  def __init__(self) -> None:
    self._contexts: Dict[uuid.UUID, RunContext] = {}
    self._lock = threading.Lock()

  def register(
      self,
      run_id: uuid.UUID,
      name: str,
      parent_run_id: Optional[uuid.UUID] = None,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> RunContext:
    """Register a run context.

    Args:
        run_id: Unique identifier for the run.
        name: Name of the operation (e.g., tool name).
        parent_run_id: Optional parent run identifier.
        metadata: Optional additional metadata.

    Returns:
        The registered RunContext.
    """
    with self._lock:
      context = RunContext(
          run_id=str(run_id),
          name=name,
          parent_run_id=str(parent_run_id) if parent_run_id else None,
          metadata=metadata or {},
      )
      self._contexts[run_id] = context
      return context

  def get(self, run_id: uuid.UUID) -> Optional[RunContext]:
    """Get a run context without removing it.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        The RunContext if found, else None.
    """
    with self._lock:
      return self._contexts.get(run_id)

  def pop(self, run_id: uuid.UUID) -> Optional[RunContext]:
    """Get and remove a run context.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        The RunContext if found, else None.
    """
    with self._lock:
      context = self._contexts.pop(run_id, None)
      if not context:
        logger.debug(
            "RunContextRegistry: pop failed, run_id %s not found", run_id
        )
      return context

  def update_metadata(
      self, run_id: uuid.UUID, metadata: Dict[str, Any]
  ) -> Optional[RunContext]:
    """Update metadata for an existing run context.

    Args:
        run_id: Unique identifier for the run.
        metadata: Metadata to merge into the existing context.

    Returns:
        The updated RunContext if found, else None.
    """
    with self._lock:
      if run_id in self._contexts:
        self._contexts[run_id].metadata.update(metadata)
        return self._contexts[run_id]
      logger.debug(
          "RunContextRegistry: update_metadata failed, run_id %s not found",
          run_id,
      )
      return None


class AsyncRunContextRegistry:
  """Async-safe registry for tracking run context (e.g., tool names)."""

  def __init__(self) -> None:
    self._contexts: Dict[uuid.UUID, RunContext] = {}
    self._lock = asyncio.Lock()

  async def register(
      self,
      run_id: uuid.UUID,
      name: str,
      parent_run_id: Optional[uuid.UUID] = None,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> RunContext:
    """Register a run context.

    Args:
        run_id: Unique identifier for the run.
        name: Name of the operation (e.g., tool name).
        parent_run_id: Optional parent run identifier.
        metadata: Optional additional metadata.

    Returns:
        The registered RunContext.
    """
    async with self._lock:
      context = RunContext(
          run_id=str(run_id),
          name=name,
          parent_run_id=str(parent_run_id) if parent_run_id else None,
          metadata=metadata or {},
      )
      self._contexts[run_id] = context
      return context

  async def get(self, run_id: uuid.UUID) -> Optional[RunContext]:
    """Get a run context without removing it.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        The RunContext if found, else None.
    """
    async with self._lock:
      return self._contexts.get(run_id)

  async def pop(self, run_id: uuid.UUID) -> Optional[RunContext]:
    """Get and remove a run context.

    Args:
        run_id: Unique identifier for the run.

    Returns:
        The RunContext if found, else None.
    """
    async with self._lock:
      context = self._contexts.pop(run_id, None)
      if not context:
        logger.debug(
            "AsyncRunContextRegistry: pop failed, run_id %s not found", run_id
        )
      return context

  async def update_metadata(
      self, run_id: uuid.UUID, metadata: Dict[str, Any]
  ) -> Optional[RunContext]:
    """Update metadata for an existing run context.

    Args:
        run_id: Unique identifier for the run.
        metadata: Metadata to merge into the existing context.

    Returns:
        The updated RunContext if found, else None.
    """
    async with self._lock:
      if run_id in self._contexts:
        self._contexts[run_id].metadata.update(metadata)
        return self._contexts[run_id]
      logger.debug(
          "AsyncRunContextRegistry: update_metadata failed, run_id %s not"
          " found",
          run_id,
      )
      return None


# ==============================================================================
# OPENTELEMETRY TRACE MANAGER
# ==============================================================================


def _prepare_arrow_batch(rows: list[dict[str, Any]], arrow_schema: Any) -> Any:
  """Prepares a PyArrow RecordBatch from a list of rows."""
  import pyarrow as pa

  data: Dict[str, List] = {
      schema_field.name: [] for schema_field in arrow_schema
  }
  for row in rows:
    for schema_field in arrow_schema:
      value = row.get(schema_field.name)
      # JSON Handling
      field_metadata = arrow_schema.field(schema_field.name).metadata
      is_json = False
      if field_metadata and b"ARROW:extension:name" in field_metadata:
        if (
            field_metadata[b"ARROW:extension:name"]  # type: ignore
            == b"google:sqlType:json"
        ):
          is_json = True
      arrow_field_type = arrow_schema.field(schema_field.name).type
      is_struct = pa.types.is_struct(arrow_field_type)
      is_list = pa.types.is_list(arrow_field_type)

      if is_json:
        if value is not None:
          if isinstance(value, (dict, list)):
            try:
              value = json.dumps(value)
            except (TypeError, ValueError):
              value = str(value)
          elif isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
              try:
                value = value.decode("utf-8")
              except UnicodeDecodeError:
                value = str(value)

            is_already_json = False
            if isinstance(value, str):
              stripped = value.strip()
              if stripped.startswith(("{", "[")) and stripped.endswith(
                  ("}", "]")
              ):
                try:
                  json.loads(value)
                  is_already_json = True
                except (ValueError, TypeError):
                  pass

            if not is_already_json:
              try:
                value = json.dumps(value)
              except (TypeError, ValueError):
                value = str(value)
          else:
            try:
              value = json.dumps(value)
            except (TypeError, ValueError):
              value = str(value)
      elif isinstance(value, (dict, list)) and not is_struct and not is_list:
        if value is not None and not isinstance(value, (str, bytes)):
          try:
            value = json.dumps(value)
          except (TypeError, ValueError):
            value = str(value)

      data[schema_field.name].append(value)
  return pa.RecordBatch.from_pydict(data, schema=arrow_schema)


# ==============================================================================
# ASYNC CORE COMPONENTS
# ==============================================================================


class _AsyncBatchProcessor:
  """Internal.

  Handles asynchronous batching and writing of events to BigQuery.
  """

  def __init__(
      self,
      write_client: Any,
      arrow_schema: Any,
      write_stream: str,
      batch_size: int,
      flush_interval: float,
      retry_config: RetryConfig,
      queue_max_size: int,
      bq_storage_types: Any,
      service_unavailable_exception: Any,
  ):
    self.write_client = write_client
    self.arrow_schema = arrow_schema
    self.write_stream = write_stream
    self.batch_size = batch_size
    self.flush_interval = flush_interval
    self.retry_config = retry_config
    self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
        maxsize=queue_max_size
    )
    self._worker_task: asyncio.Task | None = None
    self._shutdown = False
    self.bq_storage_types = bq_storage_types
    self.service_unavailable_exception = service_unavailable_exception

  async def start(self) -> None:
    if self._worker_task is None:
      self._worker_task = asyncio.create_task(self._batch_writer())

  async def append(self, row: dict[str, Any]) -> None:
    try:
      self._queue.put_nowait(row)
    except asyncio.QueueFull:
      logger.warning("BigQuery log queue full, dropping event.")

  async def _batch_writer(self) -> None:
    while not self._shutdown or not self._queue.empty():
      batch = []
      try:
        if self._shutdown:
          try:
            batch.append(self._queue.get_nowait())
          except asyncio.QueueEmpty:
            break
        else:
          batch.append(
              await asyncio.wait_for(
                  self._queue.get(), timeout=self.flush_interval
              )
          )
        self._queue.task_done()

        while len(batch) < self.batch_size:
          try:
            batch.append(self._queue.get_nowait())
            self._queue.task_done()
          except asyncio.QueueEmpty:
            break

        if batch:
          await self._write_rows_with_retry(batch)
      except asyncio.TimeoutError:
        continue
      except asyncio.CancelledError:
        logger.info("Batch writer task cancelled.")
        break
      except Exception as e:
        if isinstance(e, RuntimeError) and "Event loop is closed" in str(e):
          break
        logger.error("Error in batch writer: %s", e, exc_info=True)
        try:
          await asyncio.sleep(1)
        except RuntimeError:
          break

  async def _write_rows_with_retry(self, rows: list[dict[str, Any]]) -> None:
    attempt = 0
    delay = self.retry_config.initial_delay
    try:
      arrow_batch = _prepare_arrow_batch(rows, self.arrow_schema)
      serialized_schema = self.arrow_schema.serialize().to_pybytes()
      serialized_batch = arrow_batch.serialize().to_pybytes()
      req = self.bq_storage_types.AppendRowsRequest(
          write_stream=self.write_stream, trace_id=_DEFAULT_TRACE_ID
      )
      req.arrow_rows.writer_schema.serialized_schema = serialized_schema
      req.arrow_rows.rows.serialized_record_batch = serialized_batch
    except Exception as e:
      logger.error(
          "Failed to prepare Arrow batch (Data Loss): %s", e, exc_info=True
      )
      return

    while attempt <= self.retry_config.max_retries:
      try:

        async def req_iter() -> AsyncIterator[Any]:
          yield req

        responses = await self.write_client.append_rows(req_iter())
        async for response in responses:
          error = getattr(response, "error", None)
          error_code = getattr(error, "code", None)
          if error_code and error_code != 0:
            error_message = getattr(error, "message", "Unknown error")
            logger.warning(
                "BigQuery Write API returned error code %s: %s",
                error_code,
                error_message,
            )
            if error_code in [
                _GRPC_DEADLINE_EXCEEDED,
                _GRPC_INTERNAL,
                _GRPC_UNAVAILABLE,
            ]:
              raise self.service_unavailable_exception(error_message)
            else:
              if "schema mismatch" in error_message.lower():
                logger.error("BigQuery Schema Mismatch: %s", error_message)
              else:
                logger.error("Non-retryable BigQuery error: %s", error_message)
                row_errors = getattr(response, "row_errors", [])
                if row_errors:
                  for row_error in row_errors:
                    logger.error("Row error details: %s", row_error)
                logger.error("Row content causing error: %s", rows)
              return
        return
      except self.service_unavailable_exception as e:
        attempt += 1
        if attempt > self.retry_config.max_retries:
          logger.error(
              "BigQuery Batch Dropped after %s attempts. Last error: %s",
              self.retry_config.max_retries + 1,
              e,
          )
          return
        sleep_time = min(
            delay * (1 + random.random()), self.retry_config.max_delay
        )
        logger.warning(
            "BigQuery write failed (Attempt %s), retrying in %.2fs... "
            "Error: %s",
            attempt,
            sleep_time,
            e,
        )
        await asyncio.sleep(sleep_time)
        delay *= self.retry_config.multiplier
      except Exception as e:
        logger.error(
            "Unexpected BigQuery Write API error (Dropping batch): %s",
            e,
            exc_info=True,
        )
        return

  async def shutdown(self, timeout: float = 5.0) -> None:
    self._shutdown = True
    logger.info("BatchProcessor shutting down, draining queue...")
    if self._worker_task:
      try:
        await asyncio.wait_for(self._worker_task, timeout=timeout)
      except asyncio.TimeoutError:
        logger.warning("BatchProcessor shutdown timed out, cancelling worker.")
        self._worker_task.cancel()
      except Exception as e:
        logger.error("Error during BatchProcessor shutdown: %s", e)


class _GCSOffloader:

  def __init__(
      self,
      project_id: str,
      bucket_name: str,
      executor: ThreadPoolExecutor,
      storage_client_cls: Any,
  ):
    self.client = storage_client_cls(project=project_id)
    self.bucket = self.client.bucket(bucket_name)
    self.executor = executor

  async def upload_content(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        self.executor,
        functools.partial(self._upload_sync, data, content_type, path),
    )

  def _upload_sync(
      self, data: bytes | str, content_type: str, path: str
  ) -> str:
    blob = self.bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{self.bucket.name}/{path}"


# ==============================================================================
# SYNC CORE COMPONENTS
# ==============================================================================


class _BatchProcessor:
  """Internal. Synchronous version of `_AsyncBatchProcessor` using threading."""

  def __init__(
      self,
      write_client: Any,
      arrow_schema: Any,
      write_stream: str,
      batch_size: int,
      flush_interval: float,
      retry_config: RetryConfig,
      queue_max_size: int,
      bq_storage_types: Any,
      service_unavailable_exception: Any,
  ) -> None:
    self.write_client = write_client
    self.arrow_schema = arrow_schema
    self.write_stream = write_stream
    self.batch_size = batch_size
    self.flush_interval = flush_interval
    self.retry_config = retry_config
    self.queue_max_size = queue_max_size
    self.bq_storage_types = bq_storage_types
    self.service_unavailable_exception = service_unavailable_exception
    self._shutdown = False
    self._queue: Queue[dict[str, Any]] = Queue(maxsize=self.queue_max_size)
    self._worker_task: threading.Thread | None = None

  def start(self) -> None:
    """Starts the background worker thread."""
    if self._worker_task is None:
      self._worker_task = threading.Thread(
          target=self._batch_writer, daemon=True
      )
      self._worker_task.start()

  def append(self, row: dict[str, Any]) -> None:
    """Adds a row to the queue for processing."""
    try:
      self._queue.put_nowait(row)
    except Full:
      logger.warning("BigQuery log queue full, dropping event.")

  def _batch_writer(self) -> None:
    """The background thread's main loop for batching and writing."""
    while not self._shutdown or not self._queue.empty():
      batch = []
      try:
        if self._shutdown:
          try:
            batch.append(self._queue.get_nowait())
          except Empty:
            break
        else:
          batch.append(self._queue.get(timeout=self.flush_interval))
        self._queue.task_done()

        while len(batch) < self.batch_size:
          try:
            batch.append(self._queue.get_nowait())
            self._queue.task_done()
          except Empty:
            break

        if batch:
          self._write_rows_with_retry(batch)
      except Empty:
        continue
      except Exception as e:
        logger.error("Error in batch writer: %s", e, exc_info=True)
        time.sleep(1)

  def _write_rows_with_retry(self, rows: list[dict[str, Any]]) -> None:
    """Writes a batch to BigQuery with retry logic."""
    attempt = 0
    delay = self.retry_config.initial_delay
    try:
      arrow_batch = _prepare_arrow_batch(rows, self.arrow_schema)
      serialized_schema = self.arrow_schema.serialize().to_pybytes()
      serialized_batch = arrow_batch.serialize().to_pybytes()
      req = self.bq_storage_types.AppendRowsRequest(
          write_stream=self.write_stream, trace_id=_DEFAULT_TRACE_ID
      )
      req.arrow_rows.writer_schema.serialized_schema = serialized_schema
      req.arrow_rows.rows.serialized_record_batch = serialized_batch
    except Exception as e:
      logger.error(
          "Failed to prepare Arrow batch (Data Loss): %s", e, exc_info=True
      )
      return

    while attempt <= self.retry_config.max_retries:
      try:
        responses = self.write_client.append_rows(iter([req]))
        for response in responses:
          error = getattr(response, "error", None)
          error_code = getattr(error, "code", None)
          if error_code and error_code != 0:
            error_message = getattr(error, "message", "Unknown error")
            logger.warning(
                "BigQuery Write API returned error code %s: %s",
                error_code,
                error_message,
            )
            if error_code in [
                _GRPC_DEADLINE_EXCEEDED,
                _GRPC_INTERNAL,
                _GRPC_UNAVAILABLE,
            ]:
              raise self.service_unavailable_exception(error_message)
            else:
              # Handle non-retryable errors as before
              return
        return
      except self.service_unavailable_exception as e:
        attempt += 1
        if attempt > self.retry_config.max_retries:
          logger.error(
              "BigQuery Batch Dropped after %s attempts. Last error: %s",
              self.retry_config.max_retries + 1,
              e,
          )
          return
        sleep_time = min(
            delay * (1 + random.random()), self.retry_config.max_delay
        )
        logger.warning(
            "BigQuery write failed (Attempt %s), retrying in %.2fs... "
            "Error: %s",
            attempt,
            sleep_time,
            e,
        )
        time.sleep(sleep_time)
        delay *= self.retry_config.multiplier
      except Exception as e:
        logger.error(
            "Unexpected BigQuery Write API error (Dropping batch): %s",
            e,
            exc_info=True,
        )
        return

  def shutdown(self, timeout: float = 5.0) -> None:
    """Shuts down the worker thread."""
    self._shutdown = True
    logger.info("BatchProcessor shutting down, draining queue...")
    if self._worker_task:
      try:
        self._worker_task.join(timeout=timeout)
        if self._worker_task.is_alive():
          logger.warning("BatchProcessor shutdown timed out.")
      except Exception as e:
        logger.error("Error during BatchProcessor shutdown: %s", e)


@dataclass
class _OffloadRequest:
  """Request to offload content to GCS."""

  data: Union[str, bytes]
  mime_type: str
  path: str
  original_text: Optional[str] = None  # For fallback if needed


class _LangChainContentParserMixin:
  """Mixin for shared LangChain content parsing logic."""

  # These fields should be available in the subclass instance
  offloader: _GCSOffloader | None
  trace_id: str
  span_id: str
  max_length: int
  connection_id: str | None
  inline_text_limit: int

  def _truncate(self, text: str) -> tuple[str, bool]:
    if self.max_length != -1 and len(text) > self.max_length:
      return text[: self.max_length] + "...[TRUNCATED]", True
    return text, False

  def _get_offload_path(self, idx: int, ext: str = ".txt") -> str:
    return f"{datetime.now().date()}/{self.trace_id}/{self.span_id}_p{idx}{ext}"

  def _prepare_content_part(
      self, idx: int, part: Union[str, Dict]
  ) -> tuple[Dict[str, Any], Optional[str], Optional[_OffloadRequest]]:
    """Prepares a single content part for logging/offloading.

    Args:
        idx: The index of the part in the list.
        part: The content part (string or dictionary).

    Returns:
        A tuple containing:
        - part_data: Initial dictionary for the part.
        - summary_text: A snippet for the summary (or None if not readily
        available).
        - offload_request: An _OffloadRequest if offloading is needed, else
        None.
    """
    part_data = {
        "part_index": idx,
        "mime_type": "text/plain",
        "uri": None,
        "text": None,
        "part_attributes": "{}",
        "storage_mode": "INLINE",
    }
    summary_text = None
    offload_request = None

    # Handle String Part
    if isinstance(part, str):
      text_len = len(part.encode("utf-8"))
      if self.offloader and text_len > self.inline_text_limit:
        path = self._get_offload_path(idx)
        offload_request = _OffloadRequest(
            data=part, mime_type="text/plain", path=path, original_text=part
        )
      else:
        clean, trunc = self._truncate(part)
        part_data["text"] = clean
        summary_text = clean

    # Handle Dict Part (Multi-Modal)
    elif isinstance(part, dict):
      part_type = part.get("type")

      if part_type == "text":
        text_val = part.get("text", "")
        text_len = len(text_val.encode("utf-8"))
        if self.offloader and text_len > self.inline_text_limit:
          path = self._get_offload_path(idx)
          offload_request = _OffloadRequest(
              data=text_val,
              mime_type="text/plain",
              path=path,
              original_text=text_val,
          )
        else:
          clean, trunc = self._truncate(text_val)
          part_data["text"] = clean
          summary_text = clean

      elif part_type == "image_url":
        img_url_obj = part.get("image_url", {})
        url = (
            img_url_obj.get("url")
            if isinstance(img_url_obj, dict)
            else img_url_obj
        )

        part_data["mime_type"] = "image/jpeg"  # Default/Guess
        if url and url.startswith("data:"):
          # Base64 Image
          if self.offloader:
            try:
              header, encoded = url.split(",", 1)
              mime_type = header.split(":")[1].split(";")[0]
              import base64

              data = base64.b64decode(encoded)
              ext = mimetypes.guess_extension(mime_type) or ".bin"
              path = self._get_offload_path(idx, ext)
              offload_request = _OffloadRequest(
                  data=data, mime_type=mime_type, path=path
              )
            except Exception as e:
              logger.warning("Failed to parse base64 image: %s", e)
              part_data["text"] = "[UPLOAD FAILED]"
          else:
            part_data["text"] = "[BASE64 IMAGE]"
        elif url:
          part_data["uri"] = url
          part_data["storage_mode"] = "EXTERNAL_URI"
          part_data["text"] = "[IMAGE URL]"

        summary_text = "[IMAGE]"

      elif part_type == "tool_use":
        part_data["mime_type"] = "application/json"
        part_data["text"] = f"Tool Call: {part.get('name')}"
        part_data["part_attributes"] = json.dumps(
            {"tool_id": part.get("id"), "name": part.get("name")}
        )
        summary_text = f"[TOOL: {part.get('name')}]"

    return part_data, summary_text, offload_request


class _LangChainContentParser(_LangChainContentParserMixin):
  """Internal. Parses LangChain content (including Multi-Modal) for logging."""

  def __init__(
      self,
      offloader: _GCSOffloader | None,
      trace_id: str,
      span_id: str,
      max_length: int = 20000,
      connection_id: str | None = None,
  ):
    self.offloader = offloader
    self.trace_id = trace_id
    self.span_id = span_id
    self.max_length = max_length
    self.connection_id = connection_id
    self.inline_text_limit = 32 * 1024

  async def parse_message_content(
      self, content: Union[str, List[Union[str, Dict]]]
  ) -> tuple[str, list[dict], bool]:
    """Parses LangChain Message Content (string or list of dicts)."""
    content_parts = []
    summary_text = []
    is_truncated = False

    # Normalize input to list of parts
    raw_parts: List[Union[str, Dict]]
    if isinstance(content, str):
      raw_parts = [content]
    elif isinstance(content, list):
      raw_parts = content
    else:
      raw_parts = [str(content)]

    for idx, part in enumerate(raw_parts):
      part_data, snippet, offload_req = self._prepare_content_part(idx, part)

      if snippet:
        summary_text.append(snippet)
        if hasattr(self, "_truncate") and snippet.endswith("...[TRUNCATED]"):
          is_truncated = True

      if offload_req and self.offloader:
        try:
          uri = await self.offloader.upload_content(
              offload_req.data, offload_req.mime_type, offload_req.path
          )
          part_data["storage_mode"] = "GCS_REFERENCE"
          part_data["uri"] = uri
          object_ref = {"uri": uri}
          if self.connection_id:
            object_ref["authorizer"] = self.connection_id
          part_data["object_ref"] = object_ref

          if offload_req.mime_type == "text/plain":
            preview = str(offload_req.data)[:200] + "... [OFFLOADED]"
            part_data["text"] = preview
          else:
            part_data["text"] = "[MEDIA OFFLOADED]"
            if offload_req.mime_type != "text/plain":
              part_data["mime_type"] = offload_req.mime_type

        except Exception as e:
          logger.warning("Failed to offload content to GCS: %s", e)
          if offload_req.original_text:
            clean, trunc = self._truncate(offload_req.original_text)
            part_data["text"] = clean
            summary_text.append(clean)
            if trunc:
              is_truncated = True
          else:
            part_data["text"] = "[UPLOAD FAILED]"

      content_parts.append(part_data)

    full_summary = " | ".join(summary_text)
    return full_summary, content_parts, is_truncated


class _SyncLangChainContentParser(_LangChainContentParserMixin):
  """Internal.

  A purely synchronous parser that re-implements the parsing logic without using
  asyncio. It uses a synchronous GCS offloader.
  """

  def __init__(
      self,
      offloader: _GCSOffloader | None,
      trace_id: str,
      span_id: str,
      max_length: int = 20000,
      connection_id: str | None = None,
  ):
    # This class mirrors _LangChainContentParser but is fully synchronous.
    self.offloader = offloader
    self.trace_id = trace_id
    self.span_id = span_id
    self.max_length = max_length
    self.connection_id = connection_id
    self.inline_text_limit = 32 * 1024

  def parse_message_content(
      self, content: Union[str, List[Union[str, Dict]]]
  ) -> tuple[str, list[dict], bool]:
    """Synchronously parses LangChain Message Content."""
    content_parts = []
    summary_text = []
    is_truncated = False

    raw_parts: List[Union[str, Dict]]
    if isinstance(content, str):
      raw_parts = [content]
    elif isinstance(content, list):
      raw_parts = content
    else:
      raw_parts = [str(content)]

    for idx, part in enumerate(raw_parts):
      part_data, snippet, offload_req = self._prepare_content_part(idx, part)

      if snippet:
        summary_text.append(snippet)
        if hasattr(self, "_truncate") and snippet.endswith("...[TRUNCATED]"):
          is_truncated = True

      if offload_req and self.offloader:
        try:
          # Sync implementation uses internal _upload_sync or we should expose a sync method?
          # _GCSOffloader has _upload_sync.
          uri = self.offloader._upload_sync(
              offload_req.data, offload_req.mime_type, offload_req.path
          )
          part_data["storage_mode"] = "GCS_REFERENCE"
          part_data["uri"] = uri
          object_ref = {"uri": uri}
          if self.connection_id:
            object_ref["authorizer"] = self.connection_id
          part_data["object_ref"] = object_ref

          if offload_req.mime_type == "text/plain":
            preview = str(offload_req.data)[:200] + "... [OFFLOADED]"
            part_data["text"] = preview
          else:
            part_data["text"] = "[MEDIA OFFLOADED]"
            if offload_req.mime_type != "text/plain":
              part_data["mime_type"] = offload_req.mime_type
        except Exception as e:
          logger.warning("Failed to offload content to GCS: %s", e)
          if offload_req.original_text:
            clean, trunc = self._truncate(offload_req.original_text)
            part_data["text"] = clean
            summary_text.append(clean)
            if trunc:
              is_truncated = True
          else:
            part_data["text"] = "[UPLOAD FAILED]"

      content_parts.append(part_data)

    full_summary = " | ".join(summary_text)
    return full_summary, content_parts, is_truncated


class BaseTraceIdRegistry:
  """Base Registry to manage run_id to root_run_id mapping for trace correlation."""

  def __init__(self) -> None:
    self._run_map: Dict[uuid.UUID, uuid.UUID] = {}
    self._root_map: Dict[uuid.UUID, set[uuid.UUID]] = {}

  def _register_run(
      self, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None
  ) -> str:
    """Core logic for registering a run."""
    if run_id in self._run_map:
      return str(self._run_map[run_id])

    if parent_run_id is None:
      root_id = run_id
    else:
      root_id = self._run_map.get(parent_run_id, parent_run_id)

    self._run_map[run_id] = root_id

    if root_id not in self._root_map:
      self._root_map[root_id] = set()
    self._root_map[root_id].add(run_id)
    return str(root_id)

  def _end_run(self, run_id: uuid.UUID) -> None:
    """Core logic for ending a run."""
    if run_id in self._root_map:
      descendants = self._root_map.pop(run_id)
      for desc_id in descendants:
        self._run_map.pop(desc_id, None)


class TraceIdRegistry(BaseTraceIdRegistry):
  """Registry to manage run_id to root_run_id mapping for trace correlation."""

  def __init__(self) -> None:
    super().__init__()
    self._lock = threading.Lock()

  def register_run(
      self, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None
  ) -> str:
    """Registers a run and returns its trace ID.

    If parent_run_id is provided, the run is associated with the parent's trace.
    Otherwise, a new trace is started with this run as the root.
    """
    with self._lock:
      return self._register_run(run_id, parent_run_id)

  def end_run(self, run_id: uuid.UUID) -> None:
    """Cleans up resources for a trace when the root run completes."""
    with self._lock:
      self._end_run(run_id)


class AsyncTraceIdRegistry(BaseTraceIdRegistry):
  """Async Registry to manage run_id to root_run_id mapping for trace correlation."""

  def __init__(self) -> None:
    super().__init__()
    self._lock = asyncio.Lock()

  async def register_run(
      self, run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None
  ) -> str:
    """Registers a run and returns its trace ID.

    If parent_run_id is provided, the run is associated with the parent's trace.
    Otherwise, a new trace is started with this run as the root.
    """
    async with self._lock:
      return self._register_run(run_id, parent_run_id)

  async def end_run(self, run_id: uuid.UUID) -> None:
    """Cleans up resources for a trace when the root run completes."""
    async with self._lock:
      self._end_run(run_id)


# ==============================================================================
# ASYNC CALLBACK HANDLER
# ==============================================================================


class AsyncBigQueryCallbackHandler(AsyncCallbackHandler):
  """Callback handler for logging LangChain events to Google BigQuery."""

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str | None = None,
      config: BigQueryLoggerConfig | None = None,
      *,
      graph_name: str | None = None,
  ) -> None:
    super().__init__()
    (
        self.bigquery,
        self.google_auth,
        self.gapic_client_info,
        self.async_client,
        self.cloud_exceptions,
        self.sync_write_client_module,
        self.storage,
        self.bq_schema,
        self.bq_storage_types,
        self.api_core_exceptions,
        self.pa,
    ) = import_google_cloud_bigquery()

    self.project_id = project_id
    self.dataset_id = dataset_id
    self.config = config or BigQueryLoggerConfig()
    if table_id:
      self.config.table_id = table_id

    # LangGraph support
    self.graph_name = graph_name

    # Execution order context variable (instance-scoped via ContextVar)
    # Using id(self) in the name ensures that if multiple handlers are used in the same context,
    # their ContextVars don't collide.
    self._execution_order_cv: contextvars.ContextVar[int] = (
        contextvars.ContextVar(f"execution_order_{id(self)}", default=0)
    )

    self._started: bool = False
    self._is_shutting_down: bool = False
    self._setup_lock: asyncio.Lock = asyncio.Lock()

    self.client: Any = None
    self.write_client: Any = None
    self.async_batch_processor: _AsyncBatchProcessor | None = None
    self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    self.offloader: _GCSOffloader | None = None
    self.trace_registry = AsyncTraceIdRegistry()
    self._arrow_schema: Any = None

    # New tracking components
    self._latency_tracker = AsyncLatencyTracker()
    self._run_context_registry = AsyncRunContextRegistry()

    _ensure_dataset_exists(
        self.bigquery, self.project_id, self.dataset_id, self.cloud_exceptions
    )

  def _should_log_event(self, event_type: str) -> bool:
    """Check if an event type should be logged based on allowlist/denylist.

    Args:
        event_type: The type of event to check.

    Returns:
        True if the event should be logged, False otherwise.
    """
    if self.config.event_denylist and event_type in self.config.event_denylist:
      logger.debug("Event type %s in denylist, skipping.", event_type)
      return False
    if (
        self.config.event_allowlist
        and event_type not in self.config.event_allowlist
    ):
      logger.debug("Event type %s not in allowlist, skipping.", event_type)
      return False
    return True

  def _is_langgraph_root_invocation(
      self,
      serialized: Optional[Dict[str, Any]],
      parent_run_id: Optional[uuid.UUID],
      metadata: Optional[Dict[str, Any]],
  ) -> bool:
    """Detect if this is a LangGraph root graph invocation.

    Args:
        serialized: Serialized component data.
        parent_run_id: Parent run ID if any.
        metadata: Run metadata.

    Returns:
        True if this is a LangGraph root invocation.
    """
    # Handle None serialized
    if serialized is None:
      return False

    # Check for "Graph" in the name
    name = serialized.get("name", "") or ""
    if "Graph" not in name:
      return False

    # Root invocation has no parent
    if parent_run_id is not None:
      return False

    # Check for LangGraph-specific metadata keys
    if metadata:
      langgraph_keys = {
          "langgraph_step",
          "langgraph_node",
          "langgraph_triggers",
      }
      if any(key in metadata for key in langgraph_keys):
        return True

    return True  # If name contains Graph and no parent, likely a root

  def _build_langgraph_attributes(
      self,
      node_name: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Build LangGraph-specific attributes for logging.

    Args:
        node_name: The name of the current node.
        metadata: Run metadata containing LangGraph info.
        **kwargs: Additional attributes to include.

    Returns:
        Dict containing langgraph-specific attributes.
    """
    langgraph_attrs: Dict[str, Any] = {}

    if self.graph_name:
      langgraph_attrs["graph_name"] = self.graph_name

    if node_name:
      langgraph_attrs["node_name"] = node_name

    if metadata:
      if "langgraph_node" in metadata:
        langgraph_attrs["node_name"] = metadata["langgraph_node"]
      if "langgraph_step" in metadata:
        langgraph_attrs["step"] = metadata["langgraph_step"]
      if "langgraph_triggers" in metadata:
        langgraph_attrs["triggers"] = metadata["langgraph_triggers"]
      if "langgraph_path" in metadata:
        langgraph_attrs["path"] = metadata["langgraph_path"]

    # Add execution order
    langgraph_attrs["execution_order"] = self._get_execution_order()

    # Add any additional kwargs
    langgraph_attrs.update(kwargs)

    return {"langgraph": langgraph_attrs} if langgraph_attrs else {}

  def _build_content(
      self,
      event_type: str,
      raw_content: Any,
      metadata: Optional[Dict[str, Any]],
  ) -> Dict[str, Any]:
    """Build structured content for ADK alignment.

    Args:
        event_type: The type of event.
        raw_content: The raw content to structure.
        metadata: Additional metadata.

    Returns:
        Structured content dict.
    """
    content: Dict[str, Any] = {}

    if event_type == "LLM_REQUEST":
      if isinstance(raw_content, dict):
        if "prompts" in raw_content:
          content["prompt"] = raw_content["prompts"]
        elif "messages" in raw_content:
          content["messages"] = raw_content["messages"]
        if metadata and "system_prompt" in metadata:
          content["system_prompt"] = metadata["system_prompt"]
      else:
        content["prompt"] = raw_content

    elif event_type == "LLM_RESPONSE":
      content["response"] = raw_content
      if metadata and "usage" in metadata:
        content["usage"] = metadata["usage"]

    elif event_type in ("TOOL_STARTING", "NODE_STARTING"):
      if isinstance(raw_content, dict):
        content.update(raw_content)
      else:
        content["input"] = raw_content

    elif event_type in ("TOOL_COMPLETED", "NODE_COMPLETED"):
      if isinstance(raw_content, dict):
        content.update(raw_content)
      else:
        content["result"] = raw_content

    else:
      content["data"] = raw_content

    return content

  def _get_execution_order(self) -> int:
    """Get the current execution order."""
    return self._execution_order_cv.get()

  def _increment_execution_order(self) -> int:
    """Increment and return the execution order."""
    current = self._execution_order_cv.get()
    new_order = current + 1
    self._execution_order_cv.set(new_order)
    return new_order

  def _reset_execution_order(self) -> None:
    """Reset the execution order to 0."""
    self._execution_order_cv.set(0)

  async def _ensure_started(self) -> None:
    if self._started:
      return
    async with self._setup_lock:
      if self._started:
        return
      loop = asyncio.get_running_loop()

      self.client = await loop.run_in_executor(
          self._executor,
          lambda: self.bigquery.Client(project=self.project_id),
      )

      full_table_id = (
          f"{self.project_id}.{self.dataset_id}.{self.config.table_id}"
      )
      schema = _get_bigquery_events_schema(self.bigquery)
      await loop.run_in_executor(
          self._executor,
          lambda: self._ensure_table_exists(full_table_id, schema),
      )

      creds, _ = await loop.run_in_executor(
          self._executor,
          lambda: self.google_auth.default(
              scopes=["https://www.googleapis.com/auth/cloud-platform"]
          ),
      )
      client_info = self.gapic_client_info.ClientInfo(
          user_agent=_DEFAULT_TRACE_ID
      )
      self.write_client = self.async_client.BigQueryWriteAsyncClient(
          credentials=creds, client_info=client_info
      )
      write_stream = (
          f"projects/{self.project_id}/datasets/{self.dataset_id}/"
          f"tables/{self.config.table_id}/_default"
      )

      arrow_schema = _bigquery_schema_to_arrow_schema(
          schema, self.bq_schema, self.pa
      )
      if self.config.gcs_bucket_name:
        self.offloader = _GCSOffloader(
            self.project_id,
            self.config.gcs_bucket_name,
            self._executor,
            self.storage.Client,
        )

      self.async_batch_processor = _AsyncBatchProcessor(
          self.write_client,
          arrow_schema,
          write_stream,
          self.config.batch_size,
          self.config.batch_flush_interval,
          self.config.retry_config,
          self.config.queue_max_size,
          self.bq_storage_types,
          self.api_core_exceptions.ServiceUnavailable,
      )
      await self.async_batch_processor.start()
      self._started = True

  def _ensure_table_exists(self, table_id: str, schema: List[Any]) -> Any:
    if self.client is None:
      raise ValueError("BigQuery client is not initialized.")
    try:
      self.client.get_table(table_id)
    except self.cloud_exceptions.NotFound:
      tbl = self.bigquery.Table(table_id, schema=schema)
      tbl.time_partitioning = self.bigquery.TimePartitioning(
          type_=self.bigquery.TimePartitioningType.DAY, field="timestamp"
      )
      tbl.clustering_fields = self.config.clustering_fields
      self.client.create_table(tbl)

  async def _log(
      self,
      event_type: str,
      run_id: uuid.UUID,
      content: Any = None,
      parent_run_id: uuid.UUID | None = None,
      attributes: dict | None = None,
      error: str | None = None,
      latency: int | None = None,
      metadata: dict | None = None,
      *,
      latency_measurement: Optional[LatencyMeasurement] = None,
  ) -> None:
    if not self.config.enabled:
      return

    # Event filtering based on allowlist/denylist
    if not self._should_log_event(event_type):
      return

    await self._ensure_started()

    metadata = metadata or {}
    session_id = metadata.get("session_id")
    user_id = metadata.get("user_id")
    agent = metadata.get("agent")

    registry_trace_id = await self.trace_registry.register_run(
        run_id, parent_run_id
    )
    trace_id = metadata.get("trace_id") or registry_trace_id or str(run_id)
    span_id = str(run_id)

    parser = _LangChainContentParser(
        self.offloader,
        trace_id,
        span_id,
        self.config.max_content_length,
        connection_id=self.config.connection_id,
    )

    summary_text = ""
    content_parts = []
    is_truncated = False
    parsing_error = None

    try:
      if isinstance(content, dict) and "messages" in content:
        # Handle Chat Model Messages (Multi-Modal Potential)
        all_parts = []
        # Flatten all messages to find parts
        for msg in content["messages"]:
          msg_content = msg.get("content")
          s, p, t = await parser.parse_message_content(msg_content)
          if t:
            is_truncated = True
          all_parts.extend(p)
          if summary_text:
            summary_text += " | "
          summary_text += s
        content_parts = all_parts

      elif isinstance(content, dict) and "prompts" in content:
        # Legacy LLM (list of strings)
        for p_str in content["prompts"]:
          s, p, t = await parser.parse_message_content(p_str)
          if t:
            is_truncated = True
          content_parts.extend(p)
          if summary_text:
            summary_text += " | "
          summary_text += s

      elif isinstance(content, str):
        (
            summary_text,
            content_parts,
            is_truncated,
        ) = await parser.parse_message_content(content)

      else:
        # Fallback
        summary_text, is_truncated = _recursive_smart_truncate(
            str(content), self.config.max_content_length
        )
    except Exception as e:
      parsing_error = f"Failed to parse content: {e}"
      logger.warning("%s for run_id %s", parsing_error, run_id)
      summary_text, is_truncated = _recursive_smart_truncate(
          str(content), self.config.max_content_length
      )
      content_parts = []

    # Build latency_ms from measurement or legacy parameter
    latency_ms_value: Optional[Dict[str, Any]] = None
    if latency_measurement:
      latency_ms_value = {"total_ms": latency_measurement.total_ms}
      if latency_measurement.component_ms:
        latency_ms_value["component_ms"] = latency_measurement.component_ms
    elif latency:
      latency_ms_value = {"total_ms": latency}

    row = {
        "timestamp": datetime.now(timezone.utc),
        "event_type": event_type,
        "agent": agent,
        "session_id": session_id,
        "invocation_id": str(run_id),
        "user_id": user_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": str(parent_run_id) if parent_run_id else None,
        "content": {
            **self._build_content(event_type, content, metadata),
            "summary": summary_text,
        },  # Store summary in main content JSON
        "content_parts": (
            content_parts if self.config.log_multi_modal_content else []
        ),
        "attributes": attributes,
        "latency_ms": latency_ms_value,
        "status": "ERROR" if error or parsing_error else "OK",
        "error_message": error or parsing_error,
        "is_truncated": is_truncated,
    }
    if self.async_batch_processor is None:
      raise ValueError("Batch processor is not initialized.")
    await self.async_batch_processor.append(row)

  async def shutdown(self) -> None:
    if self._is_shutting_down:
      return
    try:
      self._is_shutting_down = True
      if self.async_batch_processor:
        await self.async_batch_processor.shutdown(self.config.shutdown_timeout)
      self._executor.shutdown(wait=True)
    finally:
      self._is_shutting_down = False

  async def __aenter__(self) -> "AsyncBigQueryCallbackHandler":
    await self._ensure_started()
    return self

  async def __aexit__(self, *args: Any) -> None:
    await self.shutdown()

  # --- Callbacks ---

  async def on_llm_start(
      self,
      serialized: Dict[str, Any],
      prompts: List[str],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      tags: List[str] | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    await self._latency_tracker.start(run_id)

    model_name = serialized.get("kwargs", {}).get("model") or serialized.get(
        "name"
    )
    await self._log(
        "LLM_REQUEST",
        run_id,
        content={"prompts": prompts},
        parent_run_id=parent_run_id,
        attributes={"tags": tags, "model": model_name},
        metadata=kwargs.get("metadata"),
    )

  async def on_chat_model_start(
      self,
      serialized: Dict[str, Any],
      messages: List[List[BaseMessage]],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      tags: List[str] | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    await self._latency_tracker.start(run_id)

    model_name = serialized.get("kwargs", {}).get("model") or serialized.get(
        "name"
    )
    # Serialize messages safely for parsing
    flat_msgs = [m.model_dump() for sub in messages for m in sub]
    await self._log(
        "LLM_REQUEST",
        run_id,
        content={"messages": flat_msgs},
        parent_run_id=parent_run_id,
        attributes={"tags": tags, "model": model_name},
        metadata=kwargs.get("metadata"),
    )

  async def on_llm_end(
      self,
      response: LLMResult,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    if response.generations and response.generations[0]:
      resp_text = response.generations[0][0].text
    else:
      resp_text = ""
    usage = (
        response.llm_output.get("token_usage") if response.llm_output else None
    )
    await self._log(
        "LLM_RESPONSE",
        run_id,
        content=resp_text,
        parent_run_id=parent_run_id,
        attributes={"usage": usage},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_llm_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    await self._log(
        "LLM_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_chain_start(
      self,
      serialized: Dict[str, Any],
      inputs: Dict[str, Any],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    await self._latency_tracker.start(run_id)

    metadata = kwargs.get("metadata") or {}

    # Detect LangGraph node vs regular chain
    langgraph_node = metadata.get("langgraph_node")
    is_graph_root = self._is_langgraph_root_invocation(
        serialized, parent_run_id, metadata
    )

    if is_graph_root:
      # This is a graph root invocation
      self._reset_execution_order()
      event_type = "GRAPH_START"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    elif langgraph_node:
      # This is a LangGraph node
      self._increment_execution_order()
      event_type = "NODE_STARTING"
      attributes = self._build_langgraph_attributes(
          node_name=langgraph_node, metadata=metadata
      )
      # Register the node name for later retrieval
      await self._run_context_registry.register(
          run_id, langgraph_node, parent_run_id, metadata
      )
    else:
      event_type = "CHAIN_START"
      attributes = None

    await self._log(
        event_type,
        run_id,
        content=json.dumps(inputs, default=str),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
    )

  async def on_chain_end(
      self,
      outputs: Dict[str, Any],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    metadata = kwargs.get("metadata") or {}

    # Check if this was a LangGraph node
    context = await self._run_context_registry.pop(run_id)
    langgraph_node = metadata.get("langgraph_node")

    if context or langgraph_node:
      # This was a LangGraph node
      node_name = context.name if context else langgraph_node
      event_type = "NODE_COMPLETED"
      attributes = self._build_langgraph_attributes(
          node_name=node_name, metadata=metadata
      )
    elif parent_run_id is None and self.graph_name:
      # This might be graph end (no parent, graph_name set)
      event_type = "GRAPH_END"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    else:
      event_type = "CHAIN_END"
      attributes = None

    await self._log(
        event_type,
        run_id,
        content=json.dumps(outputs, default=str),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_tool_start(
      self,
      serialized: Dict[str, Any],
      input_str: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    await self._latency_tracker.start(run_id)

    # Register tool name for later retrieval in on_tool_end
    tool_name = serialized.get("name", "unknown_tool")
    await self._run_context_registry.register(
        run_id, tool_name, parent_run_id, kwargs.get("metadata")
    )

    await self._log(
        "TOOL_STARTING",
        run_id,
        content={"tool": tool_name, "input": input_str},
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
    )

  async def on_tool_end(
      self,
      output: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    # Retrieve tool name from registry
    context = await self._run_context_registry.pop(run_id)
    tool_name = context.name if context else "unknown_tool"

    await self._log(
        "TOOL_COMPLETED",
        run_id,
        content={"tool": tool_name, "result": output},
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_tool_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    # Retrieve tool name from registry
    context = await self._run_context_registry.pop(run_id)
    tool_name = context.name if context else "unknown_tool"

    await self._log(
        "TOOL_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_retriever_start(
      self,
      serialized: Dict[str, Any],
      query: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    await self._latency_tracker.start(run_id)

    await self._log(
        "RETRIEVER_START",
        run_id,
        content=query,
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  async def on_retriever_end(
      self,
      documents: Any,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    docs = [doc.model_dump() for doc in documents]
    await self._log(
        "RETRIEVER_END",
        run_id,
        content=json.dumps(docs, default=str),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_retriever_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    await self._log(
        "RETRIEVER_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  async def on_text(
      self,
      text: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    await self._log(
        "TEXT",
        run_id,
        content=text,
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  async def on_agent_action(
      self,
      action: AgentAction,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    await self._log(
        "AGENT_ACTION",
        run_id,
        content=json.dumps(
            {"tool": action.tool, "input": str(action.tool_input)}, default=str
        ),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  async def on_agent_finish(
      self,
      finish: AgentFinish,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    await self._log(
        "AGENT_FINISH",
        run_id,
        content=json.dumps({"output": finish.return_values}, default=str),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  async def on_chain_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = await self._latency_tracker.end(run_id)

    metadata = kwargs.get("metadata") or {}

    # Check if this was a LangGraph node
    context = await self._run_context_registry.pop(run_id)
    langgraph_node = metadata.get("langgraph_node")

    if context or langgraph_node:
      # This was a LangGraph node
      node_name = context.name if context else langgraph_node
      event_type = "NODE_ERROR"
      attributes = self._build_langgraph_attributes(
          node_name=node_name, metadata=metadata
      )
    elif parent_run_id is None and self.graph_name:
      # This might be graph error (no parent, graph_name set)
      event_type = "GRAPH_ERROR"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    else:
      event_type = "CHAIN_ERROR"
      attributes = None

    await self._log(
        event_type,
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
        latency_measurement=latency_measurement,
    )
    await self.trace_registry.end_run(run_id)

  def graph_context(
      self,
      graph_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> "AsyncGraphExecutionContext":
    """Create an async context manager for graph execution.

    Usage:
        async with handler.graph_context("my_graph") as ctx:
            result = await graph.ainvoke(inputs)

    Args:
        graph_name: Name of the graph being executed.
        metadata: Optional metadata to include in events.

    Returns:
        AsyncGraphExecutionContext that emits GRAPH_START/GRAPH_END events.
    """
    return AsyncGraphExecutionContext(self, graph_name, metadata)

  async def close(self) -> None:
    await self.shutdown()


# ==============================================================================
# SYNC CALLBACK HANDLER
# ==============================================================================


class BigQueryCallbackHandler(BaseCallbackHandler):
  """Callback handler for logging LangChain events to Google BigQuery."""

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str | None = None,
      config: BigQueryLoggerConfig | None = None,
      *,
      graph_name: str | None = None,
  ) -> None:
    super().__init__()
    (
        self.bigquery,
        self.google_auth,
        self.gapic_client_info,
        self.async_client,
        self.cloud_exceptions,
        self.sync_write_client_module,
        self.storage,
        self.bq_schema,
        self.bq_storage_types,
        self.api_core_exceptions,
        self.pa,
    ) = import_google_cloud_bigquery()

    self.project_id = project_id
    self.dataset_id = dataset_id
    self.config = config or BigQueryLoggerConfig()
    if table_id:
      self.config.table_id = table_id

    # LangGraph support
    self.graph_name = graph_name

    # Execution order context variable (instance-scoped via ContextVar)
    self._execution_order_cv: contextvars.ContextVar[int] = (
        contextvars.ContextVar(f"sync_execution_order_{id(self)}", default=0)
    )

    self._started: bool = False
    self._is_shutting_down: bool = False
    self._setup_lock: threading.Lock = threading.Lock()

    self.client: Any = None
    self.write_client: Any = None
    self.batch_processor: _BatchProcessor | None = None
    self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
    self.offloader: _GCSOffloader | None = None
    self.trace_registry = TraceIdRegistry()
    self._arrow_schema: Any = None

    # New tracking components
    self._latency_tracker = LatencyTracker()
    self._run_context_registry = RunContextRegistry()

    _ensure_dataset_exists(
        self.bigquery, self.project_id, self.dataset_id, self.cloud_exceptions
    )

  def _should_log_event(self, event_type: str) -> bool:
    """Check if an event type should be logged based on allowlist/denylist.

    Args:
        event_type: The type of event to check.

    Returns:
        True if the event should be logged, False otherwise.
    """
    if self.config.event_denylist and event_type in self.config.event_denylist:
      logger.debug("Event type %s in denylist, skipping.", event_type)
      return False
    if (
        self.config.event_allowlist
        and event_type not in self.config.event_allowlist
    ):
      logger.debug("Event type %s not in allowlist, skipping.", event_type)
      return False
    return True

  def _is_langgraph_root_invocation(
      self,
      serialized: Optional[Dict[str, Any]],
      parent_run_id: Optional[uuid.UUID],
      metadata: Optional[Dict[str, Any]],
  ) -> bool:
    """Detect if this is a LangGraph root graph invocation.

    Args:
        serialized: Serialized component data.
        parent_run_id: Parent run ID if any.
        metadata: Run metadata.

    Returns:
        True if this is a LangGraph root invocation.
    """
    # Handle None serialized
    if serialized is None:
      return False

    # Check for "Graph" in the name
    name = serialized.get("name", "") or ""
    if "Graph" not in name:
      return False

    # Root invocation has no parent
    if parent_run_id is not None:
      return False

    # Check for LangGraph-specific metadata keys
    if metadata:
      langgraph_keys = {
          "langgraph_step",
          "langgraph_node",
          "langgraph_triggers",
      }
      if any(key in metadata for key in langgraph_keys):
        return True

    return True  # If name contains Graph and no parent, likely a root

  def _build_langgraph_attributes(
      self,
      node_name: Optional[str] = None,
      metadata: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
  ) -> Dict[str, Any]:
    """Build LangGraph-specific attributes for logging.

    Args:
        node_name: The name of the current node.
        metadata: Run metadata containing LangGraph info.
        **kwargs: Additional attributes to include.

    Returns:
        Dict containing langgraph-specific attributes.
    """
    langgraph_attrs: Dict[str, Any] = {}

    if self.graph_name:
      langgraph_attrs["graph_name"] = self.graph_name

    if node_name:
      langgraph_attrs["node_name"] = node_name

    if metadata:
      if "langgraph_node" in metadata:
        langgraph_attrs["node_name"] = metadata["langgraph_node"]
      if "langgraph_step" in metadata:
        langgraph_attrs["step"] = metadata["langgraph_step"]
      if "langgraph_triggers" in metadata:
        langgraph_attrs["triggers"] = metadata["langgraph_triggers"]
      if "langgraph_path" in metadata:
        langgraph_attrs["path"] = metadata["langgraph_path"]

    # Add execution order
    langgraph_attrs["execution_order"] = self._get_execution_order()

    # Add any additional kwargs
    langgraph_attrs.update(kwargs)

    return {"langgraph": langgraph_attrs} if langgraph_attrs else {}

  def _build_content(
      self,
      event_type: str,
      raw_content: Any,
      metadata: Optional[Dict[str, Any]],
  ) -> Dict[str, Any]:
    """Build structured content for ADK alignment.

    Args:
        event_type: The type of event.
        raw_content: The raw content to structure.
        metadata: Additional metadata.

    Returns:
        Structured content dict.
    """
    content: Dict[str, Any] = {}

    if event_type == "LLM_REQUEST":
      if isinstance(raw_content, dict):
        if "prompts" in raw_content:
          content["prompt"] = raw_content["prompts"]
        elif "messages" in raw_content:
          content["messages"] = raw_content["messages"]
        if metadata and "system_prompt" in metadata:
          content["system_prompt"] = metadata["system_prompt"]
      else:
        content["prompt"] = raw_content

    elif event_type == "LLM_RESPONSE":
      content["response"] = raw_content
      if metadata and "usage" in metadata:
        content["usage"] = metadata["usage"]

    elif event_type in ("TOOL_STARTING", "NODE_STARTING"):
      if isinstance(raw_content, dict):
        content.update(raw_content)
      else:
        content["input"] = raw_content

    elif event_type in ("TOOL_COMPLETED", "NODE_COMPLETED"):
      if isinstance(raw_content, dict):
        content.update(raw_content)
      else:
        content["result"] = raw_content

    else:
      content["data"] = raw_content

    return content

  def _get_execution_order(self) -> int:
    """Get the current execution order."""
    return self._execution_order_cv.get()

  def _increment_execution_order(self) -> int:
    """Increment and return the execution order."""
    current = self._execution_order_cv.get()
    new_order = current + 1
    self._execution_order_cv.set(new_order)
    return new_order

  def _reset_execution_order(self) -> None:
    """Reset the execution order to 0."""
    self._execution_order_cv.set(0)

  def _ensure_started(self) -> None:
    if self._started:
      return
    with self._setup_lock:
      if self._started:
        return

      self.client = self.bigquery.Client(project=self.project_id)

      full_table_id = (
          f"{self.project_id}.{self.dataset_id}.{self.config.table_id}"
      )
      schema = _get_bigquery_events_schema(self.bigquery)
      self._ensure_table_exists(full_table_id, schema)

      creds, _ = self.google_auth.default(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )
      client_info = self.gapic_client_info.ClientInfo(
          user_agent=_DEFAULT_TRACE_ID
      )
      # Use the synchronous BigQueryWriteClient
      self.write_client = self.sync_write_client_module.BigQueryWriteClient(
          credentials=creds, client_info=client_info
      )
      write_stream = (
          f"projects/{self.project_id}/datasets/{self.dataset_id}/"
          f"tables/{self.config.table_id}/_default"
      )

      arrow_schema = _bigquery_schema_to_arrow_schema(
          schema, self.bq_schema, self.pa
      )
      if self.config.gcs_bucket_name:
        self.offloader = _GCSOffloader(
            self.project_id,
            self.config.gcs_bucket_name,
            self._executor,
            self.storage.Client,
        )

      self.batch_processor = _BatchProcessor(
          self.write_client,
          arrow_schema,
          write_stream,
          self.config.batch_size,
          self.config.batch_flush_interval,
          self.config.retry_config,
          self.config.queue_max_size,
          self.bq_storage_types,
          self.api_core_exceptions.ServiceUnavailable,
      )
      if self.batch_processor is None:
        raise ValueError("Batch processor is not initialized.")
      self.batch_processor.start()
      self._started = True

  def _ensure_table_exists(self, table_id: str, schema: List[Any]) -> Any:
    if self.client is None:
      raise ValueError("BigQuery client is not initialized.")
    try:
      self.client.get_table(table_id)
    except self.cloud_exceptions.NotFound:
      tbl = self.bigquery.Table(table_id, schema=schema)
      tbl.time_partitioning = self.bigquery.TimePartitioning(
          type_=self.bigquery.TimePartitioningType.DAY, field="timestamp"
      )
      tbl.clustering_fields = self.config.clustering_fields
      self.client.create_table(tbl)

  def _log(
      self,
      event_type: str,
      run_id: uuid.UUID,
      content: Any = None,
      parent_run_id: uuid.UUID | None = None,
      attributes: dict | None = None,
      error: str | None = None,
      latency: int | None = None,
      metadata: dict | None = None,
      *,
      latency_measurement: Optional[LatencyMeasurement] = None,
  ) -> None:
    if not self.config.enabled:
      return

    # Event filtering based on allowlist/denylist
    if not self._should_log_event(event_type):
      return

    self._ensure_started()

    metadata = metadata or {}
    session_id = metadata.get("session_id")
    user_id = metadata.get("user_id")
    agent = metadata.get("agent")

    registry_trace_id = self.trace_registry.register_run(run_id, parent_run_id)
    trace_id = metadata.get("trace_id") or registry_trace_id or str(run_id)
    span_id = str(run_id)

    parser = _SyncLangChainContentParser(
        self.offloader,
        trace_id,
        span_id,
        self.config.max_content_length,
        connection_id=self.config.connection_id,
    )

    summary_text = ""
    content_parts = []
    is_truncated = False
    parsing_error = None

    try:
      if isinstance(content, dict) and "messages" in content:
        # Handle Chat Model Messages (Multi-Modal Potential)
        all_parts = []
        # Flatten all messages to find parts
        for msg in content["messages"]:
          msg_content = msg.get("content")
          s, p, t = parser.parse_message_content(msg_content)
          if t:
            is_truncated = True
          all_parts.extend(p)
          if summary_text:
            summary_text += " | "
          summary_text += s
        content_parts = all_parts

      elif isinstance(content, dict) and "prompts" in content:
        # Legacy LLM (list of strings)
        for p_str in content["prompts"]:
          s, p, t = parser.parse_message_content(p_str)
          if t:
            is_truncated = True
          content_parts.extend(p)
          if summary_text:
            summary_text += " | "
          summary_text += s

      elif isinstance(content, str):
        (
            summary_text,
            content_parts,
            is_truncated,
        ) = parser.parse_message_content(content)
      else:
        summary_text, is_truncated = _recursive_smart_truncate(
            str(content), self.config.max_content_length
        )
    except Exception as e:
      parsing_error = f"Failed to parse content: {e}"
      logger.warning("%s for run_id %s", parsing_error, run_id)
      summary_text, is_truncated = _recursive_smart_truncate(
          str(content), self.config.max_content_length
      )
      content_parts = []

    # Build latency_ms from measurement or legacy parameter
    latency_ms_value: Optional[Dict[str, Any]] = None
    if latency_measurement:
      latency_ms_value = {"total_ms": latency_measurement.total_ms}
      if latency_measurement.component_ms:
        latency_ms_value["component_ms"] = latency_measurement.component_ms
    elif latency:
      latency_ms_value = {"total_ms": latency}

    row = {
        "timestamp": datetime.now(timezone.utc),
        "event_type": event_type,
        "agent": agent,
        "session_id": session_id,
        "invocation_id": str(run_id),
        "user_id": user_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": str(parent_run_id) if parent_run_id else None,
        "content": {
            **self._build_content(event_type, content, metadata),
            "summary": summary_text,
        },
        "content_parts": (
            content_parts if self.config.log_multi_modal_content else []
        ),
        "attributes": attributes,
        "latency_ms": latency_ms_value,
        "status": "ERROR" if error or parsing_error else "OK",
        "error_message": error or parsing_error,
        "is_truncated": is_truncated,
    }
    if self.batch_processor is None:
      raise ValueError("Batch processor is not initialized.")
    self.batch_processor.append(row)

  def shutdown(self) -> None:
    if self._is_shutting_down:
      return
    try:
      self._is_shutting_down = True
      if self.batch_processor:
        self.batch_processor.shutdown(self.config.shutdown_timeout)
      self._executor.shutdown(wait=True)
    finally:
      self._is_shutting_down = False

  def on_llm_start(
      self,
      serialized: Dict[str, Any],
      prompts: List[str],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      tags: List[str] | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    self._latency_tracker.start(run_id)

    model_name = serialized.get("kwargs", {}).get("model") or serialized.get(
        "name"
    )
    self._log(
        "LLM_REQUEST",
        run_id,
        content={"prompts": prompts},
        parent_run_id=parent_run_id,
        attributes={"tags": tags, "model": model_name},
        metadata=kwargs.get("metadata"),
    )

  def on_chat_model_start(
      self,
      serialized: Dict[str, Any],
      messages: List[List[BaseMessage]],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      tags: List[str] | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    self._latency_tracker.start(run_id)

    model_name = serialized.get("kwargs", {}).get("model") or serialized.get(
        "name"
    )
    flat_msgs = [m.model_dump() for sub in messages for m in sub]
    self._log(
        "LLM_REQUEST",
        run_id,
        content={"messages": flat_msgs},
        parent_run_id=parent_run_id,
        attributes={"tags": tags, "model": model_name},
        metadata=kwargs.get("metadata"),
    )

  def on_llm_end(
      self,
      response: LLMResult,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    if response.generations and response.generations[0]:
      resp_text = response.generations[0][0].text
    else:
      resp_text = ""
    usage = (
        response.llm_output.get("token_usage") if response.llm_output else None
    )
    self._log(
        "LLM_RESPONSE",
        run_id,
        content=resp_text,
        parent_run_id=parent_run_id,
        attributes={"usage": usage},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_chain_start(
      self,
      serialized: Dict[str, Any],
      inputs: Dict[str, Any],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    self._latency_tracker.start(run_id)

    metadata = kwargs.get("metadata") or {}

    # Detect LangGraph node vs regular chain
    langgraph_node = metadata.get("langgraph_node")
    is_graph_root = self._is_langgraph_root_invocation(
        serialized, parent_run_id, metadata
    )

    if is_graph_root:
      # This is a graph root invocation
      self._reset_execution_order()
      event_type = "GRAPH_START"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    elif langgraph_node:
      # This is a LangGraph node
      self._increment_execution_order()
      event_type = "NODE_STARTING"
      attributes = self._build_langgraph_attributes(
          node_name=langgraph_node, metadata=metadata
      )
      # Register the node name for later retrieval
      self._run_context_registry.register(
          run_id, langgraph_node, parent_run_id, metadata
      )
    else:
      event_type = "CHAIN_START"
      attributes = None

    self._log(
        event_type,
        run_id,
        content=json.dumps(inputs, default=str),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
    )

  def on_chain_end(
      self,
      outputs: Union[Dict[str, Any], Any],
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    metadata = kwargs.get("metadata") or {}

    # Check if this was a LangGraph node
    context = self._run_context_registry.pop(run_id)
    langgraph_node = metadata.get("langgraph_node")

    if context or langgraph_node:
      # This was a LangGraph node
      node_name = context.name if context else langgraph_node
      event_type = "NODE_COMPLETED"
      attributes = self._build_langgraph_attributes(
          node_name=node_name, metadata=metadata
      )
    elif parent_run_id is None and self.graph_name:
      # This might be graph end (no parent, graph_name set)
      event_type = "GRAPH_END"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    else:
      event_type = "CHAIN_END"
      attributes = None

    self._log(
        event_type,
        run_id,
        content=json.dumps(outputs, default=str),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_chain_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    metadata = kwargs.get("metadata") or {}

    # Check if this was a LangGraph node
    context = self._run_context_registry.pop(run_id)
    langgraph_node = metadata.get("langgraph_node")

    if context or langgraph_node:
      # This was a LangGraph node
      node_name = context.name if context else langgraph_node
      event_type = "NODE_ERROR"
      attributes = self._build_langgraph_attributes(
          node_name=node_name, metadata=metadata
      )
    elif parent_run_id is None and self.graph_name:
      # This might be graph error (no parent, graph_name set)
      event_type = "GRAPH_ERROR"
      attributes = self._build_langgraph_attributes(metadata=metadata)
    else:
      event_type = "CHAIN_ERROR"
      attributes = None

    self._log(
        event_type,
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        attributes=attributes,
        metadata=metadata,
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_tool_start(
      self,
      serialized: Dict[str, Any],
      input_str: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    self._latency_tracker.start(run_id)

    # Register tool name for later retrieval in on_tool_end
    tool_name = serialized.get("name", "unknown_tool")
    self._run_context_registry.register(
        run_id, tool_name, parent_run_id, kwargs.get("metadata")
    )

    self._log(
        "TOOL_STARTING",
        run_id,
        content={"tool": tool_name, "input": input_str},
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
    )

  def on_tool_end(
      self,
      output: Any,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    # Retrieve tool name from registry
    context = self._run_context_registry.pop(run_id)
    tool_name = context.name if context else "unknown_tool"

    self._log(
        "TOOL_COMPLETED",
        run_id,
        content={"tool": tool_name, "result": str(output)},
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_tool_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    # Retrieve tool name from registry
    context = self._run_context_registry.pop(run_id)
    tool_name = context.name if context else "unknown_tool"

    self._log(
        "TOOL_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        attributes={"tool_name": tool_name},
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_text(
      self,
      text: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    self._log(
        "TEXT",
        run_id,
        content=text,
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  def on_agent_action(
      self,
      action: AgentAction,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    self._log(
        "AGENT_ACTION",
        run_id,
        content=json.dumps(
            {"tool": action.tool, "input": str(action.tool_input)}, default=str
        ),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  def on_agent_finish(
      self,
      finish: AgentFinish,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    self._log(
        "AGENT_FINISH",
        run_id,
        content=json.dumps({"output": finish.return_values}, default=str),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  def on_retriever_start(
      self,
      serialized: Dict[str, Any],
      query: str,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # Start latency tracking
    self._latency_tracker.start(run_id)

    self._log(
        "RETRIEVER_START",
        run_id,
        content=query,
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
    )

  def on_retriever_end(
      self,
      documents: Any,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    docs = [doc.model_dump() for doc in documents]
    self._log(
        "RETRIEVER_END",
        run_id,
        content=json.dumps(docs, default=str),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_retriever_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    self._log(
        "RETRIEVER_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def on_llm_error(
      self,
      error: BaseException,
      *,
      run_id: uuid.UUID,
      parent_run_id: uuid.UUID | None = None,
      **kwargs: Any,
  ) -> None:
    # End latency tracking
    latency_measurement = self._latency_tracker.end(run_id)

    self._log(
        "LLM_ERROR",
        run_id,
        error=str(error),
        parent_run_id=parent_run_id,
        metadata=kwargs.get("metadata"),
        latency_measurement=latency_measurement,
    )
    self.trace_registry.end_run(run_id)

  def graph_context(
      self,
      graph_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> "GraphExecutionContext":
    """Create a context manager for graph execution.

    Usage:
        with handler.graph_context("my_graph") as ctx:
            result = graph.invoke(inputs)

    Args:
        graph_name: Name of the graph being executed.
        metadata: Optional metadata to include in events.

    Returns:
        GraphExecutionContext that emits GRAPH_START/GRAPH_END events.
    """
    return GraphExecutionContext(self, graph_name, metadata)

  def close(self) -> None:
    self.shutdown()


# ==============================================================================
# GRAPH EXECUTION CONTEXT MANAGERS
# ==============================================================================


class GraphExecutionContext:
  """Context manager for wrapping graph execution with GRAPH_START/GRAPH_END events.

  Usage:
      with handler.graph_context("my_graph") as ctx:
          # Graph execution happens here
          result = graph.invoke(inputs)

  This will emit GRAPH_START when entering and GRAPH_END when exiting.
  If an exception occurs, GRAPH_ERROR will be emitted instead of GRAPH_END.
  """

  def __init__(
      self,
      handler: BigQueryCallbackHandler,
      graph_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Initialize the graph execution context.

    Args:
        handler: The BigQueryCallbackHandler instance.
        graph_name: Name of the graph being executed.
        metadata: Optional metadata to include in events.
    """
    self.handler = handler
    self.graph_name = graph_name
    self.metadata = metadata or {}
    self._run_id: Optional[uuid.UUID] = None
    self._original_graph_name: Optional[str] = None

  def __enter__(self) -> "GraphExecutionContext":
    """Enter the context and emit GRAPH_START event."""
    self._run_id = uuid.uuid4()
    self._original_graph_name = self.handler.graph_name
    self.handler.graph_name = self.graph_name
    self.handler._reset_execution_order()
    self.handler._latency_tracker.start(self._run_id)

    self.handler._log(
        "GRAPH_START",
        self._run_id,
        content=json.dumps({"graph_name": self.graph_name}, default=str),
        attributes=self.handler._build_langgraph_attributes(
            metadata=self.metadata
        ),
        metadata=self.metadata,
    )
    return self

  def __exit__(
      self,
      exc_type: Optional[type],
      exc_val: Optional[BaseException],
      exc_tb: Any,
  ) -> None:
    """Exit the context and emit GRAPH_END or GRAPH_ERROR event."""
    if self._run_id is None:
      return

    latency_measurement = self.handler._latency_tracker.end(self._run_id)

    if exc_val is not None:
      # An exception occurred - emit GRAPH_ERROR
      self.handler._log(
          "GRAPH_ERROR",
          self._run_id,
          error=str(exc_val),
          attributes=self.handler._build_langgraph_attributes(
              metadata=self.metadata
          ),
          metadata=self.metadata,
          latency_measurement=latency_measurement,
      )
    else:
      # Normal completion - emit GRAPH_END
      self.handler._log(
          "GRAPH_END",
          self._run_id,
          content=json.dumps({"graph_name": self.graph_name}, default=str),
          attributes=self.handler._build_langgraph_attributes(
              metadata=self.metadata
          ),
          metadata=self.metadata,
          latency_measurement=latency_measurement,
      )

    # Restore original graph name
    self.handler.graph_name = self._original_graph_name

  @property
  def run_id(self) -> Optional[uuid.UUID]:
    """Get the run ID for this graph execution."""
    return self._run_id


class AsyncGraphExecutionContext:
  """Async context manager for wrapping graph execution with events.

  Usage:
      async with handler.graph_context("my_graph") as ctx:
          # Graph execution happens here
          result = await graph.ainvoke(inputs)

  This will emit GRAPH_START when entering and GRAPH_END when exiting.
  If an exception occurs, GRAPH_ERROR will be emitted instead of GRAPH_END.
  """

  def __init__(
      self,
      handler: AsyncBigQueryCallbackHandler,
      graph_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Initialize the async graph execution context.

    Args:
        handler: The AsyncBigQueryCallbackHandler instance.
        graph_name: Name of the graph being executed.
        metadata: Optional metadata to include in events.
    """
    self.handler = handler
    self.graph_name = graph_name
    self.metadata = metadata or {}
    self._run_id: Optional[uuid.UUID] = None
    self._original_graph_name: Optional[str] = None

  async def __aenter__(self) -> "AsyncGraphExecutionContext":
    """Enter the context and emit GRAPH_START event."""
    self._run_id = uuid.uuid4()
    self._original_graph_name = self.handler.graph_name
    self.handler.graph_name = self.graph_name
    self.handler._reset_execution_order()
    await self.handler._latency_tracker.start(self._run_id)

    await self.handler._log(
        "GRAPH_START",
        self._run_id,
        content=json.dumps({"graph_name": self.graph_name}, default=str),
        attributes=self.handler._build_langgraph_attributes(
            metadata=self.metadata
        ),
        metadata=self.metadata,
    )
    return self

  async def __aexit__(
      self,
      exc_type: Optional[type],
      exc_val: Optional[BaseException],
      exc_tb: Any,
  ) -> None:
    """Exit the context and emit GRAPH_END or GRAPH_ERROR event."""
    if self._run_id is None:
      return

    latency_measurement = await self.handler._latency_tracker.end(self._run_id)

    if exc_val is not None:
      # An exception occurred - emit GRAPH_ERROR
      await self.handler._log(
          "GRAPH_ERROR",
          self._run_id,
          error=str(exc_val),
          attributes=self.handler._build_langgraph_attributes(
              metadata=self.metadata
          ),
          metadata=self.metadata,
          latency_measurement=latency_measurement,
      )
    else:
      # Normal completion - emit GRAPH_END
      await self.handler._log(
          "GRAPH_END",
          self._run_id,
          content=json.dumps({"graph_name": self.graph_name}, default=str),
          attributes=self.handler._build_langgraph_attributes(
              metadata=self.metadata
          ),
          metadata=self.metadata,
          latency_measurement=latency_measurement,
      )

    # Restore original graph name
    self.handler.graph_name = self._original_graph_name

  @property
  def run_id(self) -> Optional[uuid.UUID]:
    """Get the run ID for this graph execution."""
    return self._run_id
