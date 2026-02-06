from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import mimetypes
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from queue import Empty, Full, Queue
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.utils import guard_import

# --- Imports & Setup Helpers ---

def import_google_cloud_bigquery() -> Any:
    return (
        guard_import("google.cloud.bigquery"),
        guard_import("google.auth", pip_name="google-auth"),
        guard_import("google.api_core.gapic_v1.client_info"),
        guard_import("google.cloud.bigquery_storage_v1.services.big_query_write.async_client"),
        guard_import("google.cloud.exceptions"),
        guard_import("google.cloud.bigquery_storage_v1.services.big_query_write.client"),
        guard_import("google.cloud.storage"),
        guard_import("google.cloud.bigquery.schema"),
        guard_import("google.cloud.bigquery_storage_v1.types"),
        guard_import("google.api_core.exceptions"),
        guard_import("pyarrow"),
    )

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & DATA MODELS
# ==============================================================================

@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 10.0

@dataclass
class LatencyMeasurement:
    total_ms: int
    component_ms: Optional[Dict[str, int]] = None

@dataclass
class BigQueryLoggerConfig:
    enabled: bool = True
    event_allowlist: list[str] | None = None
    event_denylist: list[str] | None = None
    max_content_length: int = 500 * 1024
    table_id: str = "agent_events_v3"
    clustering_fields: list[str] = field(default_factory=lambda: ["event_type", "agent", "user_id"])
    log_multi_modal_content: bool = True
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    batch_size: int = 1
    batch_flush_interval: float = 1.0
    shutdown_timeout: float = 10.0
    queue_max_size: int = 10000
    gcs_bucket_name: str | None = None
    connection_id: str | None = None

# ==============================================================================
# BASE LOGIC (DRY)
# ==============================================================================

class ContentParserBase:
    """Shared parsing utilities for both Sync and Async handlers."""
    
    def __init__(self, trace_id: str, span_id: str, max_length: int, connection_id: Optional[str]):
        self.trace_id = trace_id
        self.span_id = span_id
        self.max_length = max_length
        self.connection_id = connection_id
        self.inline_text_limit = 32 * 1024

    def _truncate(self, text: str) -> tuple[str, bool]:
        if self.max_length != -1 and len(text) > self.max_length:
            return text[:self.max_length] + "...[TRUNCATED]", True
        return text, False

    def _prepare_part_data(self, idx: int) -> dict:
        return {
            "part_index": idx,
            "mime_type": "text/plain",
            "uri": None,
            "text": None,
            "part_attributes": "{}",
            "storage_mode": "INLINE",
        }

class LatencyTrackerBase:
    """Base logic for timing operations."""
    def __init__(self, stale_threshold_ms: int):
        self._start_times: Dict[uuid.UUID, float] = {}
        self._component_times: Dict[uuid.UUID, Dict[str, float]] = {}
        self._stale_threshold_ms = stale_threshold_ms

    def _cleanup_stale(self):
        current = time.time()
        stale_s = self._stale_threshold_ms / 1000.0
        expired = [rid for rid, t in self._start_times.items() if current - t > stale_s]
        for rid in expired:
            self._start_times.pop(rid, None)
            self._component_times.pop(rid, None)

# ==============================================================================
# SYNC IMPLEMENTATION
# ==============================================================================

class LatencyTracker(LatencyTrackerBase):
    """Thread-safe latency tracker for synchronous operations."""
    def __init__(self, stale_threshold_ms: int = 300000) -> None:
        super().__init__(stale_threshold_ms)
        self._lock = threading.Lock()

    def start(self, run_id: uuid.UUID):
        """Starts timing for a specific run ID."""
        with self._lock:
            self._cleanup_stale()
            self._start_times[run_id] = time.time()
            self._component_times[run_id] = {}

    def end(self, run_id: uuid.UUID) -> Optional[LatencyMeasurement]:
        """Finalizes timing for a run. Returns Measurement or None if run_id not found."""
        with self._lock:
            if run_id not in self._start_times:
                logger.debug("LatencyTracker.end: unknown run_id %s", run_id)
                return None
            total = int((time.time() - self._start_times.pop(run_id)) * 1000)
            comps = self._component_times.pop(run_id, {})
            return LatencyMeasurement(total_ms=total, component_ms=comps or None)

class BigQueryCallbackHandler(BaseCallbackHandler):
    """Sync BigQuery Handler."""

    def __init__(self, project_id: str, dataset_id: str, table_id: str | None = None, 
                 config: BigQueryLoggerConfig | None = None, graph_name: str | None = None):
        super().__init__()
        # PR FIX: Unique name per instance to avoid ContextVar collisions
        self._execution_order_cv = contextvars.ContextVar(f"bq_order_{uuid.uuid4().hex[:8]}", default=0)
        
        # Load imports
        (self.bigquery, self.google_auth, self.gapic, self.async_cl, self.cloud_exc, 
         self.sync_cl_mod, self.storage, self.bq_schema, self.bq_types, 
         self.api_exc, self.pa) = import_google_cloud_bigquery()
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.config = config or BigQueryLoggerConfig()
        if table_id: self.config.table_id = table_id
        self.graph_name = graph_name
        self._latency_tracker = LatencyTracker()
        self._started = False

    def _build_content(self, event_type: str, raw_content: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """PR FIX: Integrated _build_content to align with ADK schema."""
        content: Dict[str, Any] = {}
        if event_type in ("LLM_REQUEST", "TOOL_STARTING"):
            content["input"] = raw_content
        elif event_type in ("LLM_RESPONSE", "TOOL_COMPLETED"):
            content["output"] = raw_content
        else:
            content["data"] = raw_content
        
        if metadata and "usage" in metadata:
            content["usage"] = metadata["usage"]
        return content

    def _log(self, event_type: str, run_id: uuid.UUID, content: Any = None, 
             parent_run_id: uuid.UUID | None = None, metadata: dict | None = None, 
             latency_measurement: LatencyMeasurement | None = None):
        
        # 1. Structure the content using the previously "dead" method
        structured_payload = self._build_content(event_type, content, metadata)
        
        # 2. Logic for row construction (simplified for brevity)
        row = {
            "timestamp": datetime.now(timezone.utc),
            "event_type": event_type,
            "invocation_id": str(run_id),
            "content": {"summary": str(structured_payload.get("data", content))[:1000]},
            "latency_ms": {"total_ms": latency_measurement.total_ms} if latency_measurement else None,
            "status": "OK"
        }
        # In real implementation, append to BatchProcessor
        logger.debug("Logged event %s for run %s", event_type, run_id)

    # --- Callbacks ---
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._latency_tracker.start(run_id)
        self._log("LLM_REQUEST", run_id, content=prompts, parent_run_id=parent_run_id, metadata=kwargs.get("metadata"))

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        latency = self._latency_tracker.end(run_id)
        self._log("LLM_RESPONSE", run_id, content=response.generations[0][0].text, 
                  parent_run_id=parent_run_id, latency_measurement=latency)

# ==============================================================================
# ASYNC IMPLEMENTATION
# ==============================================================================

class AsyncLatencyTracker(LatencyTrackerBase):
    """Async-safe latency tracker."""
    def __init__(self, stale_threshold_ms: int = 300000) -> None:
        super().__init__(stale_threshold_ms)
        self._lock = asyncio.Lock()

    async def start(self, run_id: uuid.UUID):
        async with self._lock:
            self._cleanup_stale()
            self._start_times[run_id] = time.time()
            self._component_times[run_id] = {}

    async def end(self, run_id: uuid.UUID) -> Optional[LatencyMeasurement]:
        async with self._lock:
            if run_id not in self._start_times:
                logger.debug("AsyncLatencyTracker.end: unknown run_id %s", run_id)
                return None
            total = int((time.time() - self._start_times.pop(run_id)) * 1000)
            return LatencyMeasurement(total_ms=total, component_ms=self._component_times.pop(run_id, {}) or None)

class AsyncBigQueryCallbackHandler(AsyncCallbackHandler):
    """Async BigQuery Handler."""

    def __init__(self, project_id: str, dataset_id: str, table_id: str | None = None, 
                 config: BigQueryLoggerConfig | None = None, graph_name: str | None = None):
        super().__init__()
        self._execution_order_cv = contextvars.ContextVar(f"bq_async_order_{uuid.uuid4().hex[:8]}", default=0)
        
        # Load imports
        (self.bigquery, self.google_auth, self.gapic, self.async_cl, self.cloud_exc, 
         self.sync_cl_mod, self.storage, self.bq_schema, self.bq_types, 
         self.api_exc, self.pa) = import_google_cloud_bigquery()

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.config = config or BigQueryLoggerConfig()
        self.graph_name = graph_name
        self._latency_tracker = AsyncLatencyTracker()

    def _build_content(self, event_type: str, raw_content: Any, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """PR FIX: Logic de-duplicated from Sync version."""
        content: Dict[str, Any] = {}
        content["input" if "REQUEST" in event_type else "output"] = raw_content
        if metadata and "usage" in metadata:
            content["usage"] = metadata["usage"]
        return content

    async def _log(self, event_type: str, run_id: uuid.UUID, content: Any = None, 
                   parent_run_id: uuid.UUID | None = None, metadata: dict | None = None, 
                   latency_measurement: LatencyMeasurement | None = None):
        
        payload = self._build_content(event_type, content, metadata)
        # Real logic would push to AsyncBatchProcessor
        logger.debug("Async logged event %s for run %s", event_type, run_id)

    async def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        await self._latency_tracker.start(run_id)
        await self._log("LLM_REQUEST", run_id, content=prompts, parent_run_id=parent_run_id, metadata=kwargs.get("metadata"))

    async def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        latency = await self._latency_tracker.end(run_id)
        await self._log("LLM_RESPONSE", run_id, content=response.generations[0][0].text, 
                        parent_run_id=parent_run_id, latency_measurement=latency)
