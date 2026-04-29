# BigQuery Callback Handler Examples

This directory demonstrates the `BigQueryCallbackHandler` for shipping
LangChain / LangGraph agent telemetry to BigQuery. The handler aims for
parity with [Google ADK's `BigQueryAgentAnalyticsPlugin`][adk-plugin] so the
same dashboards and SQL work across both stacks.

[adk-plugin]: https://github.com/google/adk-python/blob/main/src/google/adk/plugins/bigquery_agent_analytics_plugin.py

## Prerequisites

1. **Google Cloud setup**:

   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Create a BigQuery dataset** (the handler creates the table and views):

   ```bash
   bq mk --dataset YOUR_PROJECT_ID:agent_analytics
   ```

3. **Install dependencies**:

   ```bash
   pip install langchain-google-community langgraph langchain-google-genai
   ```

## Examples

### 1. Basic Usage (`basic_example.py`)

Single LLM call with `BigQueryCallbackHandler` attached. Good for a first
end-to-end smoke test.

```bash
python basic_example.py
```

### 2. LangGraph Agent (`langgraph_agent_example.py`)

A complete LangGraph ReAct agent with realistic tools. Demonstrates:

- Tool calls with proper tracking (`TOOL_STARTING` / `TOOL_COMPLETED`)
- LangGraph node tracking (`AGENT_STARTING` / `AGENT_COMPLETED`)
- Top-level invocation tracking (`INVOCATION_STARTING` / `INVOCATION_COMPLETED`)
  via the `graph_context()` context manager
- Per-sub-agent attribution (LangGraph `node_name` is auto-promoted to the
  `agent` BigQuery column when no explicit `agent` is set in metadata)
- Token usage with `prompt_tokens` / `completion_tokens` / `total_tokens`
  surfaced from modern Vertex Gemini chat models

```bash
python langgraph_agent_example.py
```

### 3. Event Filtering (`event_filtering_example.py`)

Three ways to control which events land in BigQuery:

- `event_allowlist` — explicit allow list (e.g. just `LLM_RESPONSE` + errors)
- `event_denylist` — explicit deny list
- `skip_internal_chain_events=True` — drop noisy framework chains
  (`ChannelWrite`, `RunnableLambda`, `Branch`, …) without losing trace
  continuity (child events still resolve to the real graph root)

```bash
python event_filtering_example.py
```

### 4. Async Example (`async_example.py`)

`AsyncBigQueryCallbackHandler` with an async LangGraph agent. Useful as a
template for any async agent runtime (Vertex AI Agent Engine, FastAPI, …).

```bash
python async_example.py
```

### 5. Analytics Notebook (`langgraph_agent_analytics.ipynb`)

Jupyter notebook covering real-time event monitoring, tool usage analytics,
latency analysis, error debugging, user engagement, and time-series
visualization. Open in Jupyter or upload to Google Colab.

### 6. Populate Sample Data (`populate_sample_data.py`)

Generates diverse sample data across multiple agents (`finance_assistant`,
`travel_planner`, `customer_support`) so the notebook and webapp dashboards
have something to show.

```bash
python populate_sample_data.py
```

### 7. Real-time Analytics Dashboard (`webapp/`)

FastAPI dashboard with a live event stream, auto-refreshing charts, session
tracing, and error tracking. See `webapp/README.md` for details.

```bash
cd webapp && pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# http://localhost:8000
```

## Event Types

Events are written to a single events table; each row's `event_type`
identifies the lifecycle stage it represents. The vocabulary is aligned
with ADK's `BigQueryAgentAnalyticsPlugin` so the same dashboards work
across both stacks.

| Event type | Emitted when |
|---|---|
| `INVOCATION_STARTING` / `INVOCATION_COMPLETED` / `INVOCATION_ERROR` | Top-level graph invocation begins / ends / errors. Use `handler.graph_context("name")` to wrap graph execution. |
| `AGENT_STARTING` / `AGENT_COMPLETED` / `AGENT_ERROR` | A LangGraph node (sub-agent) begins / ends / errors. The `agent` column is set from `metadata['langgraph_node']` when no explicit `agent` is supplied. |
| `LLM_REQUEST` / `LLM_RESPONSE` / `LLM_ERROR` | Model call begins / ends / errors. `attributes.usage`, `attributes.usage_metadata` (incl. `cached_content_token_count`), `attributes.model_version`, `attributes.llm_config`, and `attributes.tools` are populated where available. |
| `TOOL_STARTING` / `TOOL_COMPLETED` / `TOOL_ERROR` | Tool call begins / ends / errors. |
| `RETRIEVER_START` / `RETRIEVER_END` / `RETRIEVER_ERROR` | Retriever lifecycle (LangChain-specific). |
| `CHAIN_START` / `CHAIN_END` / `CHAIN_ERROR` | Non-graph LangChain `Runnable` lifecycle (LangChain-specific). |
| `AGENT_ACTION` / `AGENT_FINISH` | Legacy LangChain agent signals. |
| `TEXT` | `on_text` callback. |

> Tip — `skip_internal_chain_events=True` drops `CHAIN_*` events emitted by
> LangGraph framework internals (`ChannelWrite`, `RunnableLambda`, …) while
> preserving trace continuity for child LLM/tool events.

## Auto-created analytics views

When the handler creates the events table, it also creates one
`CREATE OR REPLACE VIEW` per event type beside it. Each view unnests the
JSON columns into typed top-level columns so analytics queries don't have
to spell `JSON_VALUE(...)` every time.

For example, querying token usage becomes:

```sql
SELECT
  agent,
  SUM(usage_total_tokens) AS total_tokens,
  SUM(usage_prompt_tokens) AS prompt_tokens,
  SUM(usage_completion_tokens) AS completion_tokens,
  SAFE_DIVIDE(
    SUM(usage_cached_tokens),
    SUM(usage_prompt_tokens)
  ) AS context_cache_hit_rate
FROM `PROJECT.DATASET.v_llm_response`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY agent
ORDER BY total_tokens DESC;
```

The default view names are `v_invocation_starting`, `v_invocation_completed`,
`v_agent_starting`, `v_agent_completed`, `v_llm_request`, `v_llm_response`,
`v_tool_starting`, `v_tool_completed`, … Set `view_prefix="vstaging"` (etc.)
when multiple handler instances share one dataset to avoid name collisions.

To opt out of view creation, pass `create_views=False`.

## Auto schema upgrade

Existing tables are auto-upgraded additively when the handler's schema
gains new columns. The handler reads the table at startup and runs
`ALTER TABLE ADD COLUMN` for any new fields, gated by a
`langchain_bq_schema_version` table label so the diff runs at most once
per schema version. **Never** drops, renames, or retypes columns. Disable
with `auto_schema_upgrade=False`.

## Configuration

`BigQueryLoggerConfig` accepts:

| Option | Description | Default |
|---|---|---|
| `enabled` | Master enable / disable. | `True` |
| `event_allowlist` | If set, only these `event_type`s are logged. | `None` |
| `event_denylist` | If set, these `event_type`s are skipped. | `None` |
| `skip_internal_chain_events` | Drop `CHAIN_*` events emitted by framework-internal chains (preserves trace continuity for child events). | `False` |
| `max_content_length` | Per-text-block truncation threshold. | `512000` |
| `table_id` | Events table name. | `"agent_events_v2"` |
| `clustering_fields` | BigQuery clustering fields. | `["event_type", "agent", "user_id"]` |
| `log_multi_modal_content` | Include the per-part `content_parts` array. | `True` |
| `retry_config` | Retry policy for the BigQuery Storage Write API. | `RetryConfig()` |
| `batch_size` | Rows per write batch. | `1` |
| `batch_flush_interval` | Max seconds to wait before flushing a batch. | `1.0` |
| `shutdown_timeout` | Max seconds to wait for queue drain on `shutdown()`. | `10.0` |
| `queue_max_size` | In-memory queue capacity. | `10000` |
| `gcs_bucket_name` | GCS bucket for offloading large content (images, audio, …). | `None` |
| `connection_id` | BigQuery `ObjectRef` authorizer. | `None` |
| `custom_tags` | Static dict written under `attributes.custom_tags` on every event. | `{}` |
| `log_session_metadata` | Dump user-supplied `RunnableConfig` metadata under `attributes.session_metadata`. | `True` |
| `content_formatter` | Optional `(raw, event_type) -> formatted` hook for redaction / coercion. | `None` |
| `auto_schema_upgrade` | Additive `ALTER TABLE ADD COLUMN` on existing tables. | `True` |
| `create_views` | Auto-create per-event-type analytics views. | `True` |
| `view_prefix` | Prefix for view names (`v_llm_response` etc.). | `"v"` |

## Operational helpers

- **`handler.graph_context("name")`** — emits `INVOCATION_STARTING` /
  `INVOCATION_COMPLETED` (or `INVOCATION_ERROR` on exception) around a
  graph invocation. Recommended at request boundaries.
- **`handler.flush(timeout=5.0)`** — blocks until queued rows have been
  written to BigQuery (or the timeout elapses). Useful between requests
  when callers want each invocation's events durable before returning.
  Does **not** shut the handler down; subsequent events keep working.
- **`handler.shutdown()`** — drains the queue and closes resources.
  Implicit when used as a context manager (`with handler:` /
  `async with handler:`).

## Sub-agent attribution

For multi-agent LangGraph deployments, the handler auto-derives the
`agent` BigQuery column from this fallback chain:

1. `metadata["agent"]` — explicit user-supplied value (highest priority)
2. `metadata["langgraph_node"]` — the active LangGraph node, so each
   sub-agent's events are tagged with the node name
3. `metadata["checkpoint_ns"]` — LangGraph checkpoint namespace
4. `handler.graph_name` — fallback for top-level `INVOCATION_*` events

This means a multi-agent graph (e.g. supervisor → `TheCritic`, `TheMeteo`,
…) produces telemetry where each event is attributed to the originating
sub-agent without any user changes.

## Quick start: viewing results

The auto-created views give you typed columns:

```sql
-- Recent events with first-class columns
SELECT timestamp, event_type, agent, session_id, user_id,
       root_agent_name, total_ms
FROM `PROJECT.DATASET.v_llm_response`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 100;
```

If you prefer the raw events table:

```sql
SELECT timestamp, event_type, session_id,
       JSON_VALUE(attributes, '$.tool_name') AS tool_name,
       CAST(JSON_VALUE(latency_ms, '$.total_ms') AS INT64) AS latency_ms,
       status
FROM `PROJECT.DATASET.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 100;
```

## Analytics queries

### Event distribution

```sql
SELECT event_type, COUNT(*) AS n
FROM `PROJECT.DATASET.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY 1
ORDER BY 2 DESC;
```

### Tool usage by agent

```sql
SELECT agent, tool_name, COUNT(*) AS calls,
       APPROX_QUANTILES(total_ms, 100)[OFFSET(50)] AS p50_ms,
       APPROX_QUANTILES(total_ms, 100)[OFFSET(95)] AS p95_ms
FROM `PROJECT.DATASET.v_tool_completed`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY 1, 2
ORDER BY calls DESC;
```

### LLM cost / latency per sub-agent

```sql
SELECT agent,
       model_version,
       COUNT(*) AS calls,
       SUM(usage_total_tokens) AS total_tokens,
       AVG(total_ms) AS avg_ms,
       AVG(ttft_ms) AS avg_ttft_ms
FROM `PROJECT.DATASET.v_llm_response`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY 1, 2
ORDER BY total_tokens DESC;
```

### Trace a single invocation

All events emitted within one user turn share a `trace_id`:

```sql
SELECT timestamp, event_type, agent, span_id, parent_span_id,
       JSON_VALUE(content, '$.summary') AS summary
FROM `PROJECT.DATASET.agent_events_v2`
WHERE trace_id = '<trace_id from any row>'
ORDER BY timestamp;
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Dataset 'X' does not exist` | Create the dataset with `bq mk --dataset PROJECT:DATASET`. The handler creates the table but not the dataset. |
| `403 Permission denied` | The runtime account needs `roles/bigquery.dataEditor` on the dataset and `roles/bigquery.jobUser` on the project. |
| `agent` column is null on some rows | At the very top level (`INVOCATION_*`) `agent` falls back to `handler.graph_name` — set that on the handler if you want a top-level label. |
| Way too many `CHAIN_START` / `CHAIN_END` rows | Set `skip_internal_chain_events=True` in `BigQueryLoggerConfig`. |
| Auto-views fail to create | Check that the runtime account has `roles/bigquery.dataEditor` on the dataset; view creation logs a warning per view but does not fail the handler. |
