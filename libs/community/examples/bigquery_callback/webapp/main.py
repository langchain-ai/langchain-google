#!/usr/bin/env python3
"""FastAPI web application for real-time LangGraph agent analytics.

This application provides a Datadog-like dashboard for monitoring
LangGraph agent events stored in BigQuery.

Features:
- Real-time event streaming via Server-Sent Events (SSE)
- Interactive dashboards with auto-refresh
- Tool usage analytics
- Latency monitoring
- Error tracking
- Session reconstruction

Prerequisites:
    1. gcloud auth application-default login
    2. pip install fastapi uvicorn google-cloud-bigquery jinja2 sse-starlette

Usage:
    uvicorn main:app --reload --port 8000
    # Then open http://localhost:8000
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import bigquery
from sse_starlette.sse import EventSourceResponse

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "agent_events_v2")
FULL_TABLE_ID = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Agent Analytics",
    description="Real-time monitoring dashboard for LangGraph agents",
    version="1.0.0",
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# BigQuery client (lazy initialization)
_bq_client: Optional[bigquery.Client] = None


def get_bq_client() -> bigquery.Client:
    """Get or create BigQuery client."""
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client


def run_query(sql: str) -> list[dict[str, Any]]:
    """Execute a BigQuery query and return results as list of dicts."""
    client = get_bq_client()
    query_job = client.query(sql)
    results = query_job.result()
    return [dict(row) for row in results]


# =============================================================================
# HTML Pages
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "project_id": PROJECT_ID,
            "dataset_id": DATASET_ID,
            "table_id": TABLE_ID,
        },
    )


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    """Session detail page."""
    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "session_id": session_id,
        },
    )


# =============================================================================
# API Endpoints - Summary Stats
# =============================================================================


@app.get("/api/summary")
async def get_summary():
    """Get overall summary statistics."""
    sql = f"""
    SELECT
        COUNT(*) as total_events,
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(DISTINCT user_id) as unique_users,
        COUNT(DISTINCT agent) as active_agents,
        COUNTIF(event_type = 'LLM_REQUEST') as llm_requests,
        COUNTIF(event_type = 'TOOL_STARTING') as tool_invocations,
        COUNTIF(status = 'ERROR') as total_errors,
        ROUND(COUNTIF(status = 'ERROR') * 100.0 / NULLIF(COUNT(*), 0), 2) as error_rate_pct,
        ROUND(AVG(CASE WHEN event_type = 'LLM_RESPONSE'
            THEN CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64) END), 0) as avg_llm_latency_ms
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
    """
    results = run_query(sql)
    return results[0] if results else {}


@app.get("/api/summary/hourly")
async def get_hourly_summary():
    """Get summary stats for the last hour."""
    sql = f"""
    SELECT
        COUNT(*) as total_events,
        COUNT(DISTINCT session_id) as active_sessions,
        COUNTIF(event_type = 'LLM_REQUEST') as llm_requests,
        COUNTIF(event_type = 'TOOL_STARTING') as tool_invocations,
        COUNTIF(status = 'ERROR') as errors
    FROM `{FULL_TABLE_ID}`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    """
    results = run_query(sql)
    return results[0] if results else {}


# =============================================================================
# API Endpoints - Event Stream
# =============================================================================


@app.get("/api/events/recent")
async def get_recent_events(limit: int = Query(default=50, le=200)):
    """Get most recent events."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
        agent,
        event_type,
        session_id,
        user_id,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64) as latency_ms,
        status
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    return run_query(sql)


@app.get("/api/events/stream")
async def stream_events(request: Request):
    """Server-Sent Events stream for real-time updates."""

    async def event_generator() -> AsyncGenerator[dict, None]:
        last_timestamp = None
        while True:
            if await request.is_disconnected():
                break

            # Query for new events since last check
            if last_timestamp:
                sql = f"""
                SELECT
                    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S.%f', timestamp) as timestamp,
                    agent,
                    event_type,
                    session_id,
                    user_id,
                    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
                    CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64) as latency_ms,
                    status
                FROM `{FULL_TABLE_ID}`
                WHERE timestamp > TIMESTAMP('{last_timestamp}')
                ORDER BY timestamp ASC
                LIMIT 100
                """
            else:
                sql = f"""
                SELECT
                    FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S.%f', timestamp) as timestamp,
                    agent,
                    event_type,
                    session_id,
                    user_id,
                    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
                    CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64) as latency_ms,
                    status
                FROM `{FULL_TABLE_ID}`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
                ORDER BY timestamp DESC
                LIMIT 20
                """

            try:
                events = run_query(sql)
                if events:
                    last_timestamp = events[-1]["timestamp"] if last_timestamp else events[0]["timestamp"]
                    yield {"event": "events", "data": json.dumps(events)}
            except Exception as e:
                yield {"event": "error", "data": json.dumps({"error": str(e)})}

            await asyncio.sleep(2)  # Poll every 2 seconds

    return EventSourceResponse(event_generator())


# =============================================================================
# API Endpoints - Event Distribution
# =============================================================================


@app.get("/api/events/distribution")
async def get_event_distribution():
    """Get event type distribution."""
    sql = f"""
    SELECT
        event_type,
        COUNT(*) as count,
        COUNT(DISTINCT session_id) as unique_sessions
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
    GROUP BY event_type
    ORDER BY count DESC
    """
    return run_query(sql)


@app.get("/api/events/by-agent")
async def get_events_by_agent():
    """Get event counts by agent."""
    sql = f"""
    SELECT
        agent,
        COUNT(*) as total_events,
        COUNT(DISTINCT session_id) as sessions,
        COUNT(DISTINCT user_id) as unique_users,
        COUNTIF(event_type = 'LLM_REQUEST') as llm_calls,
        COUNTIF(event_type = 'TOOL_STARTING') as tool_calls,
        COUNTIF(status = 'ERROR') as errors
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND agent IS NOT NULL
    GROUP BY agent
    ORDER BY total_events DESC
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - Tool Analytics
# =============================================================================


@app.get("/api/tools/usage")
async def get_tool_usage():
    """Get tool usage statistics."""
    sql = f"""
    SELECT
        agent,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        COUNT(*) as call_count,
        ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 2) as avg_latency_ms,
        MAX(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64)) as max_latency_ms,
        COUNTIF(status = 'ERROR') as error_count
    FROM `{FULL_TABLE_ID}`
    WHERE event_type = 'TOOL_COMPLETED'
      AND DATE(timestamp) = CURRENT_DATE()
      AND JSON_EXTRACT_SCALAR(attributes, '$.tool_name') IS NOT NULL
    GROUP BY agent, tool_name
    ORDER BY call_count DESC
    """
    return run_query(sql)


@app.get("/api/tools/heatmap")
async def get_tool_heatmap():
    """Get tool usage heatmap data (agent x tool matrix)."""
    sql = f"""
    SELECT
        COALESCE(agent, 'Unknown') as agent,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        COUNT(*) as call_count
    FROM `{FULL_TABLE_ID}`
    WHERE event_type = 'TOOL_COMPLETED'
      AND DATE(timestamp) = CURRENT_DATE()
      AND JSON_EXTRACT_SCALAR(attributes, '$.tool_name') IS NOT NULL
    GROUP BY agent, tool_name
    ORDER BY agent, tool_name
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - Latency Analytics
# =============================================================================


@app.get("/api/latency/by-event-type")
async def get_latency_by_event_type():
    """Get latency statistics by event type."""
    sql = f"""
    SELECT
        event_type,
        agent,
        COUNT(*) as count,
        ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 2) as avg_latency_ms,
        ROUND(APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64), 100)[OFFSET(50)], 2) as p50_latency_ms,
        ROUND(APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64), 100)[OFFSET(95)], 2) as p95_latency_ms,
        MAX(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64)) as max_latency_ms
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND event_type IN ('LLM_RESPONSE', 'TOOL_COMPLETED', 'GRAPH_END')
      AND JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') IS NOT NULL
    GROUP BY event_type, agent
    ORDER BY avg_latency_ms DESC
    """
    return run_query(sql)


@app.get("/api/latency/trend")
async def get_latency_trend():
    """Get latency trend over time."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:00', TIMESTAMP_TRUNC(timestamp, MINUTE)) as minute,
        ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 2) as avg_latency_ms,
        ROUND(APPROX_QUANTILES(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64), 100)[OFFSET(95)], 2) as p95_latency_ms,
        COUNT(*) as request_count
    FROM `{FULL_TABLE_ID}`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
      AND event_type = 'LLM_RESPONSE'
      AND JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') IS NOT NULL
    GROUP BY minute
    ORDER BY minute
    """
    return run_query(sql)


@app.get("/api/latency/hourly")
async def get_hourly_latency():
    """Get hourly latency pattern."""
    sql = f"""
    SELECT
        EXTRACT(HOUR FROM timestamp) as hour,
        agent,
        ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 2) as avg_latency_ms,
        COUNT(*) as request_count
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND event_type = 'LLM_RESPONSE'
      AND JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') IS NOT NULL
      AND agent IS NOT NULL
    GROUP BY hour, agent
    ORDER BY hour, agent
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - Error Analytics
# =============================================================================


@app.get("/api/errors/summary")
async def get_error_summary():
    """Get error summary."""
    sql = f"""
    SELECT
        agent,
        event_type,
        COUNT(*) as error_count,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', MIN(timestamp)) as first_occurrence,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', MAX(timestamp)) as last_occurrence
    FROM `{FULL_TABLE_ID}`
    WHERE status = 'ERROR'
      AND DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    GROUP BY agent, event_type
    ORDER BY error_count DESC
    """
    return run_query(sql)


@app.get("/api/errors/recent")
async def get_recent_errors(limit: int = Query(default=20, le=100)):
    """Get recent errors."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
        agent,
        session_id,
        event_type,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        SUBSTR(TO_JSON_STRING(content), 1, 200) as content_preview
    FROM `{FULL_TABLE_ID}`
    WHERE status = 'ERROR'
      AND DATE(timestamp) = CURRENT_DATE()
    ORDER BY timestamp DESC
    LIMIT {limit}
    """
    return run_query(sql)


@app.get("/api/errors/rate")
async def get_error_rate():
    """Get error rate over time."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:00', TIMESTAMP_TRUNC(timestamp, MINUTE)) as minute,
        COUNT(*) as total_events,
        COUNTIF(status = 'ERROR') as errors,
        ROUND(COUNTIF(status = 'ERROR') * 100.0 / NULLIF(COUNT(*), 0), 2) as error_rate_pct
    FROM `{FULL_TABLE_ID}`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    GROUP BY minute
    ORDER BY minute
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - Session Analytics
# =============================================================================


@app.get("/api/sessions/recent")
async def get_recent_sessions(limit: int = Query(default=20, le=100)):
    """Get recent sessions."""
    sql = f"""
    SELECT
        session_id,
        agent,
        user_id,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', MIN(timestamp)) as start_time,
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', MAX(timestamp)) as end_time,
        COUNTIF(event_type = 'LLM_REQUEST') as llm_calls,
        COUNTIF(event_type = 'TOOL_STARTING') as tool_calls,
        MAX(CASE WHEN event_type = 'GRAPH_END'
            THEN CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64) END) as total_latency_ms,
        COUNTIF(status = 'ERROR') as errors,
        CASE WHEN COUNTIF(status = 'ERROR') > 0 THEN 'Failed' ELSE 'Success' END as status
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
    GROUP BY session_id, agent, user_id
    ORDER BY start_time DESC
    LIMIT {limit}
    """
    return run_query(sql)


@app.get("/api/sessions/{session_id}/timeline")
async def get_session_timeline(session_id: str):
    """Get detailed timeline for a session."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S.%f', timestamp) as timestamp,
        event_type,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS INT64) as latency_ms,
        status,
        SUBSTR(TO_JSON_STRING(content), 1, 500) as content_preview,
        trace_id,
        span_id
    FROM `{FULL_TABLE_ID}`
    WHERE session_id = '{session_id}'
    ORDER BY timestamp
    """
    return run_query(sql)


@app.get("/api/sessions/{session_id}/conversation")
async def get_session_conversation(session_id: str):
    """Reconstruct conversation for a session."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
        event_type,
        JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
        JSON_EXTRACT_SCALAR(content, '$.summary') as message,
        TO_JSON_STRING(content) as full_content
    FROM `{FULL_TABLE_ID}`
    WHERE session_id = '{session_id}'
      AND event_type IN ('LLM_REQUEST', 'LLM_RESPONSE', 'TOOL_STARTING', 'TOOL_COMPLETED')
    ORDER BY timestamp
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - User Analytics
# =============================================================================


@app.get("/api/users/engagement")
async def get_user_engagement():
    """Get user engagement metrics."""
    sql = f"""
    SELECT
        user_id,
        COUNT(DISTINCT session_id) as total_sessions,
        COUNT(DISTINCT agent) as agents_used,
        COUNTIF(event_type = 'LLM_REQUEST') as total_queries,
        COUNTIF(event_type = 'TOOL_STARTING') as tool_interactions,
        ROUND(AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)), 0) as avg_response_time_ms,
        COUNTIF(status = 'ERROR') as errors_encountered
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND user_id IS NOT NULL
    GROUP BY user_id
    ORDER BY total_sessions DESC
    """
    return run_query(sql)


@app.get("/api/users/agent-preference")
async def get_user_agent_preference():
    """Get user-agent preference matrix."""
    sql = f"""
    SELECT
        user_id,
        agent,
        COUNT(DISTINCT session_id) as sessions,
        COUNT(*) as total_events
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND user_id IS NOT NULL
      AND agent IS NOT NULL
    GROUP BY user_id, agent
    ORDER BY user_id, sessions DESC
    """
    return run_query(sql)


# =============================================================================
# API Endpoints - Time Series
# =============================================================================


@app.get("/api/timeseries/activity")
async def get_activity_timeseries():
    """Get activity time series (events per minute)."""
    sql = f"""
    SELECT
        FORMAT_TIMESTAMP('%Y-%m-%d %H:%M:00', TIMESTAMP_TRUNC(timestamp, MINUTE)) as minute,
        agent,
        COUNT(*) as event_count
    FROM `{FULL_TABLE_ID}`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
      AND agent IS NOT NULL
    GROUP BY minute, agent
    ORDER BY minute, agent
    """
    return run_query(sql)


@app.get("/api/timeseries/hourly-pattern")
async def get_hourly_pattern():
    """Get hourly activity pattern."""
    sql = f"""
    SELECT
        EXTRACT(HOUR FROM timestamp) as hour,
        agent,
        COUNT(*) as event_count
    FROM `{FULL_TABLE_ID}`
    WHERE DATE(timestamp) = CURRENT_DATE()
      AND agent IS NOT NULL
    GROUP BY hour, agent
    ORDER BY hour, agent
    """
    return run_query(sql)


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Quick query to verify BigQuery connection
        sql = f"SELECT 1 as ok FROM `{FULL_TABLE_ID}` LIMIT 1"
        run_query(sql)
        return {"status": "healthy", "bigquery": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
