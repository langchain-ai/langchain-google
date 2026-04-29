# BigQuery Callback Handler Examples

This directory contains examples demonstrating the enhanced BigQueryCallbackHandler
with LangGraph integration, latency tracking, and event filtering.

## Prerequisites

1. **Google Cloud Setup**:
   ```bash
   # Authenticate with your GCP account
   gcloud auth application-default login

   # Set your project
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Create a BigQuery Dataset**:
   ```bash
   # Create dataset for agent events (the handler will create the table automatically)
   bq mk --dataset YOUR_PROJECT_ID:agent_analytics
   ```

3. **Install Dependencies**:
   ```bash
   pip install langchain-google-community langgraph langchain-google-genai
   ```

## Examples

### 1. Basic Usage (`basic_example.py`)

Demonstrates basic callback handler usage with LLM calls:
- Latency tracking
- Event logging to BigQuery
- Error handling

```bash
python basic_example.py
```

### 2. LangGraph Agent (`langgraph_agent_example.py`)

A complete LangGraph ReAct agent with realistic tools:
- **Stock price lookup** - Get current stock prices
- **Weather** - Get weather conditions for cities
- **Currency conversion** - Convert between currencies
- **Calculator** - Evaluate math expressions with functions
- **Date/time** - Get current date and time
- **Random number** - Generate random numbers

Features demonstrated:
- Tool calls with proper tracking
- Node tracking (`NODE_STARTING`, `NODE_COMPLETED`)
- Graph context manager (`GRAPH_START`, `GRAPH_END`)
- Execution order tracking
- Full latency measurements

```bash
python langgraph_agent_example.py
```

### 3. Event Filtering (`event_filtering_example.py`)

Shows how to filter events using allowlist/denylist:
- Only log specific event types
- Exclude noisy events
- Production-ready configurations

```bash
python event_filtering_example.py
```

### 4. Async Example (`async_example.py`)

Demonstrates async callback handler with:
- Async LangGraph agent execution
- Concurrent query processing
- Async context manager

```bash
python async_example.py
```

## Viewing Results in BigQuery

After running the examples, query your events:

```sql
SELECT
  timestamp,
  event_type,
  session_id,
  JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
  JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') as latency_ms,
  status
FROM `YOUR_PROJECT_ID.agent_analytics.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
ORDER BY timestamp DESC
LIMIT 100;
```

## Configuration Options

The `BigQueryLoggerConfig` supports:

| Option | Description | Default |
|--------|-------------|---------|
| `enabled` | Enable/disable logging | `True` |
| `event_allowlist` | Only log these event types | `None` (all) |
| `event_denylist` | Don't log these event types | `None` (none) |
| `max_content_length` | Max content length before truncation | `512000` |
| `batch_size` | Batch size for writes | `1` |
| `batch_flush_interval` | Flush interval in seconds | `1.0` |

### 5. Analytics Notebook (`langgraph_agent_analytics.ipynb`)

A comprehensive Jupyter notebook for analyzing LangGraph agent data in BigQuery:
- Real-time event monitoring
- Tool usage analytics
- Latency analysis
- Error debugging
- User engagement metrics
- Time-series visualization

```bash
# Open in Jupyter
jupyter notebook langgraph_agent_analytics.ipynb

# Or upload to Google Colab
```

### 6. Populate Sample Data (`populate_sample_data.py`)

Generate diverse sample data across multiple agents for analytics demo:
- `finance_assistant` - Stock prices, currency conversion, calculations
- `travel_planner` - Weather, flight/hotel booking
- `customer_support` - Order status, refunds

```bash
python populate_sample_data.py
```

### 7. Real-time Analytics Dashboard (`webapp/`)

A **Datadog-like** real-time monitoring dashboard built with FastAPI:

**Features:**
- **Live Event Stream**: Watch agent events as they happen (Server-Sent Events)
- **Auto-refresh**: All metrics update every 5 seconds
- **Interactive Charts**: Event distribution, agent activity, latency trends
- **Session Tracing**: Detailed timeline view of individual sessions
- **Error Tracking**: Real-time error monitoring with alerts
- **User Analytics**: Engagement metrics and agent preferences

```bash
# Install webapp dependencies
cd webapp
pip install -r requirements.txt

# Start the dashboard
uvicorn main:app --reload --port 8000

# Open http://localhost:8000
```

**API Endpoints:**
- `/api/summary` - Overall statistics
- `/api/events/stream` - Real-time event stream (SSE)
- `/api/events/recent` - Recent events
- `/api/tools/usage` - Tool usage analytics
- `/api/latency/trend` - Latency trend
- `/api/sessions/{id}/timeline` - Session trace
- `/health` - Health check

See `webapp/README.md` for full documentation.

## Analytics Queries

### Event Distribution
```sql
SELECT event_type, COUNT(*) as count
FROM `PROJECT.DATASET.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
GROUP BY event_type;
```

### Tool Usage by Agent
```sql
SELECT
    agent,
    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
    COUNT(*) as call_count
FROM `PROJECT.DATASET.agent_events_v2`
WHERE event_type = 'TOOL_COMPLETED'
GROUP BY agent, tool_name;
```

### Latency Analysis
```sql
SELECT
    agent,
    AVG(CAST(JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') AS FLOAT64)) as avg_latency
FROM `PROJECT.DATASET.agent_events_v2`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY agent;
```

## Troubleshooting

1. **"Dataset does not exist"**: Create the dataset first using `bq mk`
2. **Authentication errors**: Run `gcloud auth application-default login`
3. **Permission errors**: Ensure your account has BigQuery Data Editor role
4. **Model not found**: Ensure you have access to the Gemini model in your project
