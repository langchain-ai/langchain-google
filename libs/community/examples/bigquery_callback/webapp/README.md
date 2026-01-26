# LangGraph Agent Analytics Dashboard

A real-time monitoring dashboard for LangGraph agents using BigQuery as the backend.
This is a **Datadog-like observability platform** for your AI agents.

![Dashboard Preview](https://via.placeholder.com/800x400?text=LangGraph+Agent+Analytics+Dashboard)

## Features

### Real-time Monitoring
- **Live Event Stream**: Watch agent events as they happen with Server-Sent Events (SSE)
- **Auto-refresh**: All metrics update every 5 seconds
- **Summary Stats**: Total events, sessions, users, LLM calls, tool calls, and latency

### Analytics Dashboards
- **Event Distribution**: Doughnut chart showing event type breakdown
- **Agent Activity**: Stacked bar chart comparing LLM calls, tool calls, and errors by agent
- **Latency Trend**: Line chart showing avg and P95 latency over the last hour
- **Tool Usage Table**: Top tools by call count with latency and error metrics
- **User Engagement**: User-level analytics including sessions, queries, and error rates

### Session Tracing
- **Session Timeline**: Detailed trace view of every event in a session
- **Conversation Reconstruction**: Human-readable conversation flow
- **Span Tracking**: Distributed tracing with trace_id and span_id

### Error Tracking
- **Error Banner**: Prominent display when errors are detected
- **Error Summary**: Error counts by agent and event type
- **Recent Errors**: Quick access to latest error details

## Prerequisites

1. **Google Cloud Setup**:
   ```bash
   # Authenticate with your GCP account
   gcloud auth application-default login

   # Set your project
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **BigQuery Data**: Run the example agents to populate data:
   ```bash
   # From the bigquery_callback directory
   python populate_sample_data.py
   ```

## Installation

```bash
# Navigate to the webapp directory
cd webapp

# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

## Configuration

Set environment variables to customize the BigQuery connection:

```bash
export GCP_PROJECT_ID="your-project-id"
export BQ_DATASET_ID="agent_analytics"
export BQ_TABLE_ID="agent_events_v2"
```

Or modify the defaults in `main.py`:

```python
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-project-id")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "agent_events_v2")
```

## Running the Dashboard

```bash
# Start the server
uvicorn main:app --reload --port 8000

# Or run directly
python main.py
```

Then open your browser to: **http://localhost:8000**

## API Endpoints

### Summary Stats
| Endpoint | Description |
|----------|-------------|
| `GET /api/summary` | Overall summary statistics for today |
| `GET /api/summary/hourly` | Stats for the last hour |

### Events
| Endpoint | Description |
|----------|-------------|
| `GET /api/events/recent?limit=50` | Most recent events |
| `GET /api/events/stream` | SSE stream of live events |
| `GET /api/events/distribution` | Event type distribution |
| `GET /api/events/by-agent` | Events grouped by agent |

### Tools
| Endpoint | Description |
|----------|-------------|
| `GET /api/tools/usage` | Tool usage statistics |
| `GET /api/tools/heatmap` | Tool usage heatmap data |

### Latency
| Endpoint | Description |
|----------|-------------|
| `GET /api/latency/by-event-type` | Latency by event type |
| `GET /api/latency/trend` | Latency trend (last hour) |
| `GET /api/latency/hourly` | Hourly latency pattern |

### Errors
| Endpoint | Description |
|----------|-------------|
| `GET /api/errors/summary` | Error summary (last 7 days) |
| `GET /api/errors/recent?limit=20` | Recent errors |
| `GET /api/errors/rate` | Error rate over time |

### Sessions
| Endpoint | Description |
|----------|-------------|
| `GET /api/sessions/recent?limit=20` | Recent sessions |
| `GET /api/sessions/{session_id}/timeline` | Session timeline |
| `GET /api/sessions/{session_id}/conversation` | Session conversation |

### Users
| Endpoint | Description |
|----------|-------------|
| `GET /api/users/engagement` | User engagement metrics |
| `GET /api/users/agent-preference` | User-agent preference matrix |

### Time Series
| Endpoint | Description |
|----------|-------------|
| `GET /api/timeseries/activity` | Activity per minute (last hour) |
| `GET /api/timeseries/hourly-pattern` | Hourly activity pattern |

### Health
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check endpoint |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Dashboard.html │  │ Session Detail  │  │   Charts.js     │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │ HTTP/SSE            │                     │
┌───────────┼─────────────────────┼─────────────────────┼─────────┐
│           ▼                     ▼                     ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Server                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │ REST APIs   │  │ SSE Stream  │  │ Jinja2 Templates│  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └─────────────────┘  │   │
│  └─────────┼────────────────┼──────────────────────────────┘   │
│            │                │                                   │
│            ▼                ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BigQuery Client (google-cloud-bigquery)     │   │
│  └────────────────────────────┬────────────────────────────┘   │
└───────────────────────────────┼─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BigQuery                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              agent_analytics.agent_events_v2             │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │timestamp│ │event_type│ │ agent  │ │   content   │   │   │
│  │  │session_id│ │ user_id │ │latency │ │ attributes  │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Customization

### Adding New Charts

1. Add a new canvas element in `dashboard.html`:
   ```html
   <canvas id="my-chart"></canvas>
   ```

2. Initialize the chart in JavaScript:
   ```javascript
   const ctx = document.getElementById('my-chart').getContext('2d');
   charts.myChart = new Chart(ctx, { /* config */ });
   ```

3. Create an API endpoint in `main.py`:
   ```python
   @app.get("/api/my-data")
   async def get_my_data():
       sql = f"SELECT ... FROM `{FULL_TABLE_ID}` ..."
       return run_query(sql)
   ```

4. Add an update function:
   ```javascript
   async function updateMyChart() {
       const response = await fetch('/api/my-data');
       const data = await response.json();
       // Update chart...
   }
   ```

### Changing Refresh Rate

Modify `REFRESH_INTERVAL` in the dashboard HTML:

```javascript
const REFRESH_INTERVAL = 5000; // 5 seconds
```

### Custom Event Types

Add colors for new event types in the dashboard:

```javascript
const eventTypeColors = {
    'MY_CUSTOM_EVENT': '#FF5733',
    // ...
};
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Cloud Run

```bash
# Build and deploy
gcloud run deploy langgraph-analytics \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Environment Variables (Production)

```bash
GCP_PROJECT_ID=production-project
BQ_DATASET_ID=agent_analytics
BQ_TABLE_ID=agent_events_v2
```

## Troubleshooting

### "Dataset does not exist"
Create the dataset first:
```bash
bq mk --dataset YOUR_PROJECT_ID:agent_analytics
```

### "Permission denied"
Ensure your account has BigQuery Data Viewer role:
```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:YOUR_EMAIL" \
  --role="roles/bigquery.dataViewer"
```

### No data showing
1. Check if data exists:
   ```bash
   bq query "SELECT COUNT(*) FROM \`PROJECT.DATASET.TABLE\`"
   ```
2. Run the sample data script:
   ```bash
   python ../populate_sample_data.py
   ```

### SSE connection drops
The dashboard automatically reconnects after 5 seconds. If issues persist, check:
- Network connectivity
- BigQuery quota limits
- Server logs for errors

## License

Apache 2.0
