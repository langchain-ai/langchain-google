#!/usr/bin/env python3
"""Using the BigQuery callback handler with plain LangChain (no LangGraph).

This example shows:
- Request-level INVOCATION_STARTING / INVOCATION_COMPLETED spans around a
  plain LCEL chain via `handler.graph_context()`
- Passing `metadata=` so `session_id` / `user_id` land on every event
- Flushing at the end of each request

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langchain-google-genai

Usage:
    python plain_langchain_example.py
"""

import os
import uuid

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration - Update these for your environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-project-id")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")


def main() -> None:
    """Run one request through a plain LangChain chain."""
    handler = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        config=BigQueryLoggerConfig(
            batch_size=1,  # Write immediately for demo purposes
            batch_flush_interval=0.5,
            custom_tags={"service": "plain-langchain-demo"},
        ),
        graph_name="plain_langchain_app",  # Default value of the `agent` column
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", project=PROJECT_ID)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using only the context.\nContext: {context}"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    # Pass the same metadata to graph_context() and to the chain so that
    # session_id / user_id are set on INVOCATION_* *and* CHAIN_* / LLM_* rows.
    request_metadata = {
        "session_id": f"session-{uuid.uuid4().hex[:8]}",
        "user_id": "demo-user",
    }

    with handler.graph_context("plain_langchain_app", metadata=request_metadata):
        answer = chain.invoke(
            {
                "context": "The plan covers massage at 80% after deductible.",
                "question": "Is massage therapy covered?",
            },
            config={"callbacks": [handler], "metadata": request_metadata},
        )
    print(f"Answer: {answer}")

    # Make this request's events durable before returning, then shut down.
    handler.flush(timeout=5.0)
    handler.shutdown()

    print("\nDone! Check BigQuery for logged events:")
    print(f"""
    SELECT timestamp, event_type, agent, session_id, user_id,
           JSON_VALUE(latency_ms, '$.total_ms') AS latency_ms
    FROM `{PROJECT_ID}.{DATASET_ID}.agent_events`
    WHERE session_id = '{request_metadata["session_id"]}'
    ORDER BY timestamp;
    """)


if __name__ == "__main__":
    main()
