#!/usr/bin/env python3
"""Basic example demonstrating the BigQuery callback handler.

This example shows:
- Basic callback handler setup
- LLM call logging with latency tracking
- Chain execution logging
- Error handling

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langchain-google-genai

Usage:
    python basic_example.py
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration - Update these for your environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")


def main() -> None:
    """Run the basic example."""
    print(f"Using project: {PROJECT_ID}")
    print(f"Using dataset: {DATASET_ID}")

    # Create the callback handler with custom config
    config = BigQueryLoggerConfig(
        batch_size=1,  # Write immediately for demo purposes
        batch_flush_interval=0.5,
    )

    handler = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config,
    )

    print("\n1. Testing basic LLM call...")

    # Create a simple chat model using Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        project=PROJECT_ID,
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
    )

    # Make a simple call with the callback handler
    response = llm.invoke(
        "What is 2 + 2? Answer in one word.",
        config={
            "callbacks": [handler],
            "metadata": {
                "session_id": "demo-session-001",
                "user_id": "demo-user",
                "agent": "basic_example",
            },
        },
    )
    print(f"LLM Response: {response.content}")

    print("\n2. Testing chain execution...")

    # Create a simple chain
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that gives brief answers."),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    # Run the chain
    response = chain.invoke(
        {"question": "What is the capital of France?"},
        config={
            "callbacks": [handler],
            "metadata": {
                "session_id": "demo-session-001",
                "user_id": "demo-user",
                "agent": "basic_example",
            },
        },
    )
    print(f"Chain Response: {response.content}")

    print("\n3. Testing error handling...")

    # Test with an invalid model to trigger an error
    try:
        bad_llm = ChatGoogleGenerativeAI(
            model="non-existent-model",
            project=PROJECT_ID,
            temperature=0,
        )
        bad_llm.invoke(
            "This will fail",
            config={
                "callbacks": [handler],
                "metadata": {
                    "session_id": "demo-session-001",
                    "agent": "basic_example",
                },
            },
        )
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}")

    # Shutdown the handler to flush remaining events
    print("\nShutting down handler...")
    handler.shutdown()

    print("\nDone! Check BigQuery for logged events:")
    print(f"""
    SELECT timestamp, event_type, status,
           JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') as latency_ms
    FROM `{PROJECT_ID}.{DATASET_ID}.agent_events_v2`
    WHERE DATE(timestamp) = CURRENT_DATE()
    ORDER BY timestamp DESC
    LIMIT 20;
    """)


if __name__ == "__main__":
    main()
