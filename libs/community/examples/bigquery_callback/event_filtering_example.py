#!/usr/bin/env python3
"""Event filtering example for BigQuery callback handler.

This example demonstrates:
- Using event_allowlist to only log specific events
- Using event_denylist to exclude noisy events
- Comparing logged events with different configurations

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langchain-google-genai

Usage:
    python event_filtering_example.py
"""

import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration - Update these for your environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")


def run_with_handler(
    handler: BigQueryCallbackHandler,
    description: str,
) -> None:
    """Run a chain with the given handler.

    Args:
        handler: The BigQuery callback handler.
        description: Description of the test.
    """
    print(f"\n{description}")
    print("-" * 50)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        project=PROJECT_ID,
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Be brief."),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {"question": "What is Python?"},
        config={
            "callbacks": [handler],
            "metadata": {
                "session_id": f"filter-test-{int(time.time())}",
                "agent": "filtering_example",
            },
        },
    )

    print(f"Response: {response.content[:100]}...")
    handler.shutdown()


def main() -> None:
    """Run the event filtering examples."""
    print("=" * 60)
    print("Event Filtering Example")
    print("=" * 60)

    # Test 1: No filtering (log everything)
    print("\n" + "=" * 60)
    print("Test 1: No Filtering (All Events)")
    print("=" * 60)
    print("Config: No allowlist or denylist")

    config1 = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        # No filtering - all events logged
    )

    handler1 = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config1,
    )

    print("\nExpected events: LLM_REQUEST, LLM_RESPONSE, CHAIN_START, CHAIN_END, TEXT")
    run_with_handler(handler1, "Running chain with no filtering...")

    # Test 2: Allowlist - only LLM events
    print("\n" + "=" * 60)
    print("Test 2: Allowlist - Only LLM Events")
    print("=" * 60)
    print("Config: event_allowlist=['LLM_REQUEST', 'LLM_RESPONSE']")

    config2 = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        event_allowlist=["LLM_REQUEST", "LLM_RESPONSE"],
    )

    handler2 = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config2,
    )

    print("\nExpected events: LLM_REQUEST, LLM_RESPONSE only")
    run_with_handler(handler2, "Running chain with LLM allowlist...")

    # Test 3: Denylist - exclude chain events
    print("\n" + "=" * 60)
    print("Test 3: Denylist - Exclude Chain Events")
    print("=" * 60)
    print("Config: event_denylist=['CHAIN_START', 'CHAIN_END']")

    config3 = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        event_denylist=["CHAIN_START", "CHAIN_END"],
    )

    handler3 = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config3,
    )

    print("\nExpected events: LLM_REQUEST, LLM_RESPONSE, TEXT (no CHAIN_*)")
    run_with_handler(handler3, "Running chain with chain events excluded...")

    # Test 4: Production-like config - only important events
    print("\n" + "=" * 60)
    print("Test 4: Production Config - Key Events Only")
    print("=" * 60)
    print("Config: allowlist for errors and responses only")

    config4 = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
        event_allowlist=[
            "LLM_RESPONSE",
            "LLM_ERROR",
            "TOOL_COMPLETED",
            "TOOL_ERROR",
            "GRAPH_END",
            "GRAPH_ERROR",
        ],
    )

    handler4 = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config4,
    )

    print("\nExpected events: LLM_RESPONSE only (no errors in this run)")
    run_with_handler(handler4, "Running chain with production config...")

    # Test 5: Disabled logging
    print("\n" + "=" * 60)
    print("Test 5: Disabled Logging")
    print("=" * 60)
    print("Config: enabled=False")

    config5 = BigQueryLoggerConfig(
        enabled=False,  # Completely disable logging
    )

    handler5 = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config5,
    )

    print("\nExpected events: None (logging disabled)")
    run_with_handler(handler5, "Running chain with logging disabled...")

    # Print query to compare results
    print("\n" + "=" * 60)
    print("Comparison Query")
    print("=" * 60)
    print(f"""
-- Compare event counts by session to see filtering in action:

SELECT
    session_id,
    event_type,
    COUNT(*) as event_count
FROM `{PROJECT_ID}.{DATASET_ID}.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
  AND agent = 'filtering_example'
GROUP BY session_id, event_type
ORDER BY session_id, event_type;
    """)


if __name__ == "__main__":
    main()
