#!/usr/bin/env python3
"""Async example demonstrating the AsyncBigQueryCallbackHandler.

This example demonstrates:
- Async callback handler usage
- Async LangGraph agent execution
- Async graph context manager
- Concurrent operations with proper tracking

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langgraph langchain-google-genai

Usage:
    python async_example.py
"""

import asyncio
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_community.callbacks.bigquery_callback import (
    AsyncBigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration - Update these for your environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")


class AgentState(TypedDict):
    """State for the async agent."""

    messages: Annotated[list[BaseMessage], add_messages]


@tool
def async_search(query: str) -> str:
    """Search for information.

    Args:
        query: The search query.

    Returns:
        Search results.
    """
    return f"Async search results for: {query}"


@tool
def async_calculator(expression: str) -> str:
    """Calculate a math expression.

    Args:
        expression: Math expression.

    Returns:
        Calculation result.
    """
    try:
        # Safe evaluation
        allowed_chars = set("0123456789+-*/().  ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def create_async_agent() -> StateGraph:
    """Create an async LangGraph agent."""
    tools = [async_search, async_calculator]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        project=PROJECT_ID,
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
    ).bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        """Async agent node."""
        messages = state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """Determine if we should continue."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def run_single_query(
    agent: StateGraph,
    handler: AsyncBigQueryCallbackHandler,
    query: str,
    session_id: str,
) -> str:
    """Run a single query asynchronously.

    Args:
        agent: The compiled agent.
        handler: The async callback handler.
        query: The user query.
        session_id: Session identifier.

    Returns:
        The agent's response.
    """
    async with handler.graph_context(
        "async_agent",
        metadata={"session_id": session_id, "agent": "async_example"},
    ):
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
                "metadata": {"session_id": session_id, "agent": "async_example"},
            },
        )

        final_message = result["messages"][-1]
        return final_message.content if isinstance(final_message, AIMessage) else str(final_message)


async def main() -> None:
    """Run the async example."""
    print("=" * 60)
    print("Async BigQuery Callback Handler Example")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")

    # Create async callback handler
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    handler = AsyncBigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config,
        graph_name="async_agent",
    )

    # Create the agent
    agent = create_async_agent()

    # Test 1: Single async query
    print("\n" + "=" * 60)
    print("Test 1: Single Async Query")
    print("=" * 60)

    response = await run_single_query(
        agent,
        handler,
        "What is 25 * 4?",
        "async-session-001",
    )
    print(f"Response: {response}")

    # Test 2: Concurrent queries
    print("\n" + "=" * 60)
    print("Test 2: Concurrent Queries")
    print("=" * 60)

    queries = [
        ("Calculate 100 + 200", "async-session-002a"),
        ("Search for Python programming", "async-session-002b"),
        ("What is 50 / 2?", "async-session-002c"),
    ]

    # Run queries concurrently
    tasks = [
        run_single_query(agent, handler, query, session)
        for query, session in queries
    ]

    print("Running 3 queries concurrently...")
    responses = await asyncio.gather(*tasks)

    for (query, _), response in zip(queries, responses):
        print(f"\nQuery: {query}")
        print(f"Response: {response[:100]}...")

    # Shutdown
    print("\n" + "=" * 60)
    print("Shutting down...")
    await handler.shutdown()

    print(f"""
Done! Query results in BigQuery:

SELECT
    session_id,
    event_type,
    JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') as latency_ms,
    timestamp
FROM `{PROJECT_ID}.{DATASET_ID}.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
  AND agent = 'async_example'
ORDER BY timestamp DESC
LIMIT 30;
    """)


if __name__ == "__main__":
    asyncio.run(main())
