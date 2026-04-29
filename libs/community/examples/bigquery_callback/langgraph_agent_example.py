#!/usr/bin/env python3
"""LangGraph ReAct agent example with BigQuery callback handler.

This example demonstrates:
- LangGraph agent with realistic tool calls
- NODE_STARTING/NODE_COMPLETED event tracking
- GRAPH_START/GRAPH_END via context manager
- Tool name tracking across callbacks
- Execution order tracking
- Full latency measurements with component breakdown
- Using Gemini 3 Flash model with google-genai SDK

Prerequisites:
    1. gcloud auth application-default login
    2. Create dataset: bq mk --dataset YOUR_PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langgraph langchain-google-genai

Usage:
    python langgraph_agent_example.py
"""

import os
import random
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_google_community.callbacks.bigquery_callback import (
    BigQueryCallbackHandler,
    BigQueryLoggerConfig,
)

# Configuration - Update these for your environment
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")


# Define the state for the agent
class AgentState(TypedDict):
    """State for the ReAct agent."""

    messages: Annotated[list[BaseMessage], add_messages]


# Define realistic tools
@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given ticker symbol.

    Args:
        symbol: The stock ticker symbol (e.g., GOOGL, AAPL, MSFT).

    Returns:
        Current stock price information.
    """
    # Simulated stock data (in production, use a real API)
    stock_data = {
        "GOOGL": {"price": 178.52, "change": 2.34, "percent": 1.33},
        "AAPL": {"price": 227.63, "change": -1.12, "percent": -0.49},
        "MSFT": {"price": 425.89, "change": 5.67, "percent": 1.35},
        "AMZN": {"price": 198.45, "change": 3.21, "percent": 1.64},
        "NVDA": {"price": 142.67, "change": -2.89, "percent": -1.98},
        "META": {"price": 612.34, "change": 8.45, "percent": 1.40},
        "TSLA": {"price": 248.92, "change": -4.56, "percent": -1.80},
    }

    symbol = symbol.upper().strip()
    if symbol in stock_data:
        data = stock_data[symbol]
        direction = "+" if data["change"] > 0 else ""
        return (
            f"{symbol}: ${data['price']:.2f} "
            f"({direction}{data['change']:.2f}, {direction}{data['percent']:.2f}%)"
        )
    return f"Stock symbol '{symbol}' not found. Try GOOGL, AAPL, MSFT, AMZN, NVDA, META, or TSLA."


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name (e.g., San Francisco, Tokyo, London).

    Returns:
        Current weather conditions.
    """
    # Simulated weather data
    weather_data = {
        "san francisco": {
            "temp": 18,
            "condition": "Partly Cloudy",
            "humidity": 72,
            "wind": "12 km/h NW",
        },
        "tokyo": {
            "temp": 24,
            "condition": "Sunny",
            "humidity": 55,
            "wind": "8 km/h E",
        },
        "london": {
            "temp": 14,
            "condition": "Overcast",
            "humidity": 85,
            "wind": "18 km/h SW",
        },
        "new york": {
            "temp": 22,
            "condition": "Clear",
            "humidity": 60,
            "wind": "15 km/h S",
        },
        "paris": {
            "temp": 19,
            "condition": "Light Rain",
            "humidity": 78,
            "wind": "10 km/h W",
        },
        "sydney": {
            "temp": 16,
            "condition": "Cloudy",
            "humidity": 68,
            "wind": "20 km/h SE",
        },
    }

    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return (
            f"Weather in {city.title()}:\n"
            f"  Temperature: {data['temp']}°C ({data['temp'] * 9 // 5 + 32}°F)\n"
            f"  Condition: {data['condition']}\n"
            f"  Humidity: {data['humidity']}%\n"
            f"  Wind: {data['wind']}"
        )
    return f"Weather data for '{city}' not available. Try: San Francisco, Tokyo, London, New York, Paris, or Sydney."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g., USD, EUR, JPY).
        to_currency: Target currency code (e.g., USD, EUR, JPY).

    Returns:
        Converted amount with exchange rate.
    """
    # Exchange rates relative to USD (simulated)
    rates_to_usd = {
        "USD": 1.0,
        "EUR": 1.08,
        "GBP": 1.27,
        "JPY": 0.0067,
        "CNY": 0.14,
        "CAD": 0.74,
        "AUD": 0.65,
        "CHF": 1.12,
        "INR": 0.012,
    }

    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()

    if from_curr not in rates_to_usd:
        return f"Unknown currency: {from_curr}"
    if to_curr not in rates_to_usd:
        return f"Unknown currency: {to_curr}"

    # Convert to USD first, then to target
    usd_amount = amount * rates_to_usd[from_curr]
    result = usd_amount / rates_to_usd[to_curr]
    rate = rates_to_usd[from_curr] / rates_to_usd[to_curr]

    return f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr} (rate: 1 {from_curr} = {rate:.4f} {to_curr})"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression (e.g., "15 * 23 + 7", "sqrt(144)", "2^10").

    Returns:
        The result of the calculation.
    """
    import math

    try:
        # Replace common math notations
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")

        # Safe evaluation with math functions
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "round": round,
            "pi": math.pi,
            "e": math.e,
        }

        # Only allow safe characters
        allowed_chars = set("0123456789+-*/().  sqrtcosintanlogexpabsroundpie")
        if not all(c in allowed_chars for c in expr.lower()):
            return f"Error: Expression contains invalid characters"

        result = eval(expr, {"__builtins__": {}}, allowed_names)
        if isinstance(result, float):
            if result == int(result):
                return f"Result: {int(result)}"
            return f"Result: {result:.6f}".rstrip("0").rstrip(".")
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time with timezone info.

    Returns:
        Current date, time, and day of week.
    """
    now = datetime.now()
    return (
        f"Current Date/Time:\n"
        f"  Date: {now.strftime('%B %d, %Y')}\n"
        f"  Time: {now.strftime('%I:%M:%S %p')}\n"
        f"  Day: {now.strftime('%A')}\n"
        f"  Week: {now.isocalendar()[1]} of {now.year}"
    )


@tool
def generate_random_number(min_val: int, max_val: int) -> str:
    """Generate a random number within a specified range.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Returns:
        A random number within the range.
    """
    if min_val > max_val:
        return f"Error: min_val ({min_val}) cannot be greater than max_val ({max_val})"
    result = random.randint(min_val, max_val)
    return f"Random number between {min_val} and {max_val}: {result}"


def create_agent() -> StateGraph:
    """Create the LangGraph ReAct agent with Gemini 3 Flash.

    Returns:
        Compiled LangGraph agent.
    """
    # Define tools
    tools = [
        get_stock_price,
        get_weather,
        convert_currency,
        calculate,
        get_current_datetime,
        generate_random_number,
    ]

    # Create the LLM with Gemini 3 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        project=PROJECT_ID,
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
    ).bind_tools(tools)

    # Define the agent node
    def agent_node(state: AgentState) -> dict:
        """The agent node that decides what to do."""
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Define the conditional edge
    def should_continue(state: AgentState) -> str:
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def run_agent(
    agent: StateGraph,
    handler: BigQueryCallbackHandler,
    query: str,
    session_id: str,
) -> None:
    """Run the agent with the graph context manager.

    Args:
        agent: The compiled LangGraph agent.
        handler: The BigQuery callback handler.
        query: The user query.
        session_id: The session ID for tracking.
    """
    print(f"\nQuery: {query}")
    print("-" * 60)

    # Use the graph context manager to wrap the execution
    with handler.graph_context(
        "finance_assistant",
        metadata={
            "session_id": session_id,
            "user_id": "demo-user",
            "agent": "gemini3_agent",
        },
    ):
        # Run the agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "callbacks": [handler],
                "metadata": {
                    "session_id": session_id,
                    "user_id": "demo-user",
                    "agent": "gemini3_agent",
                },
            },
        )

        # Get the final response
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            print(f"Response: {final_message.content}")


def main() -> None:
    """Run the LangGraph agent example."""
    print("=" * 60)
    print("LangGraph Agent with Gemini 3 Flash & BigQuery Logging")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")

    # Create the callback handler with LangGraph support
    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    handler = BigQueryCallbackHandler(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id="agent_events_v2",
        config=config,
        graph_name="finance_assistant",
    )

    # Create the agent
    agent = create_agent()

    # Test 1: Stock price lookup
    print("\n" + "=" * 60)
    print("Test 1: Stock Price Lookup")
    print("=" * 60)
    run_agent(
        agent,
        handler,
        "What's the current price of Google (GOOGL) and Apple (AAPL) stock?",
        "session-gemini3-001",
    )

    # Test 2: Weather + Currency conversion
    print("\n" + "=" * 60)
    print("Test 2: Weather & Currency Conversion")
    print("=" * 60)
    run_agent(
        agent,
        handler,
        "What's the weather in Tokyo? Also, how much is 1000 USD in Japanese Yen?",
        "session-gemini3-002",
    )

    # Test 3: Complex calculation
    print("\n" + "=" * 60)
    print("Test 3: Mathematical Calculation")
    print("=" * 60)
    run_agent(
        agent,
        handler,
        "Calculate the compound interest on $10,000 at 5% for 10 years. Use the formula: principal * (1 + rate)^years",
        "session-gemini3-003",
    )

    # Test 4: Multiple tools in sequence
    print("\n" + "=" * 60)
    print("Test 4: Multi-Tool Query")
    print("=" * 60)
    run_agent(
        agent,
        handler,
        "I'm planning a trip to Paris. What's the weather there? "
        "What time is it now? And how much would 500 EUR be in USD?",
        "session-gemini3-004",
    )

    # Test 5: Random number generation
    print("\n" + "=" * 60)
    print("Test 5: Random Number Generation")
    print("=" * 60)
    run_agent(
        agent,
        handler,
        "Generate a random lottery number between 1 and 49, then calculate its square root.",
        "session-gemini3-005",
    )

    # Shutdown the handler
    print("\n" + "=" * 60)
    print("Shutting down handler...")
    handler.shutdown()

    # Print query to view results
    print("\nDone! View results in BigQuery:")
    print(
        f"""
SELECT
    timestamp,
    event_type,
    session_id,
    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
    JSON_EXTRACT_SCALAR(latency_ms, '$.total_ms') as latency_ms,
    status
FROM `{PROJECT_ID}.{DATASET_ID}.agent_events_v2`
WHERE DATE(timestamp) = CURRENT_DATE()
  AND agent = 'gemini3_agent'
ORDER BY timestamp DESC
LIMIT 50;
    """
    )


if __name__ == "__main__":
    main()
