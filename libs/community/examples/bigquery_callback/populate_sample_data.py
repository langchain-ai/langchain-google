#!/usr/bin/env python3
"""Populate sample data for LangGraph BigQuery analytics demo.

This script runs multiple agent scenarios to populate BigQuery with
diverse sample data for analytics demonstration.

Prerequisites:
    1. gcloud auth application-default login
    2. Dataset created: bq mk --dataset PROJECT_ID:agent_analytics
    3. pip install langchain-google-community langgraph langchain-google-genai

Usage:
    python populate_sample_data.py
"""

import os
import random
import time
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

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "test-project-0728-467323")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "agent_analytics")

# User profiles for simulation
USERS = ["alice", "bob", "charlie", "diana", "eve"]
AGENTS = ["finance_assistant", "travel_planner", "customer_support"]


class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


# Define realistic tools
@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a given ticker symbol."""
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
        return f"{symbol}: ${data['price']:.2f} ({direction}{data['change']:.2f}, {direction}{data['percent']:.2f}%)"
    return f"Stock symbol '{symbol}' not found."


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "san francisco": {"temp": 18, "condition": "Partly Cloudy", "humidity": 72},
        "tokyo": {"temp": 24, "condition": "Sunny", "humidity": 55},
        "london": {"temp": 14, "condition": "Overcast", "humidity": 85},
        "new york": {"temp": 22, "condition": "Clear", "humidity": 60},
        "paris": {"temp": 19, "condition": "Light Rain", "humidity": 78},
        "sydney": {"temp": 16, "condition": "Cloudy", "humidity": 68},
    }
    city_lower = city.lower().strip()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        return f"Weather in {city.title()}: {data['temp']}°C, {data['condition']}, Humidity: {data['humidity']}%"
    return f"Weather data for '{city}' not available."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another."""
    rates_to_usd = {
        "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067,
        "CNY": 0.14, "CAD": 0.74, "AUD": 0.65, "CHF": 1.12,
    }
    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()
    if from_curr not in rates_to_usd or to_curr not in rates_to_usd:
        return f"Unknown currency"
    usd_amount = amount * rates_to_usd[from_curr]
    result = usd_amount / rates_to_usd[to_curr]
    return f"{amount:,.2f} {from_curr} = {result:,.2f} {to_curr}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import math
    try:
        expr = expression.replace("^", "**")
        allowed_names = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "pi": math.pi}
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight between two cities."""
    price = random.randint(200, 1500)
    confirmation = f"FL{random.randint(100000, 999999)}"
    return f"Flight booked: {origin} -> {destination} on {date}. Confirmation: {confirmation}. Price: ${price}"


@tool
def book_hotel(city: str, check_in: str, check_out: str) -> str:
    """Book a hotel in a city."""
    hotels = ["Marriott", "Hilton", "Hyatt", "InterContinental", "Four Seasons"]
    hotel = random.choice(hotels)
    price = random.randint(100, 500)
    confirmation = f"HT{random.randint(100000, 999999)}"
    return f"Hotel booked: {hotel} in {city}. Check-in: {check_in}, Check-out: {check_out}. Confirmation: {confirmation}. Price: ${price}/night"


@tool
def check_order_status(order_id: str) -> str:
    """Check the status of an order."""
    statuses = ["Processing", "Shipped", "Out for Delivery", "Delivered"]
    status = random.choice(statuses)
    return f"Order {order_id}: {status}"


@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for an order."""
    refund_id = f"RF{random.randint(100000, 999999)}"
    amount = random.randint(20, 500)
    return f"Refund processed for order {order_id}. Reason: {reason}. Refund ID: {refund_id}. Amount: ${amount}"


def create_agent(agent_type: str) -> StateGraph:
    """Create an agent based on type."""
    if agent_type == "finance_assistant":
        tools = [get_stock_price, convert_currency, calculate]
    elif agent_type == "travel_planner":
        tools = [get_weather, book_flight, book_hotel, convert_currency]
    else:  # customer_support
        tools = [check_order_status, process_refund]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        project=PROJECT_ID,
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
    ).bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
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


# Sample queries for each agent type
SAMPLE_QUERIES = {
    "finance_assistant": [
        "What's the current price of GOOGL and AAPL?",
        "Convert 1000 USD to EUR",
        "Calculate the compound interest on $5000 at 4% for 5 years",
        "What's NVDA stock price?",
        "How much is 500 EUR in Japanese Yen?",
        "Calculate 15% of 2500",
        "Get me the stock prices for MSFT and META",
        "What's 10000 * 1.05^10?",
    ],
    "travel_planner": [
        "What's the weather in Tokyo? I'm planning a trip there.",
        "Book a flight from San Francisco to Tokyo for next Monday",
        "I need a hotel in Paris for 3 nights",
        "What's the weather like in London and Paris?",
        "How much is 2000 USD in EUR for my Europe trip?",
        "Book a flight from New York to London and find a hotel there",
    ],
    "customer_support": [
        "Check the status of my order ORD-12345",
        "I want a refund for order ORD-67890 because the item was damaged",
        "Where is my order ORD-11111?",
        "Process a return for order ORD-22222, wrong size",
        "What's the status of ORD-33333?",
    ],
}


def run_scenario(
    agent: StateGraph,
    handler: BigQueryCallbackHandler,
    query: str,
    session_id: str,
    agent_name: str,
    user_id: str,
) -> None:
    """Run a single query scenario."""
    with handler.graph_context(
        agent_name,
        metadata={
            "session_id": session_id,
            "user_id": user_id,
            "agent": agent_name,
        },
    ):
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={
                    "callbacks": [handler],
                    "metadata": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "agent": agent_name,
                    },
                },
            )
            final_message = result["messages"][-1]
            print(f"  Response: {str(final_message.content)[:80]}...")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Populate sample data for analytics demo."""
    print("=" * 60)
    print("Populating Sample Data for LangGraph Analytics Demo")
    print("=" * 60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")

    config = BigQueryLoggerConfig(
        batch_size=1,
        batch_flush_interval=0.5,
    )

    # Run multiple scenarios for each agent type
    scenario_count = 0
    for agent_name in AGENTS:
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print("=" * 60)

        handler = BigQueryCallbackHandler(
            project_id=PROJECT_ID,
            dataset_id=DATASET_ID,
            table_id="agent_events_v2",
            config=config,
            graph_name=agent_name,
        )

        agent = create_agent(agent_name)
        queries = SAMPLE_QUERIES[agent_name]

        for i, query in enumerate(queries):
            user_id = random.choice(USERS)
            session_id = f"{agent_name}-{user_id}-{int(time.time())}-{i}"

            print(f"\n[{user_id}] {query}")
            run_scenario(agent, handler, query, session_id, agent_name, user_id)
            scenario_count += 1

            # Small delay between requests
            time.sleep(1)

        handler.shutdown()

    print(f"\n{'='*60}")
    print(f"Done! Created {scenario_count} scenarios across {len(AGENTS)} agents.")
    print("=" * 60)


if __name__ == "__main__":
    main()
