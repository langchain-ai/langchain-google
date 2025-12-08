import pytest
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-2.5-flash-lite"


@tool
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"


@tool
def check_live_traffic(location: str) -> str:
    """Return the live traffic for the specified location."""
    return f"The traffic is standstill in {location}"


@tool
def check_tennis_score(player: str) -> str:
    """Return the latest player's tennis score."""
    return f"{player} is currently winning 6-0"


@pytest.mark.flaky(retries=3, delay=1)
def test_multiple_tools(backend_config: dict) -> None:
    tools = [check_weather, check_live_traffic, check_tennis_score]

    model = ChatGoogleGenerativeAI(
        model=MODEL,
        **backend_config,
    )

    model_with_tools = model.bind_tools(tools)

    input_ = "What is the latest tennis score for Leonid?"

    result = model_with_tools.invoke(input_)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "check_tennis_score"


def test_parallel_tool_calls(backend_config: dict) -> None:
    """Test that the model can make multiple tool calls in a single response."""
    tools = [check_weather, check_live_traffic]

    model = ChatGoogleGenerativeAI(
        model=MODEL,
        **backend_config,
    )

    model_with_tools = model.bind_tools(tools)

    # Ask for multiple pieces of information that require different tools
    input_ = (
        "I need both the weather AND the traffic for San Francisco. "
        "Please call both tools to get this information."
    )

    result = model_with_tools.invoke(input_)

    # Model should make at least 2 tool calls
    assert len(result.tool_calls) >= 2, (
        f"Expected at least 2 parallel tool calls, got {len(result.tool_calls)}"
    )

    # Verify both tools were called
    tool_names = {tc["name"] for tc in result.tool_calls}
    assert "check_weather" in tool_names, "check_weather tool was not called"
    assert "check_live_traffic" in tool_names, "check_live_traffic tool was not called"
