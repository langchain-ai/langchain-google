from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL = "gemini-flash-lite-latest"


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


def test_multiple_tools() -> None:
    tools = [check_weather, check_live_traffic, check_tennis_score]

    model = ChatGoogleGenerativeAI(
        model=MODEL,
    )

    model_with_tools = model.bind_tools(tools)

    input_ = "What is the latest tennis score for Leonid?"

    result = model_with_tools.invoke(input_)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "check_tennis_score"
