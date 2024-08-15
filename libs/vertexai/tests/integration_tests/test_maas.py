import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_google_vertexai.model_garden import (
    VertexMaaS,
)

model_names = [
    "meta/llama3-405b-instruct-maas",
    "mistral-nemo@2407",
    "mistral-large@2407",
]


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_model_garden(model_name: str) -> None:
    llm = VertexMaaS(model=model_name, location="us-central1")
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)
    assert llm._llm_type == "vertexai_model_garden_maas"


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_chat_with_history(model_name: str) -> None:
    llm = VertexMaaS(model=model_name, location="us-central1")
    message1 = HumanMessage(content="Hi! My name is Leo. How you can help me?")
    output1 = llm.invoke([message1])
    assert isinstance(output1, AIMessage)
    message2 = HumanMessage(content="Do you remember what is my name?")
    output2 = llm.invoke([message1, output1, message2])
    assert isinstance(output2, AIMessage)
    assert "Leo" in output2.content


@pytest.mark.extended
def test_chat_with_tools_llama() -> None:
    llm = VertexMaaS(
        model="meta/llama3-405b-instruct-maas",
        location="us-central1",
        append_tools_to_system_message=True,
    )

    @tool
    def get_weather(city: str) -> float:
        """Get the current weather and temperature for a given city."""
        return 23.0

    tools = [get_weather]
    message1 = HumanMessage(content="What is the current temperature like in Munich?")
    output1 = llm.invoke([message1], tools=tools)
    print(output1)
    assert isinstance(output1, AIMessage)
    assert len(output1.tool_calls) == 1
    assert output1.tool_calls[0]["name"] == "get_weather"
    message2 = ToolMessage(
        name="get_weather", content="23.0", tool_call_id=output1.tool_calls[0]["id"]
    )
    output2 = llm.invoke([message1, output1, message2], tools=tools)
    print(output2)


@pytest.mark.extended
@pytest.mark.parametrize(
    "model_name",
    [
        "meta/llama3-405b-instruct-maas",
        "mistral-nemo@2407",
    ],
)
def test_maas_stream(model_name: str) -> None:
    llm = VertexMaaS(model=model_name, location="us-central1")
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    message = HumanMessage(content=question)
    sync_response = llm.stream([message], model="claude-3-sonnet@20240229")
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)
