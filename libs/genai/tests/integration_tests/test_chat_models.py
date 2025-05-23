"""Test ChatGoogleGenerativeAI chat model."""

import asyncio
import json
from typing import Dict, Generator, List

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    Modality,
)

_MODEL = "models/gemini-1.5-flash-latest"
_VISION_MODEL = "models/gemini-2.0-flash-001"
_IMAGE_OUTPUT_MODEL = "models/gemini-2.0-flash-exp-image-generation"
_THINKING_MODEL = "models/gemini-2.5-flash-preview-05-20"
_B64_string = """iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABhGlDQ1BJQ0MgUHJvZmlsZQAAeJx9kT1Iw0AcxV8/xCIVQTuIKGSoTi2IijhqFYpQIdQKrTqYXPoFTRqSFBdHwbXg4Mdi1cHFWVcHV0EQ/ABxdXFSdJES/5cUWsR4cNyPd/ced+8Af6PCVDM4DqiaZaSTCSGbWxW6XxHECPoRQ0hipj4niil4jq97+Ph6F+dZ3uf+HL1K3mSATyCeZbphEW8QT29aOud94ggrSQrxOXHMoAsSP3JddvmNc9FhP8+MGJn0PHGEWCh2sNzBrGSoxFPEUUXVKN+fdVnhvMVZrdRY6578heG8trLMdZrDSGIRSxAhQEYNZVRgIU6rRoqJNO0nPPxDjl8kl0yuMhg5FlCFCsnxg//B727NwuSEmxROAF0vtv0xCnTvAs26bX8f23bzBAg8A1da219tADOfpNfbWvQI6NsGLq7bmrwHXO4Ag0+6ZEiOFKDpLxSA9zP6phwwcAv0rLm9tfZx+gBkqKvUDXBwCIwVKXvd492hzt7+PdPq7wdzbXKn5swsVgAAA8lJREFUeJx90dtPHHUUB/Dz+81vZhb2wrDI3soUKBSRcisF21iqqCRNY01NTE0k8aHpi0k18VJfjOFvUF9M44MmGrHFQqSQiKSmFloL5c4CXW6Fhb0vO3ufvczMzweiBGI9+eW8ffI95/yQqqrwv4UxBgCfJ9w/2NfSVB+Nyn6/r+vdLo7H6FkYY6yoABR2PJujj34MSo/d/nHeVLYbydmIp/bEO0fEy/+NMcbTU4/j4Vs6Lr0ccKeYuUKWS4ABVCVHmRdszbfvTgfjR8kz5Jjs+9RREl9Zy2lbVK9wU3/kWLJLCXnqza1bfVe7b9jLbIeTMcYu13Jg/aMiPrCwVFcgtDiMhnxwJ/zXVDwSdVCVMRV7nqzl2i9e/fKrw8mqSp84e2sFj3Oj8/SrF/MaicmyYhAaXu58NPAbeAeyzY0NLecmh2+ODN3BewYBAkAY43giI3kebrnsRmvV9z2D4ciOa3EBAf31Tp9sMgdxMTFm6j74/Ogb70VCYQKAAIDCXkOAIC6pkYBWdwwnpHEdf6L9dJtJKPh95DZhzFKMEWRAGL927XpWTmMA+s8DAOBYAoR483l/iHZ/8bXoODl8b9UfyH72SXepzbyRJNvjFGHKMlhvMBze+cH9+4lEuOOlU2X1tVkFTU7Om03q080NDGXV1cflRpHwaaoiiiildB8jhDLZ7HDfz2Yidba6Vn2L4fhzFrNRKy5OZ2QOZ1U5W8VtqlVH/iUHcM933zZYWS7Wtj66zZr65bzGJQt0glHgudi9XVzEl4vKw2kUPhO020oPYI1qYc+2Xc0bRXFwTLY0VXa2VibD/lBaIXm1UChN5JSRUcQQ1Tk/47Cf3x8bY7y17Y17PVYTG1UkLPBFcqik7Zoa9JcLYoHBqHhXNgd6gS1k9EJ1TQ2l9EDy1saErmQ2kGpwGC2MLOtCM8nZEV1K0tKJtEksSm26J/rHg2zzmabKisq939nHzqUH7efzd4f/nPGW6NP8ybNFrOsWQhpoCuuhnJ4hAnPhFam01K4oQMjBg/mzBjVhuvw2O++KKT+BIVxJKzQECBDLF2qu2WTMmCovtDQ1f8iyoGkUADBCCGPsdnvTW2OtFm01VeB06msvdWlpPZU0wJRG85ns84umU3k+VyxeEcWqvYUBAGsUrbvme4be99HFeisP/pwUOIZaOqQX31ISgrKmZhLHtXNXuJq68orrr5/9mBCglCLAGGPyy81votEbcjlKLrC9E8mhH3wdHRdcyyvjidSlxjftPJpD+o25JYvRHGFoZDdks1mBQhxJu9uxvwEiXuHnHbLd1AAAAABJRU5ErkJggg=="""  # noqa: E501


def _check_usage_metadata(message: AIMessage) -> None:
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] > 0
    assert message.usage_metadata["output_tokens"] > 0
    assert message.usage_metadata["total_tokens"] > 0
    if "output_token_details" in message.usage_metadata:
        thought_tokens = message.usage_metadata["output_token_details"]["reasoning"]
    else:
        thought_tokens = 0

    assert (
        message.usage_metadata["input_tokens"]
        + message.usage_metadata["output_tokens"]
        + thought_tokens
    ) == message.usage_metadata["total_tokens"]


async def test_chat_google_genai_abatch() -> None:
    """Test streaming tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.abatch(
        ["This is a test. Say 'foo'", "This is a test, say 'bar'"]
    )
    for token in result:
        assert isinstance(token.content, str)


async def test_chat_google_genai_abatch_tags() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.abatch(
        ["This is a test", "This is another test"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_chat_google_genai_batch() -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = llm.batch(["This is a test. Say 'foo'", "This is a test, say 'bar'"])
    for token in result:
        assert isinstance(token.content, str)


async def test_chat_google_genai_ainvoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = await llm.ainvoke("This is a test. Say 'foo'", config={"tags": ["foo"]})
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    _check_usage_metadata(result)


def test_chat_google_genai_invoke() -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    result = llm.invoke(
        "This is a test. Say 'foo'",
        config=dict(tags=["foo"]),
        generation_config=dict(top_k=2, top_p=1, temperature=0.7),
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert not result.content.startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_with_image() -> None:
    """Test invoke tokens with image from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_IMAGE_OUTPUT_MODEL)

    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config=dict(tags=["meow"]),
        generation_config=dict(
            top_k=2, top_p=1, temperature=0.7, response_modalities=["TEXT", "IMAGE"]
        ),
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], dict)
    assert result.content[0].get("type") == "image_url"
    assert isinstance(result.content[1], str)
    assert not result.content[1].startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_with_modalities() -> None:
    """Test invoke tokens with image from ChatGoogleGenerativeAI with response
    modalities."""
    llm = ChatGoogleGenerativeAI(
        model=_IMAGE_OUTPUT_MODEL,
        response_modalities=[Modality.TEXT, Modality.IMAGE],  # type: ignore[list-item]
    )

    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config=dict(tags=["meow"]),
        generation_config=dict(top_k=2, top_p=1, temperature=0.7),
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], dict)
    assert result.content[0].get("type") == "image_url"
    assert isinstance(result.content[1], str)
    assert not result.content[1].startswith(" ")
    _check_usage_metadata(result)


def test_chat_google_genai_invoke_thinking() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL, thinking_budget=100)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking_default() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking_configured_include_thoughts() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(
        model=_THINKING_MODEL, thinking_budget=100, include_thoughts=True
    )

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    content = result.content

    assert isinstance(content[0], dict)
    assert content[0].get("type") == "thinking"
    assert isinstance(content[0].get("thinking"), str)

    assert isinstance(content[1], str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking_include_thoughts() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL, include_thoughts=True)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    content = result.content

    assert isinstance(content[0], dict)
    assert content[0].get("type") == "thinking"
    assert isinstance(content[0].get("thinking"), str)

    assert isinstance(content[1], str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking_include_thoughts_genreation_config() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
        generation_config={"thinking_config": {"include_thoughts": True}},
    )

    assert isinstance(result, AIMessage)
    content = result.content

    assert isinstance(content[0], dict)
    assert content[0].get("type") == "thinking"
    assert isinstance(content[0].get("thinking"), str)

    assert isinstance(content[1], str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking_disabled() -> None:
    """Test invoke thinking model from ChatGoogleGenerativeAI with
    default thinking config"""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL, thinking_budget=0)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert "output_token_details" not in result.usage_metadata


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_no_image_generation_without_modalities() -> None:
    """Test invoke tokens with image from ChatGoogleGenerativeAI without response
    modalities."""
    llm = ChatGoogleGenerativeAI(model=_IMAGE_OUTPUT_MODEL)

    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config=dict(tags=["meow"]),
        generation_config=dict(top_k=2, top_p=1, temperature=0.7),
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert not result.content.startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_image_generation_with_modalities_merge() -> None:
    """Test invoke tokens with image from ChatGoogleGenerativeAI with response
    modalities specified in both modal init and invoke generation_config."""
    llm = ChatGoogleGenerativeAI(
        model=_IMAGE_OUTPUT_MODEL,
        response_modalities=[Modality.TEXT],  # type: ignore[list-item]
    )
    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config=dict(tags=["meow"]),
        generation_config=dict(
            top_k=2,
            top_p=1,
            temperature=0.7,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], dict)
    assert result.content[0].get("type") == "image_url"
    assert isinstance(result.content[1], str)
    assert not result.content[1].startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.xfail(reason=("investigate"))
def test_chat_google_genai_invoke_multimodal() -> None:
    messages: list = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Guess what's in this picture! You have 3 guesses.",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + _B64_string,
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)
    response = llm.invoke(messages)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0

    # Try streaming
    for chunk in llm.stream(messages):
        print(chunk)  # noqa: T201
        assert isinstance(chunk.content, str)
        assert len(chunk.content.strip()) > 0


def test_chat_google_genai_invoke_multimodal_by_url() -> None:
    messages: list = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Guess what's in this picture! You have 3 guesses.",
                },
                {
                    "type": "image_url",
                    "image_url": "https://picsum.photos/seed/picsum/200/200",
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)
    response = llm.invoke(messages)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0

    # Try streaming
    any_chunk = False
    for chunk in llm.stream(messages):
        print(chunk)  # noqa: T201
        assert isinstance(chunk.content, str)
        if chunk.content:
            any_chunk = True
    assert any_chunk


def test_chat_google_genai_invoke_multimodal_multiple_messages() -> None:
    messages: list = [
        HumanMessage(content="Hi there"),
        AIMessage(content="Hi, how are you?"),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "I'm doing great! Guess what's in this picture!",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64," + _B64_string,
                },
            ]
        ),
    ]
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)
    response = llm.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert len(response.content.strip()) > 0


def test_chat_google_genai_single_call_with_history() -> None:
    model = ChatGoogleGenerativeAI(model=_MODEL)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "model_name,convert_system_message_to_human",
    [(_MODEL, True), ("models/gemini-1.5-pro-latest", False)],
)
def test_chat_google_genai_system_message(
    model_name: str, convert_system_message_to_human: bool
) -> None:
    model = ChatGoogleGenerativeAI(
        model=model_name,
        convert_system_message_to_human=convert_system_message_to_human,
    )
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content="You're supposed to answer math questions.")
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([system_message, message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_generativeai_get_num_tokens_gemini() -> None:
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL)
    output = llm.get_num_tokens("How are you?")
    assert output == 4


def test_safety_settings_gemini() -> None:
    safety_settings: Dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE  # type: ignore[dict-item]
    }
    # test with safety filters on bind
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL).bind(
        safety_settings=safety_settings
    )
    output = llm.invoke("how to make a bomb?")
    assert isinstance(output, AIMessage)
    assert len(output.content) > 0

    # test direct to stream
    streamed_messages = []
    output_stream = llm.stream("how to make a bomb?", safety_settings=safety_settings)
    assert isinstance(output_stream, Generator)
    for message in output_stream:
        streamed_messages.append(message)
    assert len(streamed_messages) > 0

    # test as init param
    llm = ChatGoogleGenerativeAI(
        temperature=0, model=_MODEL, safety_settings=safety_settings
    )
    out2 = llm.invoke("how to make a bomb")
    assert isinstance(out2, AIMessage)
    assert len(out2.content) > 0


def test_chat_function_calling_with_multiple_parts() -> None:
    @tool
    def search(
        question: str,
    ) -> str:
        """
        Useful for when you need to answer questions or visit websites.
        You should ask targeted questions.
        """
        return "brown"

    tools = [search]

    safety: Dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH  # type: ignore[dict-item]
    }
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest", safety_settings=safety
    )
    llm_with_search = llm.bind(
        functions=tools,
    )
    llm_with_search_force = llm_with_search.bind(
        tool_config={
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["search"],
            }
        }
    )
    request = HumanMessage(
        content=(
            "Please tell the primary color of following birds: "
            "sparrow, hawk, crow by using search tool."
        )
    )
    response = llm_with_search_force.invoke([request])

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) > 0
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "search"
    tool_messages = []
    for tool_call in response.tool_calls:
        tool_response = search.run(tool_call["args"])
        tool_message = ToolMessage(
            name="search",
            content=json.dumps(tool_response),
            tool_call_id=tool_call["id"],
        )
        tool_messages.append(tool_message)
    assert len(tool_messages) > 0
    assert len(response.tool_calls) == len(tool_messages)

    result = llm_with_search.invoke([request, response, *tool_messages])

    assert isinstance(result, AIMessage)
    assert "brown" in result.content


def _check_tool_calls(response: BaseMessage, expected_name: str) -> None:
    """Check tool calls are as expected."""
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""

    # function_call
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == expected_name
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    _check_tool_call_args(arguments)

    # tool_calls
    tool_calls = response.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == expected_name
    _check_tool_call_args(tool_call["args"])


def _check_tool_call_args(tool_call_args: dict) -> None:
    assert tool_call_args == {
        "age": 27.0,
        "name": "Erick",
        "likes": ["apple", "banana"],
    }


def test_chat_vertexai_gemini_function_calling() -> None:
    class MyModel(BaseModel):
        name: str
        age: int
        likes: list[str]

    safety: Dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH  # type: ignore[dict-item]
    }
    # Test .bind_tools with BaseModel
    message = HumanMessage(
        content="My name is Erick and I am 27 years old. I like apple and banana."
    )
    model = ChatGoogleGenerativeAI(model=_MODEL, safety_settings=safety).bind_tools(
        [MyModel]
    )
    response = model.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int, likes: list[str]) -> None:
        """Invoke this with names and age and likes."""
        pass

    model = ChatGoogleGenerativeAI(model=_MODEL, safety_settings=safety).bind_tools(
        [my_model]
    )
    response = model.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool
    @tool
    def my_tool(name: str, age: int, likes: list[str]) -> None:
        """Invoke this with names and age and likes."""
        pass

    model = ChatGoogleGenerativeAI(model=_MODEL, safety_settings=safety).bind_tools(
        [my_tool]
    )
    response = model.invoke([message])
    _check_tool_calls(response, "my_tool")

    # Test streaming
    stream = model.stream([message])
    first = True
    for chunk in stream:
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "my_tool"
    arguments_str = tool_call_chunk["args"]
    arguments = json.loads(str(arguments_str))
    _check_tool_call_args(arguments)


# Test with model that supports tool choice (gemini 1.5) and one that doesn't
# (gemini 1).
@pytest.mark.parametrize("model_name", [_MODEL, "models/gemini-1.5-flash-latest"])
def test_chat_google_genai_function_calling_with_structured_output(
    model_name: str,
) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety: Dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH  # type: ignore[dict-item]
    }
    llm = ChatGoogleGenerativeAI(model=model_name, safety_settings=safety)
    model = llm.with_structured_output(MyModel)
    message = HumanMessage(content="My name is Erick and I am 27 years old")

    response = model.invoke([message])
    assert isinstance(response, MyModel)
    assert response == MyModel(name="Erick", age=27)

    model = llm.with_structured_output(
        {
            "name": "MyModel",
            "description": "MyModel",
            "parameters": MyModel.model_json_schema(),
        }
    )
    response = model.invoke([message])
    expected = {"name": "Erick", "age": 27}
    assert response == expected


def test_ainvoke_without_eventloop() -> None:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    async def model_ainvoke(context: str) -> BaseMessage:
        result = await model.ainvoke(context)
        return result

    result = asyncio.run(model_ainvoke("How can you help me?"))
    assert isinstance(result, AIMessage)


def test_astream_without_eventloop() -> None:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    async def model_astream(context: str) -> List[BaseMessageChunk]:
        result = []
        async for chunk in model.astream(context):
            result.append(chunk)
        return result

    result = asyncio.run(model_astream("How can you help me?"))
    assert len(result) > 0
    assert isinstance(result[0], AIMessageChunk)
