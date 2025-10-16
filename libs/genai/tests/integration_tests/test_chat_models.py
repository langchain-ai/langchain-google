"""Test ChatGoogleGenerativeAI chat model."""

import asyncio
import json
from collections.abc import Generator, Sequence
from typing import Literal, Optional, Union, cast

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
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
    Modality,
)

_MODEL = "gemini-flash-lite-latest"
_VISION_MODEL = "models/gemini-2.0-flash-001"
_IMAGE_OUTPUT_MODEL = "models/gemini-2.0-flash-exp-image-generation"
_AUDIO_OUTPUT_MODEL = "models/gemini-2.5-flash-preview-tts"
_THINKING_MODEL = "models/gemini-2.5-flash"
_B64_string = """iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABhGlDQ1BJQ0MgUHJvZmlsZQAAeJx9kT1Iw0AcxV8/xCIVQTuIKGSoTi2IijhqFYpQIdQKrTqYXPoFTRqSFBdHwbXg4Mdi1cHFWVcHV0EQ/ABxdXFSdJES/5cUWsR4cNyPd/ced+8Af6PCVDM4DqiaZaSTCSGbWxW6XxHECPoRQ0hipj4niil4jq97+Ph6F+dZ3uf+HL1K3mSATyCeZbphEW8QT29aOud94ggrSQrxOXHMoAsSP3JddvmNc9FhP8+MGJn0PHGEWCh2sNzBrGSoxFPEUUXVKN+fdVnhvMVZrdRY6578heG8trLMdZrDSGIRSxAhQEYNZVRgIU6rRoqJNO0nPPxDjl8kl0yuMhg5FlCFCsnxg//B727NwuSEmxROAF0vtv0xCnTvAs26bX8f23bzBAg8A1da219tADOfpNfbWvQI6NsGLq7bmrwHXO4Ag0+6ZEiOFKDpLxSA9zP6phwwcAv0rLm9tfZx+gBkqKvUDXBwCIwVKXvd492hzt7+PdPq7wdzbXKn5swsVgAAA8lJREFUeJx90dtPHHUUB/Dz+81vZhb2wrDI3soUKBSRcisF21iqqCRNY01NTE0k8aHpi0k18VJfjOFvUF9M44MmGrHFQqSQiKSmFloL5c4CXW6Fhb0vO3ufvczMzweiBGI9+eW8ffI95/yQqqrwv4UxBgCfJ9w/2NfSVB+Nyn6/r+vdLo7H6FkYY6yoABR2PJujj34MSo/d/nHeVLYbydmIp/bEO0fEy/+NMcbTU4/j4Vs6Lr0ccKeYuUKWS4ABVCVHmRdszbfvTgfjR8kz5Jjs+9RREl9Zy2lbVK9wU3/kWLJLCXnqza1bfVe7b9jLbIeTMcYu13Jg/aMiPrCwVFcgtDiMhnxwJ/zXVDwSdVCVMRV7nqzl2i9e/fKrw8mqSp84e2sFj3Oj8/SrF/MaicmyYhAaXu58NPAbeAeyzY0NLecmh2+ODN3BewYBAkAY43giI3kebrnsRmvV9z2D4ciOa3EBAf31Tp9sMgdxMTFm6j74/Ogb70VCYQKAAIDCXkOAIC6pkYBWdwwnpHEdf6L9dJtJKPh95DZhzFKMEWRAGL927XpWTmMA+s8DAOBYAoR483l/iHZ/8bXoODl8b9UfyH72SXepzbyRJNvjFGHKMlhvMBze+cH9+4lEuOOlU2X1tVkFTU7Om03q080NDGXV1cflRpHwaaoiiiildB8jhDLZ7HDfz2Yidba6Vn2L4fhzFrNRKy5OZ2QOZ1U5W8VtqlVH/iUHcM933zZYWS7Wtj66zZr65bzGJQt0glHgudi9XVzEl4vKw2kUPhO020oPYI1qYc+2Xc0bRXFwTLY0VXa2VibD/lBaIXm1UChN5JSRUcQQ1Tk/47Cf3x8bY7y17Y17PVYTG1UkLPBFcqik7Zoa9JcLYoHBqHhXNgd6gS1k9EJ1TQ2l9EDy1saErmQ2kGpwGC2MLOtCM8nZEV1K0tKJtEksSm26J/rHg2zzmabKisq939nHzqUH7efzd4f/nPGW6NP8ybNFrOsWQhpoCuuhnJ4hAnPhFam01K4oQMjBg/mzBjVhuvw2O++KKT+BIVxJKzQECBDLF2qu2WTMmCovtDQ1f8iyoGkUADBCCGPsdnvTW2OtFm01VeB06msvdWlpPZU0wJRG85ns84umU3k+VyxeEcWqvYUBAGsUrbvme4be99HFeisP/pwUOIZaOqQX31ISgrKmZhLHtXNXuJq68orrr5/9mBCglCLAGGPyy81votEbcjlKLrC9E8mhH3wdHRdcyyvjidSlxjftPJpD+o25JYvRHGFoZDdks1mBQhxJu9uxvwEiXuHnHbLd1AAAAABJRU5ErkJggg=="""  # noqa: E501


def get_wav_type_from_bytes(file_bytes: bytes) -> bool:
    """Determines if the given bytes represent a WAV file by inspecting the header.

    Args:
        file_bytes: Bytes representing the file content.

    Returns:
        True if the bytes represent a WAV file, False otherwise.
    """
    if len(file_bytes) < 12:
        return False

    # Check for RIFF header (bytes 0-3)
    if file_bytes[0:4] != b"RIFF":
        return False

    # Check for WAVE format (bytes 8-11)
    # Return whether bytes 8-11 match the WAVE signature
    return file_bytes[8:12] == b"WAVE"


def _check_usage_metadata(message: AIMessage) -> None:
    """Ensure usage metadata is present and valid (greater than 0 and correct sum)."""
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] > 0
    assert message.usage_metadata["output_tokens"] > 0
    assert message.usage_metadata["total_tokens"] > 0

    assert (
        message.usage_metadata["input_tokens"] + message.usage_metadata["output_tokens"]
    ) == message.usage_metadata["total_tokens"]


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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("with_tags", [False, True])
async def test_chat_google_genai_batch(is_async: bool, with_tags: bool) -> None:
    """Test batch tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)
    messages: Sequence[str] = [
        "This is a test. Say 'foo'",
        "This is a test, say 'bar'",
    ]
    config: Union[RunnableConfig, None] = {"tags": ["foo"]} if with_tags else None

    if is_async:
        result = await llm.abatch(cast(list, messages), config=config)
    else:
        result = llm.batch(cast(list, messages), config=config)

    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.parametrize("is_async", [False, True])
async def test_chat_google_genai_invoke(is_async: bool) -> None:
    """Test invoke tokens from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(model=_MODEL)

    if is_async:
        result = await llm.ainvoke(
            "This is a test. Say 'foo'",
            config={"tags": ["foo"]},
            generation_config={"top_k": 2, "top_p": 1, "temperature": 0.7},
        )
    else:
        result = llm.invoke(
            "This is a test. Say 'foo'",
            config={"tags": ["foo"]},
            generation_config={"top_k": 2, "top_p": 1, "temperature": 0.7},
        )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert not result.content.startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_with_image() -> None:
    """Test generating an image and then text from ChatGoogleGenerativeAI.

    Using `generation_config` to specify response modalities.

    Up to 9 retries possible due to inner loop and Pytest retries.
    """
    llm = ChatGoogleGenerativeAI(model=_IMAGE_OUTPUT_MODEL)

    for _ in range(3):
        # We break as soon as we get an image back, as it's not guaranteed
        result = llm.invoke(
            "Say 'meow!' and then Generate an image of a cat.",
            config={"tags": ["meow"]},
            generation_config={
                "top_k": 2,
                "top_p": 1,
                "temperature": 0.7,
                "response_modalities": ["TEXT", "IMAGE"],
            },
        )
        if (
            isinstance(result.content, list)
            and len(result.content) > 1
            and isinstance(result.content[1], dict)
        ):
            break
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], str)
    assert isinstance(result.content[1], dict)
    assert result.content[1].get("type") == "image_url"
    assert not result.content[0].startswith(" ")
    _check_usage_metadata(result)

    # Test we can pass back in
    next_message = {"role": "user", "content": "Thanks!"}
    _ = llm.invoke([result, next_message])

    # Test content_blocks property
    content_blocks = result.content_blocks
    assert isinstance(content_blocks, list)
    assert len(content_blocks) == 2
    assert isinstance(content_blocks[0], dict)
    assert content_blocks[0].get("type") == "text"
    assert isinstance(content_blocks[1], dict)
    assert content_blocks[1].get("type") == "image"

    # Test we can pass back in content_blocks
    _ = llm.invoke(["What's this?", {"role": "assistant", "content": content_blocks}])


def test_chat_google_genai_invoke_with_audio() -> None:
    """Test generating audio from ChatGoogleGenerativeAI."""
    llm = ChatGoogleGenerativeAI(
        model=_AUDIO_OUTPUT_MODEL, response_modalities=[Modality.AUDIO]
    )

    result = llm.invoke(
        "Please say The quick brown fox jumps over the lazy dog",
    )
    assert isinstance(result, AIMessage)
    assert result.content == ""
    audio_data = result.additional_kwargs.get("audio")
    assert isinstance(audio_data, bytes)
    assert get_wav_type_from_bytes(audio_data)
    _check_usage_metadata(result)

    # Test content_blocks property
    content_blocks = result.content_blocks
    assert isinstance(content_blocks, list)
    assert len(content_blocks) == 1
    assert isinstance(content_blocks[0], dict)
    assert content_blocks[0].get("type") == "audio"

    # Test we can pass back in
    # TODO: no model currently supports audio input


def test_chat_google_genai_invoke_thinking_default() -> None:
    """Test invoke thinking model with default thinking config."""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    if (
        "output_token_details" in result.usage_metadata
        and "reasoning" in result.usage_metadata["output_token_details"]
    ):
        assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def test_chat_google_genai_invoke_thinking() -> None:
    """Test invoke thinking model with `thinking_budget`."""
    llm = ChatGoogleGenerativeAI(model=_THINKING_MODEL, thinking_budget=100)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    if (
        "output_token_details" in result.usage_metadata
        and "reasoning" in result.usage_metadata["output_token_details"]
    ):
        assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def _check_thinking_output(content: list, output_version: str) -> None:
    if output_version == "v0":
        thinking_key = "thinking"
        assert isinstance(content[-1], str)

    else:
        # v1
        thinking_key = "reasoning"
        assert isinstance(content[-1], dict)
        assert content[-1].get("type") == "text"
        assert isinstance(content[-1].get("text"), str)

    assert isinstance(content, list)
    thinking_blocks = [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == thinking_key
    ]
    assert thinking_blocks
    for block in thinking_blocks:
        assert isinstance(block[thinking_key], str)


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_chat_google_genai_invoke_thinking_include_thoughts(
    output_version: str,
) -> None:
    """Test invoke thinking model with `include_thoughts` on the chat model."""
    llm = ChatGoogleGenerativeAI(
        model=_THINKING_MODEL, include_thoughts=True, output_version=output_version
    )

    input_message = {
        "role": "user",
        "content": (
            "How many O's are in Google? Please tell me how you double checked the "
            "result."
        ),
    }

    full: AIMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessage)

    response_metadata = full.response_metadata
    model_provider = response_metadata.get("model_provider", "google_genai")
    assert model_provider == "google_genai"

    _check_thinking_output(cast(list, full.content), output_version)
    _check_usage_metadata(full)
    assert full.usage_metadata is not None
    if (
        "output_token_details" in full.usage_metadata
        and "reasoning" in full.usage_metadata["output_token_details"]
    ):
        assert full.usage_metadata["output_token_details"]["reasoning"] > 0

    # Test we can pass back in
    next_message = {"role": "user", "content": "Thanks!"}
    result = llm.invoke([input_message, full, next_message])
    assert isinstance(result, AIMessage)
    _check_thinking_output(cast(list, result.content), output_version)


@pytest.mark.flaky(retries=5, delay=1)
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_chat_google_genai_invoke_thinking_with_tools(
    output_version: str,
) -> None:
    """Test thinking with function calling to get thought signatures.

    Ensure we can pass the response back in.
    """

    @tool
    def analyze_weather(location: str, date: str) -> dict:
        """Analyze weather patterns for a location on a specific date.

        Args:
            location: The city or region to analyze.
            date: The date in YYYY-MM-DD format.

        Returns:
            A dictionary with weather analysis.
        """
        return {
            "location": location,
            "date": date,
            "temperature": "72F",
            "conditions": "sunny",
            "analysis": "Pleasant weather expected",
        }

    llm = ChatGoogleGenerativeAI(
        model=_THINKING_MODEL, include_thoughts=True, output_version=output_version
    )
    llm_with_tools = llm.bind_tools([analyze_weather])

    input_message = {
        "role": "user",
        "content": (
            "I'm planning a trip to Paris. Should I check the weather for "
            "tomorrow (2025-01-22) before deciding what to pack? "
            "Please think through whether you need to use the weather tool, "
            "and if so, use it to help me decide."
        ),
    }

    result = llm_with_tools.invoke([input_message])

    assert isinstance(result, AIMessage)
    content = result.content

    response_metadata = result.response_metadata
    model_provider = response_metadata.get("model_provider", "google_genai")
    assert model_provider == "google_genai"

    if output_version == "v0":
        # v0 format:
        # Signatures are attached to function_call Parts, not thinking Parts.
        # They appear as separate function_call_signature blocks in content.
        # [{"type": "thinking", "thinking": "..."},
        #  {"type": "function_call_signature", "signature": "..."}]

        # Check for thinking blocks (should exist)
        thinking_blocks = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "thinking"
        ]
        assert thinking_blocks, "Should have thinking blocks when include_thoughts=True"

        # Check for function_call_signature blocks (not in thinking blocks)
        signature_blocks = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "function_call_signature"
        ]
        if signature_blocks:
            # Signature should be present when using function calling
            assert "signature" in signature_blocks[0]
            assert isinstance(signature_blocks[0]["signature"], str)
            assert len(signature_blocks[0]["signature"]) > 0

        # Test we can pass the result back in (with signature)
        next_message = {"role": "user", "content": "Thanks!"}
        _ = llm_with_tools.invoke([input_message, result, next_message])
    else:
        # v1 format:
        # Signatures are attached to function_call Parts, not reasoning Parts.
        # They appear as separate function_call_signature blocks in content.
        # [{"type": "reasoning", "reasoning": "..."},
        #  {"type": "function_call_signature", "signature": "..."}]
        assert isinstance(content, list)

        # Check for reasoning blocks (should exist)
        reasoning_blocks = [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "reasoning"
        ]
        assert reasoning_blocks, (
            "Should have reasoning blocks when include_thoughts=True"
        )

        # Check for function_call_signature blocks (not in reasoning blocks)
        signature_blocks = [
            block
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "function_call_signature"
        ]
        if signature_blocks:
            # Signature should be present when using function calling
            assert "signature" in signature_blocks[0]
            assert isinstance(signature_blocks[0]["signature"], str)
            assert len(signature_blocks[0]["signature"]) > 0

        # Test we can pass the result back in (with signature)
        next_message = {"role": "user", "content": "Thanks!"}
        _ = llm_with_tools.invoke([input_message, result, next_message])


def test_chat_google_genai_invoke_thinking_disabled() -> None:
    """Test invoke thinking model with a zero `thinking_budget`."""
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
    """Test invoke tokens with image without response modalities."""
    llm = ChatGoogleGenerativeAI(model=_IMAGE_OUTPUT_MODEL)

    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config={"tags": ["meow"]},
        generation_config={"top_k": 2, "top_p": 1, "temperature": 0.7},
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert not result.content.startswith(" ")
    _check_usage_metadata(result)


@pytest.mark.parametrize(
    ("test_case", "messages"),
    [
        (
            "base64_with_history",
            [
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
            ],
        ),
        (
            "url_multimodal_message",
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Guess what's in this picture! You have 3 guesses.",
                        },
                        {
                            "type": "image_url",
                            "image_url": "https://picsum.photos/seed/picsum/200/300",
                        },
                    ]
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("use_streaming", [False, True])
def test_chat_google_genai_multimodal(
    test_case: str,
    messages: list[BaseMessage],
    use_streaming: bool,
) -> None:
    """Test multimodal functionality with various message types and streaming support.

    Args:
        test_case: Descriptive name for the test case.
        messages: List of messages to send to the model.
        has_conversation_history: Whether the test includes conversation history.
        use_streaming: Whether to test streaming functionality.
    """
    del test_case  # Parameters used for test organization
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL)

    if use_streaming:
        # Test streaming
        any_chunk = False
        for chunk in llm.stream(messages):
            print(chunk)  # noqa: T201
            assert isinstance(chunk.content, str)
            if chunk.content:
                any_chunk = True
        assert any_chunk
    else:
        # Test invoke
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
    response = model.invoke([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    ("model_name", "convert_system_message_to_human"),
    [(_MODEL, True), ("models/gemini-2.5-pro", False)],
)
def test_chat_google_genai_system_message(
    model_name: str, convert_system_message_to_human: bool
) -> None:
    """Test system message handling in ChatGoogleGenerativeAI.

    Parameterized to test different models and system message conversion settings.

    Useful since I think some models (e.g. Gemini Pro) do not like system messages?
    """
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
    response = model.invoke([system_message, message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_generativeai_get_num_tokens_gemini() -> None:
    """Test model tokenizer."""
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL)
    output = llm.get_num_tokens("How are you?")
    assert output == 4


@pytest.mark.parametrize("use_streaming", [False, True])
def test_safety_settings_gemini(use_streaming: bool) -> None:
    """Test safety settings with both invoke and stream methods."""
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE  # type: ignore[dict-item]
    }
    # Test with safety filters on bind
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL).bind(
        safety_settings=safety_settings
    )

    if use_streaming:
        # Test streaming
        output_stream = llm.stream(
            "how to make a bomb?", safety_settings=safety_settings
        )
        assert isinstance(output_stream, Generator)
        streamed_messages = list(output_stream)
        assert len(streamed_messages) > 0
    else:
        # Test invoke
        output = llm.invoke("how to make a bomb?")
        assert isinstance(output, AIMessage)
        assert len(output.content) > 0


def test_chat_function_calling_with_multiple_parts() -> None:
    @tool
    def search(
        question: str,
    ) -> str:
        """Useful for when you need to answer questions or visit websites.

        You should ask targeted questions.
        """
        return "brown"

    tools = [search]

    safety: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH  # type: ignore[dict-item]
    }
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", safety_settings=safety)
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
    content_str = (
        result.content if isinstance(result.content, str) else str(result.content)
    )
    assert "brown" in content_str.lower()


def test_chat_vertexai_gemini_function_calling() -> None:
    """Test function calling with Gemini models.

    Safety settings included but not tested.
    """

    class MyModel(BaseModel):
        name: str
        age: int
        likes: list[str]

    safety: dict[HarmCategory, HarmBlockThreshold] = {
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

    model = ChatGoogleGenerativeAI(model=_MODEL, safety_settings=safety).bind_tools(
        [my_model]
    )
    response = model.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool decorator
    @tool
    def my_tool(name: str, age: int, likes: list[str]) -> None:
        """Invoke this with names and age and likes."""

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

    # Test .content_blocks property
    content_blocks = response.content_blocks
    assert isinstance(content_blocks, list)
    assert len(content_blocks) == 1
    assert isinstance(content_blocks[0], dict)
    assert content_blocks[0].get("type") == "tool_call"


@pytest.mark.parametrize(
    ("model_name", "method"),
    [
        (_MODEL, None),
        (_MODEL, "function_calling"),
        # (_MODEL, "json_mode"), testing not needed since captured by json_schema
        (_MODEL, "json_schema"),
    ],
)
def test_chat_google_genai_with_structured_output(
    model_name: str,
    method: Optional[Literal["function_calling", "json_mode", "json_schema"]],
) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH  # type: ignore[dict-item]
    }
    llm = ChatGoogleGenerativeAI(model=model_name, safety_settings=safety)
    model = llm.with_structured_output(MyModel, method=method)
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

    # This won't work with json_schema/json_mode as it expects an OpenAPI dict
    if method is None:
        model = llm.with_structured_output(
            {
                "name": "MyModel",
                "description": "MyModel",
                "parameters": MyModel.model_json_schema(),
            },
            method=method,
        )
        response = model.invoke([message])
        assert response == {
            "name": "Erick",
            "age": 27,
        }

    model = llm.with_structured_output(
        {
            "title": "MyModel",
            "description": "MyModel",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
        method=method,
    )
    response = model.invoke([message])
    assert response == {
        "name": "Erick",
        "age": 27,
    }


def test_chat_google_genai_with_structured_output_nested_model() -> None:
    """Deeply nested model test for structured output."""

    class Argument(BaseModel):
        description: str

    class Reason(BaseModel):
        strength: int
        argument: list[Argument]

    class Response(BaseModel):
        response: str
        reasons: list[Reason]

    model = ChatGoogleGenerativeAI(model=_MODEL).with_structured_output(
        Response, method="json_schema"
    )

    response = model.invoke("Why is Real Madrid better than Barcelona?")

    assert isinstance(response, Response)
    assert isinstance(response.response, str)
    assert isinstance(response.reasons, list)
    if len(response.reasons) > 0:
        reason = response.reasons[0]
        assert isinstance(reason, Reason)
        assert isinstance(reason.strength, int)
        assert isinstance(reason.argument, list)
        if len(reason.argument) > 0:
            argument = reason.argument[0]
            assert isinstance(argument, Argument)
            assert isinstance(argument.description, str)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("use_streaming", [False, True])
def test_model_methods_without_eventloop(is_async: bool, use_streaming: bool) -> None:
    """Test invoke/ainvoke and stream/astream without event loop."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    if use_streaming:
        if is_async:

            async def model_astream(context: str) -> list[BaseMessageChunk]:
                return [chunk async for chunk in model.astream(context)]

            stream_result = asyncio.run(model_astream("How can you help me?"))
        else:
            stream_result = list(model.stream("How can you help me?"))

        assert len(stream_result) > 0
        assert isinstance(stream_result[0], AIMessageChunk)
    else:
        if is_async:

            async def model_ainvoke(context: str) -> BaseMessage:
                return await model.ainvoke(context)

            invoke_result = asyncio.run(model_ainvoke("How can you help me?"))
        else:
            invoke_result = model.invoke("How can you help me?")

        assert isinstance(invoke_result, AIMessage)


def _check_web_search_output(message: AIMessage, output_version: str) -> None:
    assert "grounding_metadata" in message.response_metadata

    # Lazy parsing
    content_blocks = message.content_blocks
    text_blocks = [block for block in content_blocks if block["type"] == "text"]
    assert len(text_blocks) == 1
    text_block = text_blocks[0]
    assert text_block["annotations"]

    if output_version == "v1":
        text_blocks = [block for block in message.content if block["type"] == "text"]  # type: ignore[misc,index]
        assert len(text_blocks) == 1
        text_block = text_blocks[0]
        assert text_block["annotations"]


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_search_builtin(output_version: str) -> None:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-001", output_version=output_version
    ).bind_tools([{"google_search": {}}])
    input_message = {
        "role": "user",
        "content": "What is today's news?",
    }

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    _check_web_search_output(full, output_version)

    # Test we can process chat history without raising errors
    next_message = {
        "role": "user",
        "content": "Tell me more about that last story.",
    }
    response = llm.invoke([input_message, full, next_message])
    _check_web_search_output(response, output_version)


@pytest.mark.parametrize("use_streaming", [False, True])
def test_search_builtin_with_citations(use_streaming: bool) -> None:
    """Test that citations are properly extracted from grounding metadata."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(
        [{"google_search": {}}]
    )
    input_message = {
        "role": "user",
        "content": "Who won the 2024 UEFA Euro championship? Use search/citations.",
    }

    if use_streaming:
        full: Optional[BaseMessageChunk] = None
        for chunk in llm.stream([input_message]):
            assert isinstance(chunk, AIMessageChunk)
            full = chunk if full is None else full + chunk
        assert isinstance(full, AIMessageChunk)

        assert "grounding_metadata" in full.response_metadata
        grounding = full.response_metadata["grounding_metadata"]

        assert "grounding_chunks" in grounding or "grounding_chunks" in grounding
        assert "grounding_supports" in grounding or "grounding_supports" in grounding

        content_blocks = full.content_blocks
        text_blocks_with_citations = [
            block
            for block in content_blocks
            if block.get("type") == "text" and block.get("annotations")
        ]

        assert len(text_blocks_with_citations) > 0, (
            "Expected citations in text block if grounding metadata present"
        )

        for block in text_blocks_with_citations:
            annotations = block.get("annotations", [])
            citations = [
                ann
                for ann in annotations  # type: ignore[attr-defined]
                if annotations and ann.get("type") == "citation"
            ]

            for citation in citations:
                # Required fields
                assert citation.get("type") == "citation"
                assert "id" in citation

                # Optional but expected fields from Google AI
                if "url" in citation:
                    assert isinstance(citation["url"], str)
                    assert citation["url"].startswith("http")
                if "title" in citation:
                    assert isinstance(citation["title"], str)
                if "start_index" in citation:
                    assert isinstance(citation["start_index"], int)
                    assert citation["start_index"] >= 0
                if "end_index" in citation:
                    assert isinstance(citation["end_index"], int)
                    assert citation["end_index"] > citation.get("start_index", 0)
                if "cited_text" in citation:
                    assert isinstance(citation["cited_text"], str)
                if "extras" in citation:
                    google_metadata = citation["extras"].get("google_ai_metadata", {})
                    if google_metadata:
                        assert isinstance(google_metadata, dict)
    else:
        # Test invoke
        response = llm.invoke([input_message])
        assert isinstance(response, AIMessage)

        assert "grounding_metadata" in response.response_metadata
        grounding = response.response_metadata["grounding_metadata"]

        assert "grounding_chunks" in grounding or "grounding_chunks" in grounding
        assert "grounding_supports" in grounding or "grounding_supports" in grounding

        content_blocks = response.content_blocks
        text_blocks_with_citations = [
            block
            for block in content_blocks
            if block.get("type") == "text" and block.get("annotations")
        ]

        assert len(text_blocks_with_citations) > 0, (
            "Expected citations in text blocks if grounding metadata present"
        )

        for block in text_blocks_with_citations:
            block_annotations = block.get("annotations", [])
            citations = [
                ann
                for ann in block_annotations  # type: ignore[attr-defined]
                if ann.get("type") == "citation"
            ]

            for citation in citations:
                # Required fields
                assert citation.get("type") == "citation"
                assert "id" in citation

                # Optional but expected fields from Google AI
                if "url" in citation:
                    assert isinstance(citation["url"], str)
                    assert citation["url"].startswith("http")
                if "title" in citation:
                    assert isinstance(citation["title"], str)
                if "start_index" in citation:
                    assert isinstance(citation["start_index"], int)
                    assert citation["start_index"] >= 0
                if "end_index" in citation:
                    assert isinstance(citation["end_index"], int)
                    assert citation["end_index"] > citation.get("start_index", 0)
                if "cited_text" in citation:
                    assert isinstance(citation["cited_text"], str)
                if "extras" in citation:
                    google_metadata = citation["extras"].get("google_ai_metadata", {})
                    if google_metadata:
                        assert isinstance(google_metadata, dict)


def _check_code_execution_output(message: AIMessage, output_version: str) -> None:
    if output_version == "v0":
        blocks = [block for block in message.content if isinstance(block, dict)]
        expected_block_types = {"executable_code", "code_execution_result"}
        assert {block.get("type") for block in blocks} == expected_block_types

    else:
        # v1
        expected_block_types = {"server_tool_call", "server_tool_result", "text"}
        assert {block["type"] for block in message.content} == expected_block_types  # type: ignore[index]

    # Lazy parsing
    expected_block_types = {"server_tool_call", "server_tool_result", "text"}
    assert {block["type"] for block in message.content_blocks} == expected_block_types


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_code_execution_builtin(output_version: str) -> None:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-001", output_version=output_version
    ).bind_tools([{"code_execution": {}}])
    input_message = {
        "role": "user",
        "content": "What is 3^3?",
    }

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    _check_code_execution_output(full, output_version)

    # Test passing back in chat history without raising errors
    next_message = {
        "role": "user",
        "content": "Can you show me the calculation again with comments?",
    }
    response = llm.invoke([input_message, full, next_message])
    _check_code_execution_output(response, output_version)
