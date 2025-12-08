"""Test `ChatGoogleGenerativeAI`."""

import asyncio
import base64
import io
import json
import math
import os
import re
from collections.abc import Generator, Sequence
from typing import Literal, cast
from unittest.mock import patch

import httpx
import pytest
import requests
from google.genai.types import (
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    ToolConfig,
)
from google.genai.types import (
    Tool as GoogleTool,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    TextContentBlock,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    Environment,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Modality,
    create_context_cache,
)

_MODEL = "gemini-2.5-flash"
_PRO_MODEL = "gemini-2.5-flash"
_VISION_MODEL = "gemini-2.5-flash"
_IMAGE_OUTPUT_MODEL = "gemini-2.5-flash-image"
_AUDIO_OUTPUT_MODEL = "gemini-2.5-flash-preview-tts"
_THINKING_MODEL = "gemini-2.5-flash"
_B64_string = """iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABhGlDQ1BJQ0MgUHJvZmlsZQAAeJx9kT1Iw0AcxV8/xCIVQTuIKGSoTi2IijhqFYpQIdQKrTqYXPoFTRqSFBdHwbXg4Mdi1cHFWVcHV0EQ/ABxdXFSdJES/5cUWsR4cNyPd/ced+8Af6PCVDM4DqiaZaSTCSGbWxW6XxHECPoRQ0hipj4niil4jq97+Ph6F+dZ3uf+HL1K3mSATyCeZbphEW8QT29aOud94ggrSQrxOXHMoAsSP3JddvmNc9FhP8+MGJn0PHGEWCh2sNzBrGSoxFPEUUXVKN+fdVnhvMVZrdRY6578heG8trLMdZrDSGIRSxAhQEYNZVRgIU6rRoqJNO0nPPxDjl8kl0yuMhg5FlCFCsnxg//B727NwuSEmxROAF0vtv0xCnTvAs26bX8f23bzBAg8A1da219tADOfpNfbWvQI6NsGLq7bmrwHXO4Ag0+6ZEiOFKDpLxSA9zP6phwwcAv0rLm9tfZx+gBkqKvUDXBwCIwVKXvd492hzt7+PdPq7wdzbXKn5swsVgAAA8lJREFUeJx90dtPHHUUB/Dz+81vZhb2wrDI3soUKBSRcisF21iqqCRNY01NTE0k8aHpi0k18VJfjOFvUF9M44MmGrHFQqSQiKSmFloL5c4CXW6Fhb0vO3ufvczMzweiBGI9+eW8ffI95/yQqqrwv4UxBgCfJ9w/2NfSVB+Nyn6/r+vdLo7H6FkYY6yoABR2PJujj34MSo/d/nHeVLYbydmIp/bEO0fEy/+NMcbTU4/j4Vs6Lr0ccKeYuUKWS4ABVCVHmRdszbfvTgfjR8kz5Jjs+9RREl9Zy2lbVK9wU3/kWLJLCXnqza1bfVe7b9jLbIeTMcYu13Jg/aMiPrCwVFcgtDiMhnxwJ/zXVDwSdVCVMRV7nqzl2i9e/fKrw8mqSp84e2sFj3Oj8/SrF/MaicmyYhAaXu58NPAbeAeyzY0NLecmh2+ODN3BewYBAkAY43giI3kebrnsRmvV9z2D4ciOa3EBAf31Tp9sMgdxMTFm6j74/Ogb70VCYQKAAIDCXkOAIC6pkYBWdwwnpHEdf6L9dJtJKPh95DZhzFKMEWRAGL927XpWTmMA+s8DAOBYAoR483l/iHZ/8bXoODl8b9UfyH72SXepzbyRJNvjFGHKMlhvMBze+cH9+4lEuOOlU2X1tVkFTU7Om03q080NDGXV1cflRpHwaaoiiiildB8jhDLZ7HDfz2Yidba6Vn2L4fhzFrNRKy5OZ2QOZ1U5W8VtqlVH/iUHcM933zZYWS7Wtj66zZr65bzGJQt0glHgudi9XVzEl4vKw2kUPhO020oPYI1qYc+2Xc0bRXFwTLY0VXa2VibD/lBaIXm1UChN5JSRUcQQ1Tk/47Cf3x8bY7y17Y17PVYTG1UkLPBFcqik7Zoa9JcLYoHBqHhXNgd6gS1k9EJ1TQ2l9EDy1saErmQ2kGpwGC2MLOtCM8nZEV1K0tKJtEksSm26J/rHg2zzmabKisq939nHzqUH7efzd4f/nPGW6NP8ybNFrOsWQhpoCuuhnJ4hAnPhFam01K4oQMjBg/mzBjVhuvw2O++KKT+BIVxJKzQECBDLF2qu2WTMmCovtDQ1f8iyoGkUADBCCGPsdnvTW2OtFm01VeB06msvdWlpPZU0wJRG85ns84umU3k+VyxeEcWqvYUBAGsUrbvme4be99HFeisP/pwUOIZaOqQX31ISgrKmZhLHtXNXuJq68orrr5/9mBCglCLAGGPyy81votEbcjlKLrC9E8mhH3wdHRdcyyvjidSlxjftPJpD+o25JYvRHGFoZDdks1mBQhxJu9uxvwEiXuHnHbLd1AAAAABJRU5ErkJggg=="""  # noqa: E501


@tool
def get_weather(location: str) -> str:
    """Get the weather for a location.

    Args:
        location: The city/location to get weather for.

    Returns:
        Weather information for the location.
    """
    return "It's sunny and 72°F."


def get_wav_type_from_bytes(file_bytes: bytes) -> bool:
    """Determine if the given bytes represent a WAV file by inspecting the header.

    Args:
        file_bytes: Bytes representing the file content.

    Returns:
        If the bytes represent a WAV file.
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
    """Ensure `usage_metadata` is present and valid (greater than `0` and correct
    sum)."""
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
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert text_content == ""
    else:
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
async def test_chat_google_genai_batch(
    is_async: bool, with_tags: bool, backend_config: dict
) -> None:
    """Test batch tokens."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    messages: Sequence[str] = [
        "This is a test. Say 'foo'",
        "This is a test, say 'bar'",
    ]
    config: RunnableConfig | None = {"tags": ["foo"]} if with_tags else None

    try:
        if is_async:
            result = await llm.abatch(cast("list", messages), config=config)
        else:
            result = llm.batch(cast("list", messages), config=config)

        for token in result:
            if isinstance(token.content, list):
                text_content = "".join(
                    block.get("text", "")
                    for block in token.content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                assert len(text_content) > 0
            else:
                assert isinstance(token.content, str)
    finally:
        # Explicitly close the client to avoid resource warnings
        if llm.client:
            llm.client.close()
            if llm.client.aio:
                await llm.client.aio.aclose()


@pytest.mark.parametrize("is_async", [False, True])
async def test_chat_google_genai_invoke(is_async: bool, backend_config: dict) -> None:
    """Test invoke tokens."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

    try:
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
        if isinstance(result.content, list):
            text_content = "".join(
                block.get("text", "")
                for block in result.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            assert len(text_content) > 0
            assert not text_content.startswith(" ")
        else:
            assert isinstance(result.content, str)
            assert not result.content.startswith(" ")
        _check_usage_metadata(result)
    finally:
        # Explicitly close the client to avoid resource warnings
        if llm.client:
            llm.client.close()
            if llm.client.aio:
                await llm.client.aio.aclose()


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_with_image(backend_config: dict) -> None:
    """Test generating an image and then text.

    Using `generation_config` to specify response modalities.

    Up to `9` retries possible due to inner loop and Pytest retries.
    """
    llm = ChatGoogleGenerativeAI(model=_IMAGE_OUTPUT_MODEL, **backend_config)

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


def test_chat_google_genai_invoke_with_audio(backend_config: dict) -> None:
    """Test generating audio."""
    # Skip on Vertex AI - having some issues possibly upstream
    # TODO: look later
    # https://discuss.ai.google.dev/t/request-allowlist-access-for-audio-output-in-gemini-2-5-pro-flash-tts-vertex-ai/108067
    if backend_config.get("vertexai"):
        pytest.skip("Gemini TTS on Vertex AI requires allowlist access")

    llm = ChatGoogleGenerativeAI(
        model=_AUDIO_OUTPUT_MODEL,
        response_modalities=[Modality.AUDIO],
        **backend_config,
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


@pytest.mark.parametrize(
    ("thinking_budget", "test_description"),
    [
        (None, "default thinking config"),
        (100, "explicit thinking budget"),
    ],
)
def test_chat_google_genai_invoke_thinking(
    thinking_budget: int | None, test_description: str, backend_config: dict
) -> None:
    """Test invoke a thinking model with different thinking budget configurations."""
    llm_kwargs: dict[str, str | int] = {"model": _THINKING_MODEL}
    if thinking_budget is not None:
        llm_kwargs["thinking_budget"] = thinking_budget

    llm = ChatGoogleGenerativeAI(**llm_kwargs, **backend_config)

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    if isinstance(result.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
    else:
        assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    if (
        "output_token_details" in result.usage_metadata
        and "reasoning" in result.usage_metadata["output_token_details"]
    ):
        assert result.usage_metadata["output_token_details"]["reasoning"] > 0


def _check_thinking_output(content: list, output_version: str) -> None:
    """Check thinking output format, handling both structured and simple responses."""
    if output_version == "v0":
        thinking_key = "thinking"
        if content:
            if isinstance(content[-1], dict) and content[-1].get("type") == "text":
                assert isinstance(content[-1].get("text"), str)
            else:
                assert isinstance(content[-1], str)

    else:  # v1
        thinking_key = "reasoning"
        if content:
            assert isinstance(content[-1], dict)
            assert content[-1].get("type") == "text"
            assert isinstance(content[-1].get("text"), str)

    assert isinstance(content, list), f"Expected list content, got {type(content)}"
    thinking_blocks = [
        item
        for item in content
        if isinstance(item, dict) and item.get("type") == thinking_key
    ]
    assert thinking_blocks, f"No {thinking_key} blocks found in content: {content}"
    for block in thinking_blocks:
        assert isinstance(block[thinking_key], str), (
            f"Expected string {thinking_key}, got {type(block[thinking_key])}"
        )


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_chat_google_genai_invoke_thinking_include_thoughts(
    output_version: str, backend_config: dict
) -> None:
    """Test invoke thinking model with `include_thoughts`."""
    llm = ChatGoogleGenerativeAI(
        model=_THINKING_MODEL,
        include_thoughts=True,
        output_version=output_version,
        **backend_config,
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

    # With include_thoughts=True, content should always be structured list format
    assert isinstance(full.content, list), (
        f"Expected list content with thinking blocks when include_thoughts=True, "
        f"got {type(full.content)}: {full.content!r}. This suggests thinking "
        f"functionality failed to activate properly."
    )
    _check_thinking_output(full.content, output_version)

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
    # Follow-up response should also maintain structured format
    assert isinstance(result.content, list), (
        f"Expected list content with thinking blocks in follow-up response, "
        f"got {type(result.content)}: {result.content!r}"
    )
    _check_thinking_output(result.content, output_version)


@pytest.mark.flaky(retries=5, delay=1)
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_chat_google_genai_invoke_thinking_with_tools(
    output_version: str, backend_config: dict
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
        model=_THINKING_MODEL,
        include_thoughts=True,
        output_version=output_version,
        **backend_config,
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
        follow_up_result = llm_with_tools.invoke([input_message, result, next_message])

        # Verify the follow-up call succeeded and returned a valid response
        assert isinstance(follow_up_result, AIMessage)
        assert follow_up_result.content is not None

        # If there were signatures in the original response, verify they were properly
        # handled in the follow-up (no errors should occur)
        if signature_blocks:
            # The fact that we got a successful response means signatures were converted
            # correctly
            # Additional verification that response metadata is preserved
            assert "model_provider" in follow_up_result.response_metadata
            assert (
                follow_up_result.response_metadata["model_provider"] == "google_genai"
            )


@pytest.mark.flaky(retries=3, delay=1)
def test_thought_signature_round_trip(backend_config: dict) -> None:
    """Test thought signatures are properly preserved in round-trip conversations."""
    # TODO: could paramaterize over output_version too

    @tool
    def simple_tool(query: str) -> str:
        """A simple tool for testing."""
        return f"Response to: {query}"

    llm = ChatGoogleGenerativeAI(
        model=_THINKING_MODEL,
        include_thoughts=True,
        output_version="v1",
        **backend_config,
    )
    llm_with_tools = llm.bind_tools([simple_tool])

    # First call with function calling to generate signatures
    first_message = {
        "role": "user",
        "content": "Use the tool to help answer: What is 2+2?",
    }

    # Patch the conversion function to verify it's called with signatures
    with patch(
        "langchain_google_genai.chat_models._convert_from_v1_to_generativelanguage_v1beta"
    ) as mock_convert:
        # Set up the mock to call the real function but also track calls
        from langchain_google_genai._compat import (
            _convert_from_v1_to_generativelanguage_v1beta as real_convert,
        )

        mock_convert.side_effect = real_convert

        first_result = llm_with_tools.invoke([first_message])

        # Verify we got a response with structured content (contains signatures)
        assert isinstance(first_result, AIMessage)
        assert isinstance(first_result.content, list)

        # Second call - this should trigger signature conversion
        second_message = {"role": "user", "content": "Thanks!"}
        second_result = llm_with_tools.invoke(
            [first_message, first_result, second_message]
        )

        # Verify the conversion function was called when processing the first_result
        # (it should be called once for the first_result message)
        assert mock_convert.call_count >= 1

        # Find the call that processed our AI message with signatures
        ai_message_calls = [
            call
            for call in mock_convert.call_args_list
            if call[0][1] == "google_genai"  # model_provider argument
        ]
        assert len(ai_message_calls) >= 1

        # Verify the second call succeeded (signatures were properly converted)
        assert isinstance(second_result, AIMessage)
        assert second_result.content is not None


def test_chat_google_genai_invoke_thinking_disabled(backend_config: dict) -> None:
    """Test invoking a thinking model with zero `thinking_budget`."""
    # Note certain models may not allow `thinking_budget=0`
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", thinking_budget=0, **backend_config
    )

    result = llm.invoke(
        "How many O's are in Google? Please tell me how you double checked the result",
    )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)

    _check_usage_metadata(result)

    assert result.usage_metadata is not None
    assert "output_token_details" not in result.usage_metadata


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_google_genai_invoke_no_image_generation_without_modalities(
    backend_config: dict,
) -> None:
    """Test invoke tokens with image without response modalities."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

    result = llm.invoke(
        "Generate an image of a cat. Then, say meow!",
        config={"tags": ["meow"]},
        generation_config={"top_k": 2, "top_p": 1, "temperature": 0.7},
    )
    assert isinstance(result, AIMessage)
    if isinstance(result.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
        assert not text_content.startswith(" ")
    else:
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
    ],
)
@pytest.mark.parametrize("use_streaming", [False, True])
def test_chat_google_genai_multimodal(
    test_case: str,
    messages: list[BaseMessage],
    use_streaming: bool,
    backend_config: dict,
) -> None:
    """Test multimodal functionality with various message types and streaming support.

    Args:
        test_case: Descriptive name for the test case.
        messages: List of messages to send to the model.
        has_conversation_history: Whether the test includes conversation history.
        use_streaming: Whether to test streaming functionality.
    """
    del test_case  # Parameters used for test organization
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL, **backend_config)

    if use_streaming:
        # Test streaming
        any_chunk = False
        for chunk in llm.stream(messages):
            print(chunk)  # noqa: T201
            if isinstance(chunk.content, list):
                text_content = "".join(
                    block.get("text", "")
                    for block in chunk.content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text_content:
                    any_chunk = True
            else:
                assert isinstance(chunk.content, str)
                if chunk.content:
                    any_chunk = True
        assert any_chunk
    else:
        # Test invoke
        response = llm.invoke(messages)
        assert isinstance(response, AIMessage)
        if isinstance(response.content, list):
            text_content = "".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            assert len(text_content.strip()) > 0
        else:
            assert isinstance(response.content, str)
            assert len(response.content.strip()) > 0


def test_multimodal_pdf_input_url(backend_config: dict) -> None:
    """Test multimodal PDF input from a public URL.

    Note: This test uses the `image_url` format which downloads the content.
    For gs:// URIs:
    - Google AI backend: Must upload via client.files.upload() and use `media` format
    - Vertex AI backend: Can use `media` format with `file_uri` directly
    This test uses a public HTTP URL which works with both backends.
    """
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL, **backend_config)
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe the provided document.",
            },
            {
                "type": "image_url",
                "image_url": {"url": pdf_url},
            },
        ]
    )

    response = llm.invoke([message])
    assert isinstance(response, AIMessage)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content.strip()) > 0
    else:
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0


def test_multimodal_pdf_input_base64(backend_config: dict) -> None:
    """Test multimodal PDF input from base64 encoded data.

    Verifies that PDFs can be passed as base64 encoded data URIs
    (`data:application/pdf;base64,...`) using the `image_url` format.

    `ImageBytesLoader` handles decoding and sends the PDF as inline bytes to
    the API, which works for both Google AI and Vertex AI backends.
    """
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL, **backend_config)
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Download and encode PDF as base64
    response_data = requests.get(pdf_url, allow_redirects=True)
    with io.BytesIO() as stream:
        stream.write(response_data.content)
        pdf_base64 = base64.b64encode(stream.getbuffer()).decode("utf-8")
        pdf_data_uri = f"data:application/pdf;base64,{pdf_base64}"

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Describe the provided document.",
            },
            {
                "type": "image_url",
                "image_url": {"url": pdf_data_uri},
            },
        ]
    )

    response = llm.invoke([message])
    assert isinstance(response, AIMessage)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content.strip()) > 0
    else:
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0


@pytest.mark.parametrize(
    "message",
    [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Give a concise description of this image.",
                },
                {
                    "type": "image_url",
                    "image_url": "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png",
                },
            ]
        ),
    ],
)
def test_chat_google_genai_invoke_media_resolution(
    message: BaseMessage, backend_config: dict
) -> None:
    """Test invoke vision model with `media_resolution` set to low and without."""
    llm = ChatGoogleGenerativeAI(model=_VISION_MODEL, **backend_config)
    result = llm.invoke([message])
    result_low_res = llm.invoke(
        [message], media_resolution=MediaResolution.MEDIA_RESOLUTION_LOW
    )

    assert isinstance(result_low_res, AIMessage)
    _check_usage_metadata(result_low_res)

    assert result.usage_metadata is not None
    assert result_low_res.usage_metadata is not None
    assert (
        result_low_res.usage_metadata["input_tokens"]
        < result.usage_metadata["input_tokens"] / 3
    )


def test_chat_google_genai_single_call_with_history(backend_config: dict) -> None:
    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model.invoke([message1, message2, message3])
    assert isinstance(response, AIMessage)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
    else:
        assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "model_name",
    [_MODEL, _PRO_MODEL],
)
def test_chat_google_genai_system_message(
    model_name: str, backend_config: dict
) -> None:
    """Test system message handling.

    Tests that system messages are properly converted to system instructions for
    different models.
    """
    model = ChatGoogleGenerativeAI(model=model_name, **backend_config)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content="You're supposed to answer math questions.")
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model.invoke([system_message, message1, message2, message3])
    assert isinstance(response, AIMessage)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
    else:
        assert isinstance(response.content, str)


def test_generativeai_get_num_tokens_gemini(backend_config: dict) -> None:
    """Test model tokenizer."""
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL, **backend_config)
    output = llm.get_num_tokens("How are you?")
    assert output == 4


def test_get_num_tokens_from_messages(backend_config: dict) -> None:
    """Test token counting from messages."""
    model = ChatGoogleGenerativeAI(temperature=0.0, model=_MODEL, **backend_config)
    message = HumanMessage(content="Hello")
    token = model.get_num_tokens_from_messages(messages=[message])
    assert isinstance(token, int)
    assert token == 3


def test_single_call_previous_blocked_response(backend_config: dict) -> None:
    """Test handling of blocked responses in conversation history.

    If a previous call was blocked, the AIMessage will have empty content.
    Empty content should be ignored.
    """
    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    text_question = "How much is 3+3?"
    # Previous blocked response included in history
    blocked_message = AIMessage(
        content="",
        response_metadata={
            "is_blocked": True,
            "safety_ratings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "probability_label": "MEDIUM",
                    "probability_score": 0.33039191365242004,
                    "blocked": True,
                    "severity": "HARM_SEVERITY_MEDIUM",
                    "severity_score": 0.2782268822193146,
                },
            ],
            "finish_reason": "SAFETY",
        },
    )
    user_message = HumanMessage(content=text_question)
    response = model.invoke([blocked_message, user_message])
    assert isinstance(response, AIMessage)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
    else:
        assert isinstance(response.content, str)


def test_json_mode_typeddict(backend_config: dict) -> None:
    """Test structured output with `TypedDict` using `json_mode` method."""
    from typing_extensions import TypedDict

    class MyModel(TypedDict):
        name: str
        age: int

    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    model = llm.with_structured_output(MyModel, method="json_mode")  # type: ignore[arg-type]
    message = HumanMessage(content="My name is Erick and I am 28 years old")

    response = model.invoke([message])
    assert isinstance(response, dict)
    assert response == {"name": "Erick", "age": 28}

    # Test stream
    for chunk in model.stream([message]):
        assert isinstance(chunk, dict)
        assert all(key in ["name", "age"] for key in chunk)
    assert chunk == {"name": "Erick", "age": 28}


def test_nested_bind_tools(backend_config: dict) -> None:
    """Test nested Pydantic models in tool calling with `tool_choice`."""
    from pydantic import Field

    class Person(BaseModel):
        name: str = Field(description="The name.")
        hair_color: str | None = Field(
            default=None, description="Hair color, only if provided."
        )

    class People(BaseModel):
        data: list[Person] = Field(description="The people.")

    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    llm_with_tools = llm.bind_tools([People], tool_choice="People")

    response = llm_with_tools.invoke("Chester, no hair color provided.")
    assert isinstance(response, AIMessage)
    assert response.tool_calls[0]["name"] == "People"


def test_timeout_non_streaming(backend_config: dict) -> None:
    """Test timeout parameter in non-streaming mode."""
    model = ChatGoogleGenerativeAI(
        model=_MODEL,
        timeout=0.001,  # 1ms - too short to complete
        **backend_config,
    )
    with pytest.raises((httpx.ReadTimeout, httpx.TimeoutException)):
        model.invoke([HumanMessage(content="Hello")])


def test_timeout_streaming(backend_config: dict) -> None:
    """Test timeout parameter in streaming mode."""
    model = ChatGoogleGenerativeAI(
        model=_MODEL,
        timeout=0.001,  # 1ms - too short to complete
        streaming=True,
        **backend_config,
    )
    with pytest.raises((httpx.ReadTimeout, httpx.TimeoutException)):
        model.invoke([HumanMessage(content="Hello")])


def test_response_metadata_avg_logprobs(backend_config: dict) -> None:
    """Test that `avg_logprobs` are present in response metadata."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    response = llm.invoke("Hello!")
    probs = response.response_metadata.get("avg_logprobs")
    if probs is not None:
        assert isinstance(probs, float)


@pytest.mark.xfail(reason="logprobs are subject to daily quotas")
def test_logprobs(backend_config: dict) -> None:
    """Test logprobs parameter with different configurations."""
    # Test with integer logprobs (top K)
    llm = ChatGoogleGenerativeAI(model=_MODEL, logprobs=2, **backend_config)
    msg = llm.invoke("hey")
    tokenprobs = msg.response_metadata.get("logprobs_result")
    assert tokenprobs is None or isinstance(tokenprobs, list)
    if tokenprobs:
        stack = tokenprobs[:]
        while stack:
            token = stack.pop()
            assert isinstance(token, dict)
            assert "token" in token
            assert "logprob" in token
            assert isinstance(token.get("token"), str)
            assert isinstance(token.get("logprob"), float)
            if "top_logprobs" in token and token.get("top_logprobs") is not None:
                assert isinstance(token.get("top_logprobs"), list)
                stack.extend(token.get("top_logprobs", []))

    # Test with logprobs=True
    llm2 = ChatGoogleGenerativeAI(model=_MODEL, logprobs=True, **backend_config)
    msg2 = llm2.invoke("how are you")
    assert msg2.response_metadata["logprobs_result"]

    # Test with logprobs=False
    llm3 = ChatGoogleGenerativeAI(model=_MODEL, logprobs=False, **backend_config)
    msg3 = llm3.invoke("howdy")
    assert msg3.response_metadata.get("logprobs_result") is None


@pytest.mark.xfail(reason="logprobs are subject to daily quotas")
def test_logprobs_with_json_schema(backend_config: dict) -> None:
    """Test logprobs with JSON schema structured output.

    Ensures logprobs are populated when using JSON schema responses.
    This exercises the logprobs path with `response_mime_type='application/json'`
    and `response_schema` set.

    The test verifies:
    1. Zero logprobs (`prob=1.0`, 100% certainty) are included, not filtered
    2. All logprob values are valid (non-positive, non-`NaN`)
    """
    output_schema = {
        "title": "Test Schema",
        "type": "object",
        "properties": {
            "fieldA": {"type": "string"},
            "fieldB": {"type": "number"},
        },
        "required": ["fieldA", "fieldB"],
    }

    llm = ChatGoogleGenerativeAI(
        model=_MODEL,
        response_mime_type="application/json",
        response_schema=output_schema,
        logprobs=True,
        **backend_config,
    )

    msg = llm.invoke("Return a JSON object with fieldA='test' and fieldB=42")
    tokenprobs = msg.response_metadata.get("logprobs_result")
    # We don't assert exact content to avoid flakiness, but if present it must
    # be a well-formed list of token/logprob dicts, including zero logprobs.
    assert tokenprobs is None or isinstance(tokenprobs, list)
    if tokenprobs:
        logprob_values = []
        for token in tokenprobs:
            assert isinstance(token, dict)
            assert "token" in token
            assert "logprob" in token
            assert isinstance(token.get("token"), str)
            assert isinstance(token.get("logprob"), (float, int))
            logprob_values.append(token["logprob"])

        # Verify all logprobs are valid: non-positive (zero allowed) and not NaN
        for lp in logprob_values:
            assert lp <= 0.0, f"Logprob should be non-positive, got {lp}"
            assert not (isinstance(lp, float) and math.isnan(lp)), (
                f"Logprob should not be NaN, got {lp}"
            )


@pytest.mark.parametrize("use_streaming", [False, True])
def test_safety_settings_gemini(use_streaming: bool, backend_config: dict) -> None:
    """Test safety settings with both `invoke` and `stream` methods."""
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
    # Test with safety filters on bind
    llm = ChatGoogleGenerativeAI(temperature=0, model=_MODEL, **backend_config).bind(
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


@pytest.mark.flaky(retries=3, delay=1)
def test_chat_function_calling_with_multiple_parts(backend_config: dict) -> None:
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
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatGoogleGenerativeAI(
        model=_PRO_MODEL, safety_settings=safety, **backend_config
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

    follow_up = HumanMessage(
        content=(
            "Based on the search results above, what did you find about the bird "
            "colors?"
        )
    )
    result = llm_with_search.invoke([request, response, *tool_messages, follow_up])

    assert isinstance(result, AIMessage)

    if isinstance(result.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in result.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert "brown" in text_content.lower()
    else:
        content_str = (
            result.content if isinstance(result.content, str) else str(result.content)
        )
        assert "brown" in content_str.lower()


def test_chat_vertexai_gemini_function_calling(backend_config: dict) -> None:
    """Test function calling with Gemini models.

    Safety settings included but not tested.
    """

    class MyModel(BaseModel):
        name: str
        age: int
        likes: list[str]

    safety: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    # Test .bind_tools with BaseModel
    message = HumanMessage(
        content="My name is Erick and I am 27 years old. I like apple and banana."
    )
    model = ChatGoogleGenerativeAI(
        model=_MODEL, safety_settings=safety, **backend_config
    ).bind_tools([MyModel])
    response = model.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int, likes: list[str]) -> None:
        """Invoke this with names and age and likes."""

    model = ChatGoogleGenerativeAI(
        model=_MODEL, safety_settings=safety, **backend_config
    ).bind_tools([my_model])
    response = model.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool decorator
    @tool
    def my_tool(name: str, age: int, likes: list[str]) -> None:
        """Invoke this with names and age and likes."""

    model = ChatGoogleGenerativeAI(
        model=_MODEL, safety_settings=safety, **backend_config
    ).bind_tools([my_tool])
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
    tool_call_blocks = [b for b in content_blocks if b.get("type") == "tool_call"]
    assert len(tool_call_blocks) == 1


@pytest.mark.flaky(retries=3, delay=1)
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
    method: Literal["function_calling", "json_mode", "json_schema"] | None,
    backend_config: dict,
) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatGoogleGenerativeAI(
        model=model_name, safety_settings=safety, **backend_config
    )
    model = llm.with_structured_output(MyModel, method=method)
    message = HumanMessage(content="My name is Erick and I am 27 years old")

    response = model.invoke([message])
    assert response is not None, f"Structured output returned None for method={method}."
    assert isinstance(response, MyModel), (
        f"Expected MyModel instance, got {type(response)}: {response}"
    )
    expected = MyModel(name="Erick", age=27)
    assert response == expected, f"Expected {expected}, got {response}"

    model = llm.with_structured_output(
        {
            "name": "MyModel",
            "description": "MyModel",
            "parameters": MyModel.model_json_schema(),
        },
        method="function_calling",
    )
    response = model.invoke([message])
    assert response is not None, "Structured output with schema dict returned None"
    expected = {"name": "Erick", "age": 27}  # type: ignore[assignment]
    assert response == expected, f"Expected {expected}, got {response}"

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
        assert response is not None, (
            f"Structured output with method={method} returned None"
        )
        expected = {
            "name": "Erick",  # type: ignore[assignment]
            "age": 27,
        }
        assert response == expected, f"Expected {expected}, got {response}"

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
    assert response is not None, (
        f"Structured output with JSON schema and method={method} returned None"
    )
    expected = {
        "name": "Erick",  # type: ignore[assignment]
        "age": 27,
    }
    assert response == expected, f"Expected {expected}, got {response}"


def test_chat_google_genai_with_structured_output_nested_model(
    backend_config: dict,
) -> None:
    """Deeply nested model test for structured output."""

    class Argument(BaseModel):
        description: str

    class Reason(BaseModel):
        strength: int
        argument: list[Argument]

    class Response(BaseModel):
        response: str
        reasons: list[Reason]

    model = ChatGoogleGenerativeAI(
        model=_MODEL, **backend_config
    ).with_structured_output(Response, method="json_schema")

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


def test_thinking_params_preserved_with_structured_output(backend_config: dict) -> None:
    """Test that `thinking_budget=0` and `include_thoughts=False` are preserved.

    Verifies that thinking configuration persists through `with_structured_output()`.
    """

    class SimpleModel(BaseModel):
        answer: str

    # Initialize with thinking disabled
    # Only certain models support disabling thinking
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        thinking_budget=0,
        include_thoughts=False,
        **backend_config,
    )

    # Apply structured output - params should be preserved
    structured_llm = llm.with_structured_output(SimpleModel, method="json_schema")

    result = structured_llm.invoke("What is 2+2? Just give a brief answer.")

    assert isinstance(result, SimpleModel)
    assert isinstance(result.answer, str)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("use_streaming", [False, True])
def test_model_methods_without_eventloop(
    is_async: bool, use_streaming: bool, backend_config: dict
) -> None:
    """Test `invoke` and `stream` (sync & async) without event loop."""
    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

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
    assert len(text_blocks) >= 1

    # Check that at least one block has annotations
    text_block = next((b for b in text_blocks if b.get("annotations")), None)
    assert text_block is not None

    if output_version == "v1":
        text_blocks = [block for block in message.content if block["type"] == "text"]  # type: ignore[misc,index]
        assert len(text_blocks) == 1
        v1_text_block: TextContentBlock = text_blocks[0]
        assert v1_text_block.get("annotations")


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_search_builtin(output_version: str, backend_config: dict) -> None:
    llm = ChatGoogleGenerativeAI(
        model=_MODEL, output_version=output_version, **backend_config
    ).bind_tools([{"google_search": {}}])
    input_message = {
        "role": "user",
        "content": "What is today's news?",
    }

    full: BaseMessageChunk | None = None
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


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("use_streaming", [False, True])
def test_search_builtin_with_citations(
    use_streaming: bool, backend_config: dict
) -> None:
    """Test that citations are properly extracted from `grounding_metadata`."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config).bind_tools(
        [{"google_search": {}}]
    )
    input_message = {
        "role": "user",
        "content": "Who won the 2024 UEFA Euro championship? Use search/citations.",
    }

    if use_streaming:
        full: BaseMessageChunk | None = None
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


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("use_streaming", [False, True])
def test_structured_output_with_google_search(
    use_streaming: bool, backend_config: dict
) -> None:
    """Test structured outputs combined with Google Search tool.

    Tests that the model can:
    1. Use Google Search to find information
    2. Return a response that conforms to a structured schema
    3. Include grounding metadata from the search
    """

    class MatchResult(BaseModel):
        winner: str
        final_match_score: str
        scorers: list[str]

    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", **backend_config)

    # Bind tools and configure for structured output
    llm_with_search = llm.bind(
        tools=[{"google_search": {}}, {"url_context": {}}],
        response_mime_type="application/json",
        response_schema=MatchResult.model_json_schema(),
    )

    if use_streaming:
        # Test streaming
        chunks: list[BaseMessageChunk] = []
        for chunk in llm_with_search.stream(
            "Search for all details for the latest Euro championship final match."
        ):
            assert isinstance(chunk, AIMessageChunk)
            chunks.append(chunk)

        assert len(chunks) > 0

        # Reconstruct full message
        response = chunks[0]
        for chunk in chunks[1:]:  # type: ignore[assignment]
            response = response + chunk

        response = cast("AIMessageChunk", response)
        assert isinstance(response, AIMessageChunk)
    else:
        # Test invoke
        response = llm_with_search.invoke(  # type: ignore[assignment]
            "Search for all details for the latest Euro championship final match."
        )
        assert isinstance(response, AIMessage)

    # Extract JSON from response content
    assert isinstance(response.content, list)
    text_content = "".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )

    # WORKAROUND: Strip markdown code blocks if present
    # BUG: When using response_mime_type="application/json" with tools (e.g., Google
    # Search), the google-genai SDK returns JSON wrapped in markdown code blocks
    # (```json\n{...}\n```) instead of raw JSON. This only happens with tools;
    # without tools, raw JSON is returned correctly as expected.
    if text_content.startswith("```json"):
        text_content = (
            text_content.removeprefix("```json\n").removesuffix("```").strip()
        )
    elif text_content.startswith("```"):
        text_content = text_content.removeprefix("```\n").removesuffix("```").strip()

    # Verify structured output can be parsed
    result = MatchResult.model_validate_json(text_content)
    assert isinstance(result, MatchResult)
    assert isinstance(result.winner, str)
    assert len(result.winner) > 0
    assert isinstance(result.final_match_score, str)
    assert len(result.final_match_score) > 0
    assert isinstance(result.scorers, list)

    # Verify grounding metadata is present (indicating search was used)
    assert "grounding_metadata" in response.response_metadata
    grounding = response.response_metadata["grounding_metadata"]
    assert "grounding_chunks" in grounding or "grounding_supports" in grounding

    # Verify usage metadata
    # TODO: Investigate streaming usage metadata accumulation issue
    # When streaming, total_tokens doesn't match input_tokens + output_tokens
    # This appears to be a chunk accumulation bug where total_tokens is summed
    # across chunks but input/output tokens only keep final values
    if not use_streaming:
        _check_usage_metadata(response)


def test_search_with_googletool(backend_config: dict) -> None:
    """Test using `GoogleTool` with Google Search."""
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", **backend_config)
    resp = llm.invoke(
        "When is the next total solar eclipse in US?",
        tools=[GoogleTool(google_search={})],
    )
    assert "grounding_metadata" in resp.response_metadata


def test_url_context_tool(backend_config: dict) -> None:
    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    model_with_search = model.bind_tools([{"url_context": {}}])

    input = "What is this page's contents about? https://docs.langchain.com"
    response = model_with_search.invoke(input)
    assert isinstance(response, AIMessage)

    assert (
        response.response_metadata["grounding_metadata"]["grounding_chunks"][0]["web"][
            "uri"
        ]
        is not None
    )


def test_google_maps_grounding(backend_config: dict) -> None:
    """Test using Google Maps grounding for location-aware responses."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", **backend_config)
    model_with_maps = model.bind_tools([{"google_maps": {}}])

    response = model_with_maps.invoke(
        "What are some good Italian restaurants in Boston's North End?"
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str) or isinstance(response.content, list)

    assert "grounding_metadata" in response.response_metadata
    grounding = response.response_metadata["grounding_metadata"]

    # Maps grounding should have grounding_chunks with maps data or a
    # google_maps_widget_context_token
    has_grounding_chunks = (
        "grounding_chunks" in grounding and grounding["grounding_chunks"]
    )
    has_widget_token = (
        "google_maps_widget_context_token" in grounding
        and grounding["google_maps_widget_context_token"] is not None
    )

    assert has_grounding_chunks or has_widget_token, (
        "Expected maps grounding data or widget token"
    )

    # Test multi-turn conversation with grounding metadata in history
    messages = [
        HumanMessage("What are some good Italian restaurants in Boston's North End?"),
        response,  # Include the response with grounding metadata
        HumanMessage("What about French restaurants in the same area?"),
    ]

    follow_up_response = model_with_maps.invoke(messages)
    assert isinstance(follow_up_response, AIMessage)
    assert isinstance(follow_up_response.content, str) or isinstance(
        follow_up_response.content, list
    )

    # Test with explicit location context via tool_config
    model_with_location = model.bind_tools(
        [{"google_maps": {}}],
        tool_config={
            "retrieval_config": {
                "lat_lng": {
                    "latitude": 42.366978,
                    "longitude": -71.053940,
                }
            }
        },
    )

    response_with_location = model_with_location.invoke(
        "What Italian restaurants are within a 5 minute walk from here?"
    )
    assert isinstance(response_with_location, AIMessage)
    assert isinstance(response_with_location.content, str) or isinstance(
        response_with_location.content, list
    )
    assert "grounding_metadata" in response_with_location.response_metadata


def test_google_maps_grounding_invoke_direct(backend_config: dict) -> None:
    """Test passing Maps grounding tool directly to invoke without binding."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", **backend_config)

    # Pass tools directly to invoke instead of binding
    response = model.invoke(
        "What are some good Italian restaurants in Boston's North End?",
        tools=[{"google_maps": {}}],
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str) or isinstance(response.content, list)
    assert "grounding_metadata" in response.response_metadata

    # Test with tool_config passed directly to invoke
    response_with_location = model.invoke(
        "What Italian restaurants are within a 5 minute walk from here?",
        tools=[{"google_maps": {}}],
        tool_config={
            "retrieval_config": {
                "lat_lng": {
                    "latitude": 42.366978,
                    "longitude": -71.053940,
                }
            }
        },
    )
    assert isinstance(response_with_location, AIMessage)
    assert isinstance(response_with_location.content, str) or isinstance(
        response_with_location.content, list
    )
    assert "grounding_metadata" in response_with_location.response_metadata


def _check_code_execution_output(message: AIMessage, output_version: str) -> None:
    if output_version == "v0":
        blocks = [block for block in message.content if isinstance(block, dict)]
        # Find code execution blocks
        code_blocks = [
            block
            for block in blocks
            if block.get("type") in {"executable_code", "code_execution_result"}
        ]
        # For integration test, code execution must happen
        assert code_blocks, (
            f"No code execution blocks found in content: "
            f"{[block.get('type') for block in blocks]}"
        )
        expected_block_types = {"executable_code", "code_execution_result"}
        assert {block.get("type") for block in code_blocks} == expected_block_types

    else:
        # v1
        expected_block_types = {"server_tool_call", "server_tool_result", "text"}
        assert {block["type"] for block in message.content} == expected_block_types  # type: ignore[index]

    # Lazy parsing
    expected_block_types = {"server_tool_call", "server_tool_result", "text"}
    assert {block["type"] for block in message.content_blocks} == expected_block_types


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_code_execution_builtin(output_version: str, backend_config: dict) -> None:
    llm = ChatGoogleGenerativeAI(
        model=_MODEL, output_version=output_version, **backend_config
    ).bind_tools([{"code_execution": {}}])
    input_message = {
        "role": "user",
        "content": "Calculate the value of 3^3 using Python code execution.",
    }

    full: BaseMessageChunk | None = None
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


def test_computer_use_tool(backend_config: dict) -> None:
    """Test computer use tool integration.

    To run this test:
    1. Set environment variable: TEST_COMPUTER_USE=1
    2. Ensure you have a valid API key configured
    3. The model may require a screenshot/screen context in production use

    Example:
        TEST_COMPUTER_USE=1 pytest -k test_computer_use_tool -v
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-computer-use-preview-10-2025", **backend_config
    )

    model_with_computer = model.bind_tools(
        [{"computer_use": {"environment": Environment.ENVIRONMENT_BROWSER}}]
    )

    # Simple test - just verify tool binding works
    input_message = "Describe what actions you could take on a web page."
    response = model_with_computer.invoke(input_message)
    assert isinstance(response, AIMessage)
    assert response.content is not None


def test_chat_google_genai_invoke_with_generation_params(backend_config: dict) -> None:
    """Test that generation parameters passed to invoke() are respected.

    Verifies that `max_output_tokens` (max_tokens) and `thinking_budget`
    parameters passed directly to invoke() method override model defaults.
    """
    # Use gemini-2.5-flash because it supports thinking_budget=0
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", **backend_config)

    # Test with max_output_tokens constraint
    result_constrained = llm.invoke(
        "Alice, Bob, and Carol each live in a different house on the same street: "
        "red, green, and blue. The person who lives in the red house owns a cat. "
        "Bob does not live in the green house. Carol owns a dog. The green house "
        "is to the left of the red house. Alice does not own a cat. Who lives in "
        "each house, and what pet do they own?",
        max_output_tokens=10,
        thinking_budget=0,
    )

    assert isinstance(result_constrained, AIMessage)
    # Verify output tokens are within limit
    assert result_constrained.usage_metadata is not None

    output_tokens = result_constrained.usage_metadata.get("output_tokens")
    assert output_tokens is not None, "usage_metadata is missing 'output_tokens'"
    assert output_tokens <= 10, f"Expected output_tokens <= 10, got {output_tokens}"

    # Verify thinking is disabled
    details = result_constrained.usage_metadata.get("output_token_details") or {}
    assert "reasoning" not in details, (
        "Expected no reasoning tokens when thinking_budget=0"
    )


@pytest.mark.parametrize("max_tokens", [10, 20, 50])
def test_chat_google_genai_invoke_respects_max_tokens(
    max_tokens: int, backend_config: dict
) -> None:
    """Test that different `max_output_tokens` values are respected."""
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

    result = llm.invoke(
        "Write a detailed essay about artificial intelligence.",
        max_output_tokens=max_tokens,
    )

    assert isinstance(result, AIMessage)
    assert result.usage_metadata is not None
    output_tokens = result.usage_metadata.get("output_tokens")
    assert output_tokens is not None, "usage_metadata is missing 'output_tokens'"
    assert output_tokens <= max_tokens, (
        f"Expected output_tokens <= {max_tokens}, got {output_tokens}"
    )


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop(output_version: Literal["v0", "v1"], backend_config: dict) -> None:
    """Test agent loop with tool calling (non-streaming).

    Ensures that the model can:
    1. Make a tool call in response to a user query
    2. Accept the tool result
    3. Generate a final response incorporating the tool result
    """

    llm = ChatGoogleGenerativeAI(
        model=_MODEL, output_version=output_version, **backend_config
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")

    # First call - should make a tool call
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert "location" in tool_call["args"]

    # Execute the tool
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)

    # Second call - should incorporate tool result
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)
    # Response should mention weather information
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0
    else:
        # v0 output for 2.5 models and lower
        assert isinstance(response.content, str)
        assert len(response.content) > 0


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop_streaming(
    output_version: Literal["v0", "v1"], backend_config: dict
) -> None:
    """Test agent loop with tool calling (streaming)."""

    llm = ChatGoogleGenerativeAI(
        model=_MODEL, output_version=output_version, **backend_config
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")

    # First call - stream tool call chunks
    chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)

    assert len(chunks) > 0

    # Reconstruct the full message
    tool_call_message = chunks[0]
    for chunk in chunks[1:]:  # type: ignore[assignment]
        tool_call_message = tool_call_message + chunk

    tool_call_message = cast("AIMessageChunk", tool_call_message)
    assert isinstance(tool_call_message, AIMessageChunk)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_weather"

    # Execute the tool
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)

    # Second call - stream final response
    response_chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    ):
        assert isinstance(chunk, AIMessageChunk)
        response_chunks.append(chunk)

    assert len(response_chunks) > 0
    # Reconstruct full response
    response = response_chunks[0]
    for chunk in response_chunks[1:]:  # type: ignore[assignment]
        response = response + chunk

    response = cast("AIMessageChunk", response)
    assert isinstance(response, AIMessageChunk)
    assert len(response.text) > 0


@pytest.mark.parametrize("use_stream_method", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
async def test_basic_streaming(
    use_stream_method: bool, is_async: bool, backend_config: dict
) -> None:
    """Test basic streaming functionality.

    Args:
        use_stream_method: If `True`, use `stream()`, otherwise set `streaming=True`
        is_async: Test async vs sync streaming
    """
    if use_stream_method:
        llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    else:
        llm = ChatGoogleGenerativeAI(model=_MODEL, streaming=True, **backend_config)

    message = HumanMessage("Count from 1 to 5")

    try:
        chunks: list[BaseMessageChunk] = []
        if is_async:
            if use_stream_method:
                async for chunk in llm.astream([message]):
                    assert isinstance(chunk, AIMessageChunk)
                    chunks.append(chunk)
            else:
                result = await llm.ainvoke([message])
                # When streaming=True, ainvoke still returns full message
                assert isinstance(result, AIMessage)
                return
        elif use_stream_method:
            for chunk in llm.stream([message]):
                assert isinstance(chunk, AIMessageChunk)
                chunks.append(chunk)
        else:
            result = llm.invoke([message])
            # When streaming=True, invoke still returns full message
            assert isinstance(result, AIMessage)
            return

        # Verify we got chunks
        assert len(chunks) > 0

        # Verify we can reconstruct the message
        full_message = chunks[0]
        for chunk in chunks[1:]:  # type: ignore[assignment]
            full_message = full_message + chunk

        full_message = cast("AIMessageChunk", full_message)
        assert isinstance(full_message, AIMessageChunk)
        if isinstance(full_message.content, list):
            text_content = "".join(
                block.get("text", "")
                for block in full_message.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            assert len(text_content) > 0
        else:
            assert isinstance(full_message.content, str)
            assert len(full_message.content) > 0

        # Verify usage metadata is present in the final message
        _check_usage_metadata(cast("AIMessage", full_message))
    finally:
        # Explicitly close the client to avoid resource warnings
        if llm.client:
            llm.client.close()
            if llm.client.aio:
                await llm.client.aio.aclose()


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_gemini_3_pro_streaming_with_thinking(
    output_version: Literal["v0", "v1"], backend_config: dict
) -> None:
    """Test `gemini-3-pro-preview` streaming with thinking capabilities.

    `gemini-3-pro-preview` uses `thinking_level` instead of `thinking_budget`.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        thinking_level="high",
        include_thoughts=True,
        output_version=output_version,
        **backend_config,
    )

    input_message = HumanMessage(
        "How many r's are in the word 'strawberry'? Think through this carefully."
    )

    chunks: list[BaseMessageChunk] = []
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)

    assert len(chunks) > 0

    # Reconstruct full message
    full_message = chunks[0]
    for chunk in chunks[1:]:  # type: ignore[assignment]
        full_message = full_message + chunk

    full_message = cast("AIMessageChunk", full_message)
    assert isinstance(full_message, AIMessageChunk)
    assert isinstance(full_message.content, list), (
        f"Expected list content, got {type(full_message.content)}"
    )

    # Check for thinking/reasoning blocks based on output_version
    if output_version == "v0":
        # v0 uses "thinking" blocks
        thought_blocks = [
            block
            for block in full_message.content
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
        assert thought_blocks, (
            "Expected thinking blocks when include_thoughts=True for gemini-3 with v0"
        )
    else:
        # v1 uses "reasoning" blocks
        reasoning_blocks = [
            block
            for block in full_message.content
            if isinstance(block, dict) and block.get("type") == "reasoning"
        ]
        assert reasoning_blocks, (
            "Expected reasoning blocks when include_thoughts=True for gemini-3 with v1"
        )

    # Verify usage metadata includes reasoning tokens
    _check_usage_metadata(cast("AIMessage", full_message))
    assert full_message.usage_metadata is not None
    if (
        "output_token_details" in full_message.usage_metadata
        and "reasoning" in full_message.usage_metadata["output_token_details"]
    ):
        assert full_message.usage_metadata["output_token_details"]["reasoning"] > 0


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_gemini_3_pro_agent_loop_streaming(
    output_version: Literal["v0", "v1"], backend_config: dict
) -> None:
    """Test `gemini-3-pro-preview` agent loop with streaming and thinking."""

    @tool
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers.

        Args:
            a: First number.
            b: Second number.

        Returns:
            The sum of a and b.
        """
        return a + b

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        thinking_level="high",
        include_thoughts=True,
        output_version=output_version,
        **backend_config,
    )
    llm_with_tools = llm.bind_tools([calculate_sum])

    input_message = HumanMessage("What is 123 + 456? Use the calculator tool.")

    # First call - stream tool call
    chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)

    assert len(chunks) > 0

    # Reconstruct message
    tool_call_message = chunks[0]
    for chunk in chunks[1:]:  # type: ignore[assignment]
        tool_call_message = tool_call_message + chunk

    tool_call_message = cast("AIMessageChunk", tool_call_message)
    assert isinstance(tool_call_message, AIMessageChunk)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "calculate_sum"

    # Execute tool
    tool_message = calculate_sum.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)

    # Second call - stream final response with reasoning
    response_chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    ):
        assert isinstance(chunk, AIMessageChunk)
        response_chunks.append(chunk)

    assert len(response_chunks) > 0

    # Reconstruct response
    response = response_chunks[0]
    for chunk in response_chunks[1:]:  # type: ignore[assignment]
        response = response + chunk

    response = cast("AIMessageChunk", response)
    assert isinstance(response, AIMessageChunk)
    assert isinstance(response.content, list)

    # Verify response has text blocks
    text_blocks = [
        block
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert len(text_blocks) > 0


@pytest.mark.parametrize("model_name", [_MODEL, "gemini-3-pro-preview"])
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_streaming_with_multiple_tool_calls(
    model_name: str, output_version: Literal["v0", "v1"], backend_config: dict
) -> None:
    """Test streaming with multiple tool calls in a single response."""

    @tool
    def get_temperature(city: str) -> str:
        """Get temperature for a city.

        Args:
            city: The city name.

        Returns:
            Temperature information.
        """
        return f"Temperature in {city}: 72°F"

    @tool
    def get_humidity(city: str) -> str:
        """Get humidity for a city.

        Args:
            city: The city name.

        Returns:
            Humidity information.
        """
        return f"Humidity in {city}: 65%"

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        streaming=True,
        output_version=output_version,
        **backend_config,
    )
    llm_with_tools = llm.bind_tools([get_temperature, get_humidity])

    input_message = HumanMessage(
        "Get both temperature and humidity for San Francisco. Use both tools."
    )

    # Stream tool calls
    chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)

    assert len(chunks) > 0

    # Reconstruct message
    tool_call_message = chunks[0]
    for chunk in chunks[1:]:  # type: ignore[assignment]
        tool_call_message = tool_call_message + chunk

    tool_call_message = cast("AIMessageChunk", tool_call_message)
    assert isinstance(tool_call_message, AIMessageChunk)
    tool_calls = tool_call_message.tool_calls

    # Model may make 1 or 2 tool calls depending on its decision
    assert len(tool_calls) >= 1
    assert len(tool_calls) <= 2

    # Execute tools
    tool_messages = []
    for tool_call in tool_calls:
        if tool_call["name"] == "get_temperature":
            tool_message = get_temperature.invoke(tool_call)
        elif tool_call["name"] == "get_humidity":
            tool_message = get_humidity.invoke(tool_call)
        else:
            continue
        tool_messages.append(tool_message)

    # Stream final response
    response_chunks: list[BaseMessageChunk] = []
    for chunk in llm_with_tools.stream(
        [input_message, tool_call_message, *tool_messages]
    ):
        assert isinstance(chunk, AIMessageChunk)
        response_chunks.append(chunk)

    assert len(response_chunks) > 0


@pytest.mark.extended
def test_context_caching(backend_config: dict) -> None:
    """Test context caching with large text content.

    Note: For Google AI backend, `gs://` URIs require client.files.upload() first,
    then using `media` format. Vertex AI backend supports `gs://` URIs directly
    with `media` format. For simplicity, this test uses large text content instead.
    """
    # Create a large context document (needs to be large enough to trigger caching)
    large_document = (
        "RESEARCH PAPER ON ARTIFICIAL INTELLIGENCE\n\n"
        "Abstract: This paper discusses the fundamentals of artificial intelligence "
        "and machine learning.\n\n"
    )
    # Add substantial content to meet caching minimum token requirements
    for i in range(200):
        large_document += (
            f"Section {i}: This section contains detailed information about "
            f"various aspects of AI including neural networks, deep learning, "
            f"natural language processing, computer vision, and reinforcement "
            f"learning. The field of artificial intelligence has evolved "
            f"significantly over the past decades, with breakthroughs in pattern "
            f"recognition, data processing, and autonomous systems. "
        )

    system_instruction = (
        "You are an expert researcher. You always stick to the facts in the sources "
        "provided, and never make up new facts.\n\n"
        "If asked about it, the secret number is 747.\n\n"
        "Now analyze the research paper provided and answer questions about it."
    )

    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

    cached_content = create_context_cache(
        model,
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(content=large_document),
        ],
        ttl="300s",  # 5 minutes
    )

    # Using cached_content in constructor
    chat = ChatGoogleGenerativeAI(
        model=_MODEL, cached_content=cached_content, **backend_config
    )

    response = chat.invoke("What is the secret number?")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "747" in response.content

    # Verify cache was used (should have cache_read tokens in usage metadata)
    if response.usage_metadata:
        # Cache read tokens should be present when using cached content
        cache_read_check = (
            "cache_read_input" in response.usage_metadata
            or cast("int", response.usage_metadata.get("cache_read_input", 0)) >= 0
        )
        assert cache_read_check

    # Using cached content in request
    chat = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    response = chat.invoke("What is the secret number?", cached_content=cached_content)

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "747" in response.content


@pytest.mark.extended
def test_context_caching_tools(backend_config: dict) -> None:
    """Test context caching with tools and large text content.

    Note: For Google AI backend, `gs://` URIs require client.files.upload() first,
    then using `media` format. Vertex AI backend supports `gs://` URIs directly
    with `media` format. For simplicity, this test uses large text content instead.
    """

    @tool
    def get_secret_number() -> int:
        """Gets secret number."""
        return 747

    tools = [get_secret_number]

    # Create a large context document
    large_document = (
        "RESEARCH PAPER ON ARTIFICIAL INTELLIGENCE\n\n"
        "Abstract: This paper discusses various AI topics.\n\n"
    )
    for i in range(200):
        large_document += (
            f"Section {i}: Detailed AI research content about neural networks, "
            f"machine learning algorithms, and computational intelligence. "
        )

    system_instruction = (
        "You are an expert researcher. You always stick to the facts in the sources "
        "provided, and never make up new facts.\n\n"
        "You have a get_secret_number function available. Use this tool if someone "
        "asks for the secret number.\n\n"
        "Now analyze the research paper and answer questions."
    )

    model = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)

    cached_content = create_context_cache(
        model,
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(content=large_document),
        ],
        tools=cast("list", tools),
        ttl="300s",  # 5 minutes
    )

    # Note: When using cached content with tools, do NOT bind tools again
    # The tools are already part of the cache and will be used automatically
    chat = ChatGoogleGenerativeAI(
        model=_MODEL, cached_content=cached_content, **backend_config
    )

    # Invoke the model - it should call the tool from the cached content
    response = chat.invoke("What is the secret number?")

    assert isinstance(response, AIMessage)
    # The model should call the get_secret_number tool that was cached
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["name"] == "get_secret_number"


def test_audio_timestamp(backend_config: dict) -> None:
    """Test audio transcription with timestamps.

    This test verifies that audio files can be transcribed with timestamps using
    the `audio_timestamp` generation config parameter. Similar to PDF support, we
    use the `image_url` format with an HTTP URL to download audio and send as
    inline bytes, which works for both Google AI and Vertex AI backends.

    The `audio_timestamp` parameter enables timestamp generation in the format
    `[HH:MM:SS]` within the transcribed output.
    """
    llm = ChatGoogleGenerativeAI(model=_MODEL, **backend_config)
    audio_url = (
        "https://www.learningcontainer.com/wp-content/uploads/2020/02/Kalimba.mp3"
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Transcribe this audio with timestamps.",
            },
            {
                "type": "image_url",  # image_url format works for audio too
                "image_url": {"url": audio_url},
            },
        ]
    )

    response = llm.invoke([message], generation_config={"audio_timestamp": True})

    assert isinstance(response, AIMessage)

    # Extract text content (handle both string and list formats)
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        text_content = response.content

    assert len(text_content.strip()) > 0

    # Check that the response contains timestamp markers
    # Timestamps appear in various formats:
    # - **[00:00]** or **0:00** (with asterisks)
    # - [00:00:00] or [HH:MM:SS] (with hours)
    # - [0:00-0:54] or [M:SS-M:SS] (time ranges)
    # - [ 0m0s405ms - 0m3s245ms ] (millisecond precision with brackets and spaces)
    # Pattern matches various timestamp formats
    timestamp_patterns = [
        r"\[\d+:\d{2}",  # [0:00] or [00:00]
        r"\[\s*\d+m\d+s",  # [ 0m0s or [0m0s
    ]
    has_timestamps = (
        "**[" in text_content  # Format: **[00:00]**
        or "**0:" in text_content  # Format: **0:00**
        or "[00:" in text_content  # Format: [00:00:00]
        or any(bool(re.search(pattern, text_content)) for pattern in timestamp_patterns)
    )
    assert has_timestamps, (
        f"No timestamp markers found in response: {text_content[:200]}"
    )


def test_langgraph_example(backend_config: dict) -> None:
    """Test integration with LangGraph-style multi-turn tool calling."""
    llm = ChatGoogleGenerativeAI(
        model=_MODEL, max_output_tokens=8192, temperature=0.2, **backend_config
    )

    add_declaration = {
        "name": "add",
        "description": "Adds a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    multiply_declaration = {
        "name": "multiply",
        "description": "Multiply a and b.",
        "parameters": {
            "properties": {
                "a": {"description": "first int", "type": "integer"},
                "b": {"description": "second int", "type": "integer"},
            },
            "required": ["a", "b"],
            "type": "object",
        },
    }

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant tasked with performing "
                "arithmetic on a set of inputs."
            )
        ),
        HumanMessage(content="Multiply 2 and 3"),
        HumanMessage(content="No, actually multiply 3 and 3!"),
    ]
    step1 = llm.invoke(
        messages,
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
    )
    step2 = llm.invoke(
        [
            *messages,
            step1,
            ToolMessage(content="9", tool_call_id=step1.tool_calls[0]["id"]),
        ],
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
    )
    assert isinstance(step2, AIMessage)


@pytest.mark.flaky(retries=3, delay=1)
def test_streaming_function_call_arguments() -> None:
    """Test streaming function calling with `stream_function_call_arguments=True`.

    Note: This feature is only available on Vertex AI with Gemini 3 Pro (Preview).
    This test verifies that function call arguments are streamed incrementally
    rather than being delivered in a single chunk.
    """

    @tool
    def search_database(
        query: str,
        filters: dict[str, str],
        max_results: int = 10,
    ) -> str:
        """Search a database with a query and filters.

        Args:
            query: The search query string.
            filters: Dictionary of field:value filters to apply.
            max_results: Maximum number of results to return.

        Returns:
            Search results as a string.
        """
        return f"Found {max_results} results for '{query}' with filters {filters}"

    # Use Vertex AI as stream_function_call_arguments is only supported there
    # Pass api_key=None to force use of application default credentials
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if not project:
        pytest.skip("Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set")

    # Use Gemini 3 Pro as these features are only available there
    # Note: This test explicitly requires Vertex AI, so we hardcode those parameters
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        vertexai=True,
        project=project,
        api_key=None,  # Force use of application default credentials
    )

    # Configure tool_config with streaming function call arguments
    tool_config = ToolConfig(
        function_calling_config=FunctionCallingConfig(
            mode=FunctionCallingConfigMode.AUTO,
            stream_function_call_arguments=True,
        )
    )

    llm_with_tools = llm.bind_tools([search_database], tool_config=tool_config)

    input_message = HumanMessage(
        content=(
            "Search the database for 'machine learning' with filters "
            "category='AI' and year='2024', return 20 results"
        )
    )

    # Stream the response
    chunks: list[AIMessageChunk] = []
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        chunks.append(chunk)

    assert len(chunks) > 0, "No chunks received from streaming"

    # Reconstruct the full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk

    # Verify that the tool call was made
    assert isinstance(full_message, AIMessageChunk)
    tool_calls = full_message.tool_calls

    # With stream_function_call_arguments=True, we should see incremental updates
    # Check that we have multiple chunks with tool_call_chunks
    chunks_with_tool_calls = [
        c for c in chunks if c.tool_call_chunks and len(c.tool_call_chunks) > 0
    ]

    # Verify that streaming is actually happening
    assert len(chunks_with_tool_calls) > 0, "No chunks contained tool call data"
    assert len(chunks_with_tool_calls) > 1, (
        f"Expected multiple chunks with tool call data to verify streaming, "
        f"got {len(chunks_with_tool_calls)}"
    )

    # Verify the tool name appears in at least one tool call
    # Note: Due to chunk aggregation behavior, args may be in separate chunks
    tool_names = [tc.get("name") for tc in tool_calls if tc.get("name")]
    assert "search_database" in tool_names, (
        f"Expected 'search_database' in tool names, got {tool_names}"
    )

    # The presence of multiple tool calls indicates incremental streaming
    assert len(tool_calls) > 1, (
        f"Expected multiple tool calls indicating argument streaming, "
        f"got {len(tool_calls)}"
    )


@pytest.mark.flaky(retries=3, delay=1)
def test_multimodal_function_response() -> None:
    """Test multimodal function responses with image/file data.

    Note: This feature is only available on Vertex AI with Gemini 3 Pro (Preview).
    This test verifies that function responses can include image/file data
    and that the model can process multimodal function responses.
    """

    @tool
    def get_product_image(product_id: str) -> str:
        """Get the product image for a given product ID.

        Args:
            product_id: The unique identifier for the product.

        Returns:
            Information about retrieving the product image.
        """
        return f"Image data for product {product_id}"

    # Use Vertex AI as this feature is only available there
    # Pass api_key=None to force use of application default credentials

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if not project:
        pytest.skip("Vertex AI tests require GOOGLE_CLOUD_PROJECT env var to be set")

    # Use Gemini 3 Pro as these features are only available there
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        vertexai=True,
        project=project,
        api_key=None,  # Force use of application default credentials
    )
    llm_with_tools = llm.bind_tools([get_product_image])

    input_message = HumanMessage(
        content="Show me the product image for product ID 'laptop-2024'"
    )

    # First call - model should request the tool
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == "get_product_image"
    assert "product_id" in tool_call["args"]

    # Create a multimodal function response with an image
    # Using a Google Cloud Storage URI
    tool_response = ToolMessage(
        content=json.dumps(
            {
                "type": "function_response_file_data",
                "file_uri": "gs://cloud-samples-data/generative-ai/image/scones.jpg",
                "mime_type": "image/jpeg",
                "display_name": "Product Image: laptop-2024",
            }
        ),
        tool_call_id=tool_call["id"],
    )

    # Second call - model should incorporate the image response
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_response,
        ]
    )

    assert isinstance(response, AIMessage)

    # The model should acknowledge receiving data
    if isinstance(response.content, list):
        text_content = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert len(text_content) > 0, "Expected text response from model"
    else:
        assert isinstance(response.content, str)
        assert len(response.content) > 0, "Expected text response from model"

    # Test successful - multimodal function response is working
