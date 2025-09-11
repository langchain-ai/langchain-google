"""Test Google GenerativeAI API wrapper.

Note: This test must be run with the GOOGLE_API_KEY environment variable set to a
      valid API key.
"""

from typing import Dict, Generator

import pytest
from langchain_core.outputs import LLMResult

from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

model_names = ["gemini-1.5-flash-latest"]


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GoogleGenerativeAI(max_tokens=10, model=model_name)
    else:
        llm = GoogleGenerativeAI(max_tokens=10)  # type: ignore[call-arg]
    output = llm("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_gemini"
    assert llm.client.model == f"models/{model_name}"


@pytest.mark.parametrize(
    "model_name",
    model_names,
)
def test_google_generativeai_generate(model_name: str) -> None:
    llm = GoogleGenerativeAI(temperature=0.3, model=model_name)
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 1
    # check the usage data
    generation_info = output.generations[0][0].generation_info
    assert generation_info is not None
    assert len(generation_info.get("usage_metadata", {})) > 0


async def test_google_generativeai_agenerate() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="models/gemini-2.0-flash-001")
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


def test_generativeai_stream() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-flash-latest")
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


def test_generativeai_get_num_tokens_gemini() -> None:
    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-flash-latest")
    output = llm.get_num_tokens("How are you?")
    assert output == 4


def test_safety_settings_gemini() -> None:
    # test with blocked prompt
    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-flash-latest")
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0

    # safety filters
    safety_settings: Dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore[dict-item]
    }

    # test with safety filters directly to generate
    output = llm.generate(["how to make a bomb?"], safety_settings=safety_settings)
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0

    # test with safety filters directly to stream
    streamed_messages = []
    output_stream = llm.stream("how to make a bomb?", safety_settings=safety_settings)
    assert isinstance(output_stream, Generator)
    for message in output_stream:
        streamed_messages.append(message)
    assert len(streamed_messages) > 0

    # test  with safety filters on instantiation
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        safety_settings=safety_settings,
        temperature=0,
    )
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0
