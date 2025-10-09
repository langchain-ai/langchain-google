"""Test Google GenerativeAI API wrapper.

This test must be run with the GOOGLE_API_KEY env variable set to a valid API key.
"""

from collections.abc import Generator

import pytest
from langchain_core.outputs import LLMResult

from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

MODEL_NAMES = ["gemini-flash-lite-latest"]


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_google_generativeai_call(model_name: str) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GoogleGenerativeAI(max_tokens=10, model=model_name)
    else:
        llm = GoogleGenerativeAI(max_tokens=10)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_gemini"
    assert llm.client.model == f"models/{model_name}"


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
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


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
async def test_google_generativeai_agenerate(model_name: str) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_generativeai_stream(model_name: str) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name)
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_generativeai_get_num_tokens_gemini(model_name: str) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name)
    output = llm.get_num_tokens("How are you?")
    assert output == 4


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_safety_settings_gemini(model_name: str) -> None:
    # test with blocked prompt
    llm = GoogleGenerativeAI(temperature=0, model=model_name)
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0

    # safety filters
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,  # type: ignore[dict-item]
    }

    # test with safety filters directly to generate
    output = llm.generate(["how to make a bomb?"], safety_settings=safety_settings)
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0

    # test with safety filters directly to stream
    output_stream = llm.stream("how to make a bomb?", safety_settings=safety_settings)
    assert isinstance(output_stream, Generator)
    streamed_messages = list(output_stream)
    assert len(streamed_messages) > 0

    # test  with safety filters on instantiation
    llm = GoogleGenerativeAI(
        model=model_name,
        safety_settings=safety_settings,
        temperature=0,
    )
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0
