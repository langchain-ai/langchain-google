"""Test the `GoogleGenerativeAI` LLM (text completion) interface.

Chat model tests are in `test_chat_models.py` and use `ChatGoogleGenerativeAI`.
"""

from collections.abc import Generator

import pytest
from langchain_core.outputs import LLMResult

from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory

MODEL_NAMES = ["gemini-2.5-flash-lite"]


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_google_generativeai_call(model_name: str, backend_config: dict) -> None:
    """Test valid call to Google GenerativeAI text API."""
    if model_name:
        llm = GoogleGenerativeAI(max_tokens=10, model=model_name, **backend_config)
    else:
        llm = GoogleGenerativeAI(max_tokens=10, **backend_config)
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert llm._llm_type == "google_gemini"
    # Vertex AI strips the "models/" prefix, Google AI keeps it
    expected_model = (
        model_name if backend_config.get("vertexai") else f"models/{model_name}"
    )
    assert llm.client.model == expected_model


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_google_generativeai_generate(model_name: str, backend_config: dict) -> None:
    llm = GoogleGenerativeAI(temperature=0.3, model=model_name, **backend_config)
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
async def test_google_generativeai_agenerate(
    model_name: str, backend_config: dict
) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name, **backend_config)
    try:
        output = await llm.agenerate(["Please say foo:"])
        assert isinstance(output, LLMResult)
    finally:
        # Explicitly close the client to avoid resource warnings
        if llm.client and hasattr(llm.client, "client") and llm.client.client:
            llm.client.client.close()
            if llm.client.client.aio:
                await llm.client.client.aio.aclose()


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_generativeai_stream(model_name: str, backend_config: dict) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name, **backend_config)
    outputs = list(llm.stream("Please say foo:"))
    assert isinstance(outputs[0], str)


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_generativeai_get_num_tokens_gemini(
    model_name: str, backend_config: dict
) -> None:
    llm = GoogleGenerativeAI(temperature=0, model=model_name, **backend_config)
    output = llm.get_num_tokens("How are you?")
    assert output == 4


@pytest.mark.parametrize(
    "model_name",
    MODEL_NAMES,
)
def test_safety_settings_gemini(model_name: str, backend_config: dict) -> None:
    # test with blocked prompt
    llm = GoogleGenerativeAI(temperature=0, model=model_name, **backend_config)
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0

    # safety filters
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
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

    # test with safety filters on instantiation
    llm = GoogleGenerativeAI(
        model=model_name,
        safety_settings=safety_settings,
        temperature=0,
        **backend_config,
    )
    output = llm.generate(prompts=["how to make a bomb?"])
    assert isinstance(output, LLMResult)
    assert len(output.generations[0]) > 0
