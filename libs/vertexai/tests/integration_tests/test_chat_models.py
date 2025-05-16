"""Test ChatGoogleVertexAI chat model."""

import base64
import io
import json
import os
import re
from typing import List, Literal, Optional, cast

import pytest
import requests
import vertexai  # type: ignore[import-untyped, unused-ignore]
from google.cloud import storage
from google.cloud.aiplatform_v1beta1.types import Blob, Content, Part
from google.oauth2 import service_account
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_google_vertexai import (
    ChatVertexAI,
    FunctionCallingConfig,
    HarmBlockThreshold,
    HarmCategory,
    create_context_cache,
)
from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai.chat_models import _parse_chat_history_gemini
from tests.integration_tests.conftest import (
    _DEFAULT_MODEL_NAME,
    _DEFAULT_THINKING_MODEL_NAME,
)

model_names_to_test = [_DEFAULT_MODEL_NAME]
endpoint_versions = ["v1", "v1beta1"]

rate_limiter = InMemoryRateLimiter(requests_per_second=1.0)


def _check_usage_metadata(message: AIMessage) -> None:
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] > 0
    assert message.usage_metadata["output_tokens"] > 0
    assert message.usage_metadata["total_tokens"] > 0
    assert (
        message.usage_metadata["input_tokens"] + message.usage_metadata["output_tokens"]
    ) == message.usage_metadata["total_tokens"]


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
@pytest.mark.parametrize("endpoint_version", endpoint_versions)
def test_initialization(model_name: Optional[str], endpoint_version: str) -> None:
    """Test chat model initialization."""
    model = ChatVertexAI(
        model_name=model_name,
        rate_limiter=rate_limiter,
        endpoint_version=endpoint_version,
    )
    assert model._llm_type == "vertexai"


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
@pytest.mark.parametrize("endpoint_version", endpoint_versions)
def test_vertexai_single_call(model_name: Optional[str], endpoint_version: str) -> None:
    model = ChatVertexAI(
        model_name=model_name,
        rate_limiter=rate_limiter,
        endpoint_version=endpoint_version,
    )
    message = HumanMessage(content="Hello")
    response = model([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    _check_usage_metadata(response)


@pytest.mark.release
@pytest.mark.xfail(reason="vertex api doesn't respect n/candidate_count")
def test_candidates() -> None:
    model = ChatVertexAI(
        model_name="chat-bison@001", temperature=0.3, n=2, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")
    response = model.generate(messages=[[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 2


@pytest.mark.release
async def test_vertexai_agenerate() -> None:
    model = ChatVertexAI(
        temperature=0, model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")
    response = await model.agenerate([[message]])
    assert isinstance(response, LLMResult)
    async_generation = cast(ChatGeneration, response.generations[0][0])
    output_message = async_generation.message
    assert isinstance(output_message, AIMessage)
    _check_usage_metadata(output_message)

    sync_response = model.generate([[message]])
    sync_generation = cast(ChatGeneration, sync_response.generations[0][0])

    usage_metadata = sync_generation.generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) > 0
    assert int(usage_metadata["candidates_token_count"]) > 0
    usage_metadata = async_generation.generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) > 0
    assert int(usage_metadata["candidates_token_count"]) > 0


@pytest.mark.release
def test_vertexai_stream() -> None:
    model = ChatVertexAI(
        temperature=0, model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")

    sync_response = model.stream([message])
    full: Optional[BaseMessageChunk] = None
    chunks_with_usage_metadata = 0
    chunks_with_model_name = 0
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        if chunk.response_metadata.get("model_name"):
            chunks_with_model_name += 1
        full = chunk if full is None else full + chunk
    if model._is_gemini_model:
        if chunks_with_usage_metadata != 1 or chunks_with_model_name != 1:
            pytest.fail("Expected exactly one chunk with usage metadata or model_name.")
        assert isinstance(full, AIMessageChunk)
        _check_usage_metadata(full)
        assert full.response_metadata["model_name"] == _DEFAULT_MODEL_NAME


@pytest.mark.release
async def test_vertexai_astream() -> None:
    model = ChatVertexAI(
        temperature=0, model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")

    full: Optional[BaseMessageChunk] = None
    chunks_with_usage_metadata = 0
    chunks_with_model_name = 0
    async for chunk in model.astream([message]):
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        if chunk.response_metadata.get("model_name"):
            chunks_with_model_name += 1
        full = chunk if full is None else full + chunk
    if chunks_with_usage_metadata != 1 or chunks_with_model_name != 1:
        pytest.fail("Expected exactly one chunk with usage metadata or model_name.")
    assert isinstance(full, AIMessageChunk)
    _check_usage_metadata(full)
    assert full.response_metadata["model_name"] == _DEFAULT_MODEL_NAME


@pytest.mark.release
def test_multimodal() -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    gcs_url = (
        "gs://cloud-samples-data/generative-ai/image/"
        "320px-Felis_catus-cat_on_snow.jpg"
    )
    image_message = {
        "type": "image_url",
        "image_url": {"url": gcs_url},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    message = HumanMessage(content=[text_message, image_message])
    output = llm.invoke([message])
    assert isinstance(output.content, str)
    assert isinstance(output, AIMessage)
    _check_usage_metadata(output)

    # Test streaming with gemini-1.5-pro
    llm = ChatVertexAI(model_name="gemini-1.5-pro", rate_limiter=rate_limiter)
    for chunk in llm.stream([message]):
        assert isinstance(chunk, AIMessageChunk)


video_param = pytest.param(
    "gs://cloud-samples-data/generative-ai/video/pixel8.mp4",
    "video/mp4",
    id="video",
)
multimodal_inputs = [
    video_param,
    pytest.param(
        "gs://cloud-samples-data/generative-ai/audio/pixel.mp3", "audio/mp3", id="audio"
    ),
    pytest.param(
        "gs://cloud-samples-data/generative-ai/image/cricket.jpeg",
        "image/jpeg",
        id="image",
    ),
]


@pytest.mark.release
@pytest.mark.parametrize("file_uri,mime_type", multimodal_inputs)
def test_multimodal_media_file_uri(file_uri, mime_type) -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    media_message = {
        "type": "media",
        "file_uri": file_uri,
        "mime_type": mime_type,
    }
    text_message = {
        "type": "text",
        "text": "Describe the attached media in 5 words!",
    }
    message = HumanMessage(content=[text_message, media_message])
    output = llm([message])
    assert isinstance(output.content, str)


@pytest.mark.release
@pytest.mark.parametrize("file_uri,mime_type", multimodal_inputs)
@pytest.mark.first
def test_multimodal_media_inline_base64(file_uri, mime_type) -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    storage_client = storage.Client()
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    text_message = {
        "type": "text",
        "text": "Describe the attached media in 5 words!",
    }
    message = HumanMessage(content=[text_message, media_message])
    output = llm([message])
    assert isinstance(output.content, str)


@pytest.mark.release
@pytest.mark.first
@pytest.mark.xfail(reason="need a model supporting more than 1M input tokens")
def test_multimodal_media_inline_base64_template() -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    storage_client = storage.Client()
    file_uri = (
        "gs://cloud-samples-data/generative-ai/audio/audio_summary_clean_energy.mp3"
    )
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    text_message = {"type": "text", "text": "Describe the attached media in 5 words!"}
    message = HumanMessage(content=[media_message, text_message])
    chain = prompt_template | llm
    output = chain.invoke({"input": [message]})
    assert isinstance(output.content, str)


@pytest.mark.extended
def test_multimodal_media_inline_base64_agent() -> None:
    from langchain import agents

    @tool
    def get_climate_info(query: str):
        """Retrieves information about the Climate."""
        return "MOCK CLIMATE INFO STRING"

    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    storage_client = storage.Client()
    # Can't use the pixel.mp3, since it has too many tokens it will hit quota
    # error.
    file_uri = (
        "gs://cloud-samples-data/generative-ai/audio/audio_summary_clean_energy.mp3"
    )
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    text_message = {"type": "text", "text": "Describe the attached media in 5 words."}
    message = [media_message, text_message]
    tools = [get_climate_info]
    agent = agents.create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template,
    )
    agent_executor = agents.AgentExecutor(  # type: ignore[call-arg]
        agent=agent,
        tools=tools,
        verbose=False,
        stream_runnable=False,
    )
    output = agent_executor.invoke({"input": message})
    assert isinstance(output["output"], str)


def test_audio_timestamp():
    storage_client = storage.Client()
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)

    file_uri = (
        "gs://cloud-samples-data/generative-ai/audio/audio_summary_clean_energy.mp3"
    )
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    instruction = """
    Transcribe the video.
    """
    text_message = {"type": "text", "text": instruction}

    message = HumanMessage(content=[media_message, text_message])
    output = llm.invoke([message], audio_timestamp=True)

    assert isinstance(output.content, str)
    assert re.search(r"^\d+:\d+", output.content)


def test_parse_history_gemini_multimodal_FC():
    storage_client = storage.Client()
    # Can't use the pixel.mp3, since it has too many tokens it will hit quota
    # error.
    file_uri = (
        "gs://cloud-samples-data/generative-ai/audio/audio_summary_clean_energy.mp3"
    )
    mime_type = "audio/mp3"
    blob = storage.Blob.from_string(file_uri, client=storage_client)
    media_base64 = base64.b64encode(blob.download_as_bytes()).decode()
    media_message = {
        "type": "media",
        "data": media_base64,
        "mime_type": mime_type,
    }
    instruction = "Describe the attached media in 5 words."
    text_message = {"type": "text", "text": instruction}
    message = str([media_message, text_message])
    history: List[BaseMessage] = [HumanMessage(content=message)]
    parts = [
        Part(inline_data=Blob(data=media_base64, mime_type=mime_type)),
        Part(text=instruction),
    ]
    expected = [Content(role="user", parts=parts)]
    imageBytesLoader = ImageBytesLoader()
    _, response = _parse_chat_history_gemini(
        history=history,
        imageBytesLoader=imageBytesLoader,
        perform_literal_eval_on_string_raw_content=True,
    )
    assert expected == response


@pytest.mark.xfail(reason="investigating")
@pytest.mark.release
@pytest.mark.parametrize("file_uri,mime_type", [video_param])
def test_multimodal_video_metadata(file_uri, mime_type) -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    media_message = {
        "type": "media",
        "file_uri": file_uri,
        "mime_type": mime_type,
        "video_metadata": {
            "start_offset": {"seconds": 22, "nanos": 5000},
            "end_offset": {"seconds": 25, "nanos": 5000},
        },
    }
    text_message = {
        "type": "text",
        "text": "What is shown in the subtitles",
    }

    message = HumanMessage(content=[text_message, media_message])
    output = llm([message])
    assert isinstance(output.content, str)


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call_with_history(model_name: Optional[str]) -> None:
    model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
def test_vertexai_system_message() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    system_instruction = """CymbalBank is a bank located in London"""
    text_question1 = "Where is Cymbal located? Provide only the name of the city."
    sys_message = SystemMessage(content=system_instruction)
    message1 = HumanMessage(content=text_question1)
    response = model([sys_message, message1])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "london" in response.content.lower()


@pytest.mark.release
def test_vertexai_single_call_with_no_system_messages() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_get_num_tokens_from_messages(model_name: str) -> None:
    model = ChatVertexAI(
        model_name=model_name, temperature=0.0, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")
    token = model.get_num_tokens_from_messages(messages=[message])
    assert isinstance(token, int)
    assert token == 3


def _check_tool_calls(response: BaseMessage, expected_name: str) -> None:
    """Check tool calls are as expected."""
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == expected_name
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert arguments == {
        "name": "Erick",
        "age": 27.0,
    }
    tool_calls = response.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["name"] == expected_name
    assert tool_call["args"] == {"age": 27.0, "name": "Erick"}


@pytest.mark.extended
@pytest.mark.parametrize("endpoint_version", endpoint_versions)
def test_chat_vertexai_gemini_function_calling(endpoint_version: str) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    # Test .bind_tools with BaseModel
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=rate_limiter,
        endpoint_version=endpoint_version,
    ).bind_tools([MyModel])
    response = model.invoke([message])
    _check_tool_calls(response, "MyModel")

    # Test .bind_tools with function
    def my_model(name: str, age: int) -> None:
        """Invoke this with names and ages."""
        pass

    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=rate_limiter,
    ).bind_tools([my_model])
    response = model.invoke([message])
    _check_tool_calls(response, "my_model")

    # Test .bind_tools with tool
    @tool
    def my_tool(name: str, age: int) -> None:
        """Invoke this with names and ages."""
        pass

    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME, safety_settings=safety
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
    assert tool_call_chunk["args"] == '{"age": 27.0, "name": "Erick"}'


@pytest.mark.release
def test_chat_vertexai_gemini_function_calling_tool_config_any() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=rate_limiter,
    ).bind(
        functions=[MyModel],
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.ANY,
                "allowed_function_names": ["MyModel"],
            }
        },
    )
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call
    assert function_call["name"] == "MyModel"
    arguments_str = function_call.get("arguments")
    assert arguments_str
    arguments = json.loads(arguments_str)
    assert arguments == {
        "name": "Erick",
        "age": 27.0,
    }


@pytest.mark.release
def test_chat_vertexai_gemini_function_calling_tool_config_none() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, safety_settings=safety).bind(
        functions=[MyModel],
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.NONE,
            }
        },
    )
    message = HumanMessage(content="My name is Erick and I am 27 years old")
    response = model.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content != ""
    function_call = response.additional_kwargs.get("function_call")
    assert function_call is None


@pytest.mark.release
def test_chat_model_multiple_system_message() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)
    response = model.invoke(
        [
            SystemMessage("Be helpful"),
            AIMessage("Hi, I'm LeoAI. How can I help?"),
            SystemMessage("Your name is LeoAI"),
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.release
@pytest.mark.parametrize("method", [None, "json_mode"])
def test_chat_vertexai_gemini_with_structured_output(
    method: Optional[Literal["json_mode"]],
) -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        rate_limiter=rate_limiter,
    )
    model = llm.with_structured_output(MyModel, method=method)
    message = HumanMessage(content="My name is Erick and I am 27 years old")

    response = model.invoke([message])
    assert isinstance(response, MyModel)
    assert response == MyModel(name="Erick", age=27)

    if method is None:  # This won't work with json_schema as it expects an OpenAPI dict
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


@pytest.mark.release
def test_chat_vertexai_gemini_with_structured_output_nested_model() -> None:
    class Argument(BaseModel):
        description: str

    class Reason(BaseModel):
        strength: int
        argument: list[Argument]

    class Response(BaseModel):
        response: str
        reasons: list[Reason]

    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME).with_structured_output(
        Response, method="json_mode"
    )

    response = model.invoke("Why is Real Madrid better than Barcelona?")

    assert isinstance(response, Response)


@pytest.mark.flaky(retries=3)
@pytest.mark.release
def test_chat_vertexai_gemini_function_calling_with_multiple_parts() -> None:
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

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        safety_settings=safety,
        temperature=0,
        rate_limiter=rate_limiter,
    )
    llm_with_search = llm.bind(
        functions=tools,
    )
    llm_with_search_force = llm_with_search.bind(
        tool_config={
            "function_calling_config": {
                "mode": FunctionCallingConfig.Mode.ANY,
                "allowed_function_names": ["search"],
            }
        },
    )
    request = HumanMessage(
        content="Please tell the primary color of following birds: sparrow, hawk, crow",
    )
    response = llm_with_search_force.invoke([request])

    assert isinstance(response, AIMessage)
    tool_calls = response.tool_calls
    assert len(tool_calls) == 3

    tool_response = search("sparrow")
    tool_messages: List[BaseMessage] = []

    for tool_call in tool_calls:
        assert tool_call["name"] == "search"
        tool_message = ToolMessage(
            name=tool_call["name"],
            content=json.dumps(tool_response),
            tool_call_id=(tool_call["id"] or ""),
        )
        tool_messages.append(tool_message)

    result = llm_with_search.invoke([request, response] + tool_messages)

    assert isinstance(result, AIMessage)
    assert "brown" in result.content
    assert len(result.tool_calls) == 0


@pytest.mark.release
def test_chat_vertexai_gemini_thinking_auto() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_THINKING_MODEL_NAME)
    response = model.invoke("How many O's are in Google?")
    assert isinstance(response, AIMessage)
    assert (
        response.usage_metadata["total_tokens"]  # type: ignore
        > response.usage_metadata["input_tokens"]  # type: ignore
        + response.usage_metadata["output_tokens"]  # type: ignore
    )


@pytest.mark.release
def test_chat_vertexai_gemini_thinking_configured() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_THINKING_MODEL_NAME, thinking_budget=100)
    response = model.invoke("How many O's are in Google?")
    assert isinstance(response, AIMessage)
    assert (
        response.usage_metadata["total_tokens"]  # type: ignore
        > response.usage_metadata["input_tokens"]  # type: ignore
        + response.usage_metadata["output_tokens"]  # type: ignore
    )


@pytest.mark.release
def test_chat_vertexai_gemini_thinking() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_THINKING_MODEL_NAME)
    response = model.invoke("How many O's are in Google?", thinking_budget=100)
    assert isinstance(response, AIMessage)
    assert (
        response.usage_metadata["total_tokens"]  # type: ignore
        > response.usage_metadata["input_tokens"]  # type: ignore
        + response.usage_metadata["output_tokens"]  # type: ignore
    )


@pytest.mark.release
def test_chat_vertexai_gemini_thinking_disabled() -> None:
    model = ChatVertexAI(model_name=_DEFAULT_THINKING_MODEL_NAME, thinking_budget=0)
    response = model.invoke([HumanMessage("How many O's are in Google?")])
    assert isinstance(response, AIMessage)
    assert (
        response.usage_metadata["total_tokens"]  # type: ignore
        == response.usage_metadata["input_tokens"]  # type: ignore
        + response.usage_metadata["output_tokens"]  # type: ignore
    )
    assert "output_token_details" not in response.usage_metadata  # type: ignore


@pytest.mark.extended
@pytest.mark.first
def test_prediction_client_transport():
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)

    assert model.prediction_client.transport.kind == "grpc"
    assert model.async_prediction_client.transport.kind == "grpc_asyncio"

    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter, api_transport="rest"
    )

    assert model.prediction_client.transport.kind == "rest"
    assert model.async_prediction_client.transport.kind == "grpc_asyncio"

    vertexai.init(api_transport="grpc")  # Reset global config to "grpc"


@pytest.mark.extended
def test_structured_output_schema_json():
    model = ChatVertexAI(
        rate_limiter=rate_limiter,
        model_name=_DEFAULT_MODEL_NAME,
        response_mime_type="application/json",
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        },
    )

    response = model.invoke("List a few popular cookie recipes")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    parsed_response = json.loads(response.content)
    assert isinstance(parsed_response, list)
    assert len(parsed_response) > 0
    assert "recipe_name" in parsed_response[0]

    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                    },
                },
                "required": ["recipe_name"],
            },
        },
        rate_limiter=rate_limiter,
    )
    with pytest.raises(ValueError, match="response_mime_type"):
        response = model.invoke("List a few popular cookie recipes")


@pytest.mark.release
def test_json_mode_typeddict() -> None:
    class MyModel(TypedDict):
        name: str
        age: int

    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-exp",
        rate_limiter=rate_limiter,
    )
    model = llm.with_structured_output(MyModel, method="json_mode")
    message = HumanMessage(content="My name is Erick and I am 28 years old")

    response = model.invoke([message])
    assert isinstance(response, dict)
    assert response == {"name": "Erick", "age": 28}

    # Test stream
    for chunk in model.stream([message]):
        assert isinstance(chunk, dict)
        assert all(key in ["name", "age"] for key in chunk.keys())
    assert chunk == {"name": "Erick", "age": 28}


@pytest.mark.extended
def test_structured_output_schema_enum():
    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        response_schema={"type": "STRING", "enum": ["drama", "comedy", "documentary"]},
        response_mime_type="text/x.enum",
        rate_limiter=rate_limiter,
    )

    response = model.invoke(
        """
        The film aims to educate and inform viewers about real-life subjects, events, or
        people. It offers a factual record of a particular topic by combining interviews
        , historical footage and narration. The primary purpose of a film is to present 
        information and provide insights into various aspects of reality.
        """
    )

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

    assert response.content in ("drama", "comedy", "documentary")


@pytest.mark.extended
@pytest.mark.first
def test_context_catching():
    system_instruction = """
    
    You are an expert researcher. You always stick to the facts in the sources provided,
    and never make up new facts.
    
    If asked about it, the secret number is 747.
    
    Now look at these research papers, and answer the following questions.
    
    """

    cached_content = create_context_cache(
        ChatVertexAI(
            model_name=_DEFAULT_MODEL_NAME,
            rate_limiter=rate_limiter,
        ),
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
                        },
                    },
                ]
            ),
        ],
    )

    # Using cached_content in constructor
    chat = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        cached_content=cached_content,
        rate_limiter=rate_limiter,
    )

    response = chat.invoke("What is the secret number?")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

    # Using cached content in request
    chat = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)
    response = chat.invoke("What is the secret number?", cached_content=cached_content)

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.extended
@pytest.mark.first
def test_context_catching_tools():
    from langchain import agents

    @tool
    def get_secret_number() -> int:
        """Gets secret number."""
        return 747

    tools = [get_secret_number]
    system_instruction = """
    You are an expert researcher. You always stick to the facts in the sources
    provided, and never make up new facts.

    You have a get_secret_number function available. Use this tool if someone asks
    for the secret number.
        
    Now look at these research papers, and answer the following questions.
        
    """

    cached_content = create_context_cache(
        model=ChatVertexAI(
            model_name=_DEFAULT_MODEL_NAME,
        ),
        messages=[
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
                        },
                    },
                ]
            ),
        ],
        tools=tools,
    )

    chat = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        cached_content=cached_content,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = agents.create_tool_calling_agent(
        llm=chat,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = agents.AgentExecutor(  # type: ignore[call-arg]
        agent=agent, tools=tools, verbose=False, stream_runnable=False
    )
    response = agent_executor.invoke({"input": "what is the secret number?"})
    assert isinstance(response["output"], str)


@pytest.mark.release
def test_json_serializable() -> None:
    llm = ChatVertexAI(
        model_name="gemini-pro-1.5",
    )
    # Needed to init self.client and self.async_client
    llm.prediction_client
    llm.async_prediction_client
    json.loads(llm.model_dump_json())


@pytest.mark.release
def test_langgraph_example() -> None:
    llm = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        max_output_tokens=8192,
        temperature=0.2,
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
        messages
        + [step1, ToolMessage(content="9", tool_call_id=step1.tool_calls[0]["id"])],  # type: ignore[attr-defined]
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
    )
    assert isinstance(step2, AIMessage)


@pytest.mark.asyncio
@pytest.mark.release
async def test_astream_events_langgraph_example() -> None:
    llm = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME,
        max_output_tokens=8192,
        temperature=0.2,
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
    agenerator = llm.astream_events(
        messages,
        tools=[{"function_declarations": [add_declaration, multiply_declaration]}],
        version="v2",
    )
    events = [events async for events in agenerator]
    assert len(events) > 0
    # Check the function call in the output
    output = events[-1]["data"]["output"]
    assert output.additional_kwargs["function_call"]["name"] == "multiply"


@pytest.mark.xfail(reason="can't create service account key on gcp")
@pytest.mark.release
def test_init_from_credentials_obj() -> None:
    credentials_dict = json.loads(os.environ["GOOGLE_VERTEX_AI_WEB_CREDENTIALS"])
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )
    llm = ChatVertexAI(model="gemini-1.5-flash", credentials=credentials)
    llm.invoke("how are you")


@pytest.mark.xfail(reason="can't add labels to the gemini content")
@pytest.mark.release
def test_label_metadata() -> None:
    llm = ChatVertexAI(
        model="gemini-1.5-flash",
        labels={
            "task": "labels_using_declaration",
            "environment": "testing",
        },
    )
    llm.invoke("hey! how are you")


@pytest.mark.xfail(reason="can't add labels to the gemini content using invoke method")
@pytest.mark.release
def test_label_metadata_invoke_method() -> None:
    llm = ChatVertexAI(model="gemini-1.5-flash")
    llm.invoke(
        "hello! invoke method",
        labels={
            "task": "labels_using_invoke",
            "environment": "testing",
        },
    )


@pytest.mark.release
def test_response_metadata_avg_logprobs() -> None:
    llm = ChatVertexAI(model="gemini-1.5-flash")
    response = llm.invoke("Hello!")
    probs = response.response_metadata.get("avg_logprobs")
    if probs is not None:
        assert isinstance(probs, float)


@pytest.fixture
def multimodal_pdf_chain() -> RunnableSerializable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """
            Describe the provided document.
            """
            ),
            HumanMessagePromptTemplate.from_template(
                [
                    {"type": "text", "text": "# Document"},
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                ]
            ),
        ]
    )

    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)

    chain = prompt | model

    return chain


@pytest.mark.release
def test_multimodal_pdf_input_gcs(multimodal_pdf_chain: RunnableSerializable) -> None:
    gcs_uri = "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf"
    # GCS URI
    response = multimodal_pdf_chain.invoke(dict(image=gcs_uri))
    assert isinstance(response, AIMessage)


@pytest.mark.release
def test_multimodal_pdf_input_url(multimodal_pdf_chain: RunnableSerializable) -> None:
    url = "https://abc.xyz/assets/95/eb/9cef90184e09bac553796896c633/2023q4-alphabet-earnings-release.pdf"
    # URL
    response = multimodal_pdf_chain.invoke(dict(image=url))
    assert isinstance(response, AIMessage)


@pytest.mark.release
def test_multimodal_pdf_input_b64(multimodal_pdf_chain: RunnableSerializable) -> None:
    url = "https://abc.xyz/assets/95/eb/9cef90184e09bac553796896c633/2023q4-alphabet-earnings-release.pdf"
    request_response = requests.get(url, allow_redirects=True)
    # B64
    with io.BytesIO() as stream:
        stream.write(request_response.content)
        image_data = base64.b64encode(stream.getbuffer()).decode("utf-8")
        image = f"data:application/pdf;base64,{image_data}"
        response = multimodal_pdf_chain.invoke(dict(image=image))
        assert isinstance(response, AIMessage)


@pytest.mark.xfail(reason="logprobs are subject to daily quotas")
@pytest.mark.release
def test_logprobs() -> None:
    llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=2)
    msg = llm.invoke("hey")
    tokenprobs = msg.response_metadata.get("logprobs_result")
    assert tokenprobs is None or isinstance(tokenprobs, list)
    if tokenprobs:
        stack = tokenprobs[:]
        while stack:
            token = stack.pop()
            assert isinstance(token, dict)
            assert "token" in token and "logprob" in token
            assert isinstance(token.get("token"), str)
            assert isinstance(token.get("logprob"), float)
            if "top_logprobs" in token and token.get("top_logprobs") is not None:
                assert isinstance(token.get("top_logprobs"), list)
                stack.extend(token.get("top_logprobs", []))

    llm2 = ChatVertexAI(model="gemini-1.5-flash", logprobs=True)
    msg2 = llm2.invoke("how are you")
    assert msg2.response_metadata["logprobs_result"]

    llm3 = ChatVertexAI(model="gemini-1.5-flash", logprobs=False)
    msg3 = llm3.invoke("howdy")
    assert msg3.response_metadata.get("logprobs_result") is None


def test_location_init() -> None:
    # If I don't initialize vertexai before, defaults to us-central-1
    llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=2)
    assert llm.location == "us-central1"

    # If I init vertexai with other region the model is in that particular region
    vertexai.init(location="europe-west1")
    llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=2)
    assert llm.location == "europe-west1"

    # If I specify the location, it follows that location
    llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=2, location="europe-west2")
    assert llm.location == "europe-west2"

    # It reverts to the default
    vertexai.init(location="us-central1")
    llm = ChatVertexAI(model="gemini-1.5-flash", logprobs=2)
    assert llm.location == "us-central1"


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
@pytest.mark.parametrize("endpoint_version", endpoint_versions)
def test_vertexai_global_location_single_call(
    model_name: Optional[str], endpoint_version: str
) -> None:
    """Test ChatVertexAI single call with global location."""
    model = ChatVertexAI(
        model_name=model_name,
        location="global",
        rate_limiter=rate_limiter,
        endpoint_version=endpoint_version,
    )
    assert model.location == "global"
    message = HumanMessage(content="Hello")
    response = model([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    _check_usage_metadata(response)


def test_nested_bind_tools():
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME)

    class Person(BaseModel):
        name: str = Field(description="The name.")
        hair_color: Optional[str] = Field("Hair color, only if provided.")

    class People(BaseModel):
        data: list[Person] = Field(description="The people.")

    llm = ChatVertexAI(model="gemini-2.0-flash-001")
    llm_with_tools = llm.bind_tools([People], tool_choice="People")

    response = llm_with_tools.invoke("Chester, no hair color provided.")
    assert isinstance(response, AIMessage)
    assert response.tool_calls[0]["name"] == "People"
