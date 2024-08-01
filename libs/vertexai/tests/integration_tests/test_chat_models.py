"""Test ChatGoogleVertexAI chat model."""

import base64
import json
from typing import List, Optional, cast

import pytest
from google.cloud import storage
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
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool

from langchain_google_vertexai import (
    ChatVertexAI,
    FunctionCallingConfig,
    HarmBlockThreshold,
    HarmCategory,
    create_context_cache,
)
from tests.integration_tests.conftest import _DEFAULT_MODEL_NAME

model_names_to_test = [None, "codechat-bison", "chat-bison", _DEFAULT_MODEL_NAME]

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)


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
def test_initialization(model_name: Optional[str]) -> None:
    """Test chat model initialization."""
    if model_name:
        model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    else:
        model = ChatVertexAI(rate_limiter=rate_limiter)
    assert model._llm_type == "vertexai"


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    else:
        model = ChatVertexAI(rate_limiter=rate_limiter)
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
@pytest.mark.parametrize("model_name", ["chat-bison@001", _DEFAULT_MODEL_NAME])
async def test_vertexai_agenerate(model_name: str) -> None:
    model = ChatVertexAI(
        temperature=0, model_name=model_name, rate_limiter=rate_limiter
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
@pytest.mark.parametrize("model_name", ["chat-bison@001", _DEFAULT_MODEL_NAME])
def test_vertexai_stream(model_name: str) -> None:
    model = ChatVertexAI(
        temperature=0, model_name=model_name, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")

    sync_response = model.stream([message])
    full: Optional[BaseMessageChunk] = None
    chunks_with_usage_metadata = 0
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        full = chunk if full is None else full + chunk
    if model._is_gemini_model:
        if chunks_with_usage_metadata != 1:
            pytest.fail("Expected exactly one chunk with usage metadata")
        assert isinstance(full, AIMessageChunk)
        _check_usage_metadata(full)


@pytest.mark.release
async def test_vertexai_astream() -> None:
    model = ChatVertexAI(
        temperature=0, model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter
    )
    message = HumanMessage(content="Hello")

    full: Optional[BaseMessageChunk] = None
    chunks_with_usage_metadata = 0
    async for chunk in model.astream([message]):
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata:
            chunks_with_usage_metadata += 1
        full = chunk if full is None else full + chunk
    if chunks_with_usage_metadata != 1:
        pytest.fail("Expected exactly one chunk with usage metadata")
    assert isinstance(full, AIMessageChunk)
    _check_usage_metadata(full)


@pytest.mark.release
def test_vertexai_single_call_with_context() -> None:
    model = ChatVertexAI(rate_limiter=rate_limiter)
    raw_context = (
        "My name is Peter. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
def test_multimodal() -> None:
    llm = ChatVertexAI(model_name="gemini-pro-vision", rate_limiter=rate_limiter)
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
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-preview-0514", rate_limiter=rate_limiter
    )
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
def test_multimodal_media_inline_base64(file_uri, mime_type) -> None:
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-preview-0514", rate_limiter=rate_limiter
    )
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


@pytest.mark.xfail(reason="investigating")
@pytest.mark.release
@pytest.mark.parametrize("file_uri,mime_type", [video_param])
def test_multimodal_video_metadata(file_uri, mime_type) -> None:
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-preview-0514", rate_limiter=rate_limiter
    )
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


@pytest.mark.xfail(reason="investigating")
@pytest.mark.extended
def test_multimodal_history() -> None:
    llm = ChatVertexAI(model_name="gemini-pro-vision", rate_limiter=rate_limiter)
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
    message1 = HumanMessage(content=[text_message, image_message])
    message2 = AIMessage(
        content=(
            "This is a picture of a cat in the snow. The cat is a tabby cat, which is "
            "a type of cat with a striped coat. The cat is standing in the snow, and "
            "its fur is covered in snow."
        )
    )
    message3 = HumanMessage(content="What time of day is it?")
    response = llm([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
def test_vertexai_single_call_with_examples() -> None:
    model = ChatVertexAI(rate_limiter=rate_limiter)
    raw_context = "My name is Peter. You are my personal assistant."
    question = "2+2"
    text_question, text_answer = "4+4", "8"
    inp = HumanMessage(content=text_question)
    output = AIMessage(content=text_answer)
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message], examples=[inp, output])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call_with_history(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    else:
        model = ChatVertexAI(rate_limiter=rate_limiter)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
@pytest.mark.parametrize("model_name", ["gemini-1.0-pro-002"])
def test_vertexai_system_message(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    else:
        model = ChatVertexAI(rate_limiter=rate_limiter)
    system_instruction = """CymbalBank is a bank located in London"""
    text_question1 = "Where is Cymbal located? Provide only the name of the city."
    sys_message = SystemMessage(content=system_instruction)
    message1 = HumanMessage(content=text_question1)
    response = model([sys_message, message1])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "london" in response.content.lower()


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call_with_no_system_messages(
    model_name: Optional[str],
) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name, rate_limiter=rate_limiter)
    else:
        model = ChatVertexAI(rate_limiter=rate_limiter)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.release
def test_vertexai_single_call_fails_no_message() -> None:
    chat = ChatVertexAI(rate_limiter=rate_limiter)
    with pytest.raises(ValueError) as exc_info:
        _ = chat([])
    assert (
        str(exc_info.value)
        == "You should provide at least one message to start the chat!"
    )


@pytest.mark.release
@pytest.mark.parametrize("model_name", model_names_to_test)
def test_get_num_tokens_from_messages(model_name: str) -> None:
    if model_name:
        model = ChatVertexAI(
            model_name=model_name, temperature=0.0, rate_limiter=rate_limiter
        )
    else:
        model = ChatVertexAI(temperature=0.0, rate_limiter=rate_limiter)
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
def test_chat_vertexai_gemini_function_calling() -> None:
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
        model_name="gemini-1.5-pro-preview-0409",
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
    model = ChatVertexAI(
        model_name="gemini-1.5-pro-preview-0409", safety_settings=safety
    ).bind(
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
def test_chat_vertexai_gemini_function_calling_with_structured_output() -> None:
    class MyModel(BaseModel):
        name: str
        age: int

    safety = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
    }
    llm = ChatVertexAI(
        model_name="gemini-1.5-pro-preview-0409",
        safety_settings=safety,
        rate_limiter=rate_limiter,
    )
    model = llm.with_structured_output(MyModel)
    message = HumanMessage(content="My name is Erick and I am 27 years old")

    response = model.invoke([message])
    assert isinstance(response, MyModel)
    assert response == MyModel(name="Erick", age=27)

    model = llm.with_structured_output(
        {"name": "MyModel", "description": "MyModel", "parameters": MyModel.schema()}
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
        }
    )
    response = model.invoke([message])
    assert response == {
        "name": "Erick",
        "age": 27,
    }


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
        model_name="gemini-1.5-pro-preview-0409",
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


@pytest.mark.extended
def test_prediction_client_transport():
    model = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter)

    assert model.prediction_client.transport.kind == "grpc"

    # Not implemented for async_grpc
    # assert model.async_prediction_client.transport.kind == "async_grpc"

    model = ChatVertexAI(
        model_name=_DEFAULT_MODEL_NAME, rate_limiter=rate_limiter, api_transport="rest"
    )

    assert model.prediction_client.transport.kind == "rest"
    assert model.async_prediction_client.transport.kind == "rest"


@pytest.mark.extended
def test_structured_output_schema():
    model = ChatVertexAI(
        rate_limiter=rate_limiter,
        model_name="gemini-1.5-pro-001",
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
        model_name="gemini-1.5-pro-001",
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


@pytest.mark.extended
def test_context_catching():
    system_instruction = """
    
    You are an expert researcher. You always stick to the facts in the sources provided,
    and never make up new facts.
    
    If asked about it, the secret number is 747.
    
    Now look at these research papers, and answer the following questions.
    
    """

    cached_content = create_context_cache(
        ChatVertexAI(
            model_name="gemini-1.5-pro-001",
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
        model_name="gemini-1.5-pro-001",
        cached_content=cached_content,
        rate_limiter=rate_limiter,
    )

    response = chat.invoke("What is the secret number?")

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "747" in response.content

    # Using cached content in request
    chat = ChatVertexAI(model_name="gemini-1.5-pro-001", rate_limiter=rate_limiter)
    response = chat.invoke("What is the secret number?", cached_content=cached_content)

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "747" in response.content
