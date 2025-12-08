"""Test chat model integration."""

import base64
import json
import os
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from google.genai.errors import ClientError
from google.genai.types import (
    Blob,
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    FunctionResponse,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Language,
    Part,
    ThinkingLevel,
)
from google.genai.types import (
    Outcome as CodeExecutionResultOutcome,
)
from google.protobuf.struct_pb2 import Struct
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.google_genai import (
    _convert_to_v1_from_genai,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field, SecretStr
from pydantic_core._pydantic_core import ValidationError

from langchain_google_genai import HarmBlockThreshold, HarmCategory, Modality
from langchain_google_genai._compat import (
    _convert_from_v1_to_generativelanguage_v1beta,
)
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    ChatGoogleGenerativeAIError,
    _convert_to_parts,
    _convert_tool_message_to_parts,
    _get_ai_message_tool_messages_parts,
    _is_gemini_3_or_later,
    _is_gemini_25_model,
    _parse_chat_history,
    _parse_response_candidate,
    _response_to_result,
)

MODEL_NAME = "gemini-2.5-flash"

FAKE_API_KEY = "fake-api-key"

SMALL_VIEWABLE_BASE64_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="  # noqa: E501


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        top_k=2,
        top_p=1,
        temperature=0.7,
        n=2,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": MODEL_NAME,
        "ls_model_type": "chat",
        "ls_temperature": 0.7,
    }

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        max_output_tokens=10,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": MODEL_NAME,
        "ls_model_type": "chat",
        "ls_temperature": 0.7,
        "ls_max_tokens": 10,
    }

    ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=SecretStr(FAKE_API_KEY),
        top_k=2,
        top_p=1,
        temperature=0.7,
    )

    # test initialization with an invalid argument to check warning
    with patch("langchain_google_genai.chat_models.logger.warning") as mock_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            llm = ChatGoogleGenerativeAI(
                model=MODEL_NAME,
                google_api_key=SecretStr(FAKE_API_KEY),
                safety_setting={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
                },  # Invalid arg
            )
        assert llm.model == f"{MODEL_NAME}"
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


def test_safety_settings_initialization() -> None:
    """Test chat model initialization with `safety_settings` parameter."""
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }

    # Test initialization with safety_settings
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        temperature=0.7,
        safety_settings=safety_settings,
    )

    # Verify the safety_settings are stored correctly
    assert llm.safety_settings == safety_settings
    assert llm.temperature == 0.7
    assert llm.model == f"{MODEL_NAME}"


def test_initialization_inside_threadpool() -> None:
    # new threads don't have a running event loop,
    # thread pool executor easiest way to create one
    with ThreadPoolExecutor() as executor:
        executor.submit(
            ChatGoogleGenerativeAI,
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
        ).result()


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: pytest.CaptureFixture,
) -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    print(chat.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_profile() -> None:
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert model.profile
    assert not model.profile["reasoning_output"]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert model.profile
    assert model.profile["reasoning_output"]

    model = ChatGoogleGenerativeAI(
        model="foo",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert model.profile == {}


def test_parse_history() -> None:
    convert_system_message_to_human = False
    system_input = "You're supposed to answer math questions."
    text_question1, text_answer1 = "How much is 2+2?", "4"
    function_name = "calculator"
    function_call_1 = {
        "name": function_name,
        "args": {"arg1": "2", "arg2": "2", "op": "+"},
        "id": "0",
    }
    function_answer1 = json.dumps({"result": 4})
    function_call_2 = {
        "name": function_name,
        "arguments": json.dumps({"arg1": "2", "arg2": "2", "op": "*"}),
    }
    function_answer2 = json.dumps({"result": 4})
    function_call_3 = {
        "name": function_name,
        "args": {"arg1": "2", "arg2": "2", "op": "*"},
        "id": "1",
    }
    function_answer_3 = json.dumps({"result": 4})
    function_call_4 = {
        "name": function_name,
        "args": {"arg1": "2", "arg2": "3", "op": "*"},
        "id": "2",
    }
    function_answer_4 = json.dumps({"result": 6})
    text_answer1 = "They are same"

    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(
        content="",
        tool_calls=[function_call_1],
    )
    message3 = ToolMessage(
        name="calculator", content=function_answer1, tool_call_id="0"
    )
    message4 = AIMessage(
        content="",
        additional_kwargs={
            "function_call": function_call_2,
        },
    )
    message5 = FunctionMessage(name="calculator", content=function_answer2)
    message6 = AIMessage(content="", tool_calls=[function_call_3, function_call_4])
    message7 = ToolMessage(
        name="calculator", content=function_answer_3, tool_call_id="1"
    )
    message8 = ToolMessage(
        name="calculator", content=function_answer_4, tool_call_id="2"
    )
    message9 = AIMessage(content=text_answer1)
    messages = [
        system_message,
        message1,
        message2,
        message3,
        message4,
        message5,
        message6,
        message7,
        message8,
        message9,
    ]
    system_instruction, history = _parse_chat_history(messages)
    assert len(history) == 8
    assert history[0] == Content(role="user", parts=[Part(text=text_question1)])
    assert history[1] == Content(
        role="model",
        parts=[
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=function_call_1["args"],
                )
            )
        ],
    )
    assert history[2] == Content(
        role="user",
        parts=[
            Part(
                function_response=FunctionResponse(
                    name="calculator",
                    response={"result": 4},
                )
            )
        ],
    )
    assert history[3] == Content(
        role="model",
        parts=[
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=json.loads(function_call_2["arguments"]),
                )
            )
        ],
    )
    assert history[4] == Content(
        role="user",
        parts=[
            Part(
                function_response=FunctionResponse(
                    name="calculator",
                    response={"result": 4},
                )
            )
        ],
    )
    assert history[5] == Content(
        role="model",
        parts=[
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=function_call_3["args"],
                )
            ),
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=function_call_4["args"],
                )
            ),
        ],
    )
    assert history[6] == Content(
        role="user",
        parts=[
            Part(
                function_response=FunctionResponse(
                    name="calculator",
                    response={"result": 4},
                )
            ),
            Part(
                function_response=FunctionResponse(
                    name="calculator",
                    response={"result": 6},
                )
            ),
        ],
    )
    assert history[7] == Content(role="model", parts=[Part(text=text_answer1)])
    if convert_system_message_to_human:
        assert system_instruction is None
    else:
        assert system_instruction == Content(parts=[Part(text=system_input)])


@pytest.mark.parametrize("content", ['["a"]', '{"a":"b"}', "function output"])
def test_parse_function_history(content: str | list[str | dict]) -> None:
    function_message = FunctionMessage(name="search_tool", content=content)
    _parse_chat_history([function_message])


@pytest.mark.parametrize(
    "headers", [None, {}, {"X-User-Header": "Coco", "X-User-Header2": "Jamboo"}]
)
def test_additional_headers_support(headers: dict[str, str] | None) -> None:
    mock_client = Mock()
    mock_models = Mock()
    mock_generate_content = Mock()
    mock_generate_content.return_value = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="test response")]))],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )
    mock_models.generate_content = mock_generate_content
    mock_client.return_value.models = mock_models
    api_endpoint = "http://127.0.0.1:8000/ai"
    param_api_key = FAKE_API_KEY
    param_secret_api_key = SecretStr(param_api_key)

    with patch("langchain_google_genai.chat_models.Client", mock_client):
        chat = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            base_url=api_endpoint,
            additional_headers=headers,
        )

    expected_default_metadata: tuple = ()
    if not headers:
        assert chat.additional_headers == headers
    else:
        assert chat.additional_headers
        assert all(header in chat.additional_headers for header in headers)
        expected_default_metadata = tuple(headers.items())
        assert chat.default_metadata == expected_default_metadata

    response = chat.invoke("test")
    assert response.content == "test response"

    mock_client.assert_called_once_with(
        api_key=param_api_key,
        http_options=ANY,
    )
    call_http_options = mock_client.call_args_list[0].kwargs["http_options"]
    assert call_http_options.base_url == api_endpoint

    # Verify user-agent header is set
    assert "User-Agent" in call_http_options.headers
    assert "langchain-google-genai" in call_http_options.headers["User-Agent"]
    assert "ChatGoogleGenerativeAI" in call_http_options.headers["User-Agent"]

    # Verify user-provided headers are included
    if headers:
        for key, value in headers.items():
            assert call_http_options.headers[key] == value


def test_base_url_set_in_constructor() -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        base_url="http://localhost:8000",
    )
    assert chat.base_url == "http://localhost:8000"


def test_base_url_passed_to_client() -> None:
    with patch("langchain_google_genai.chat_models.Client") as mock_client:
        ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
            base_url="http://localhost:8000",
        )
        mock_client.assert_called_once_with(
            api_key=FAKE_API_KEY,
            http_options=ANY,
        )
        call_http_options = mock_client.call_args_list[0].kwargs["http_options"]
        assert call_http_options.base_url == "http://localhost:8000"
        assert "langchain-google-genai" in call_http_options.headers["User-Agent"]


def test_async_client_property() -> None:
    """Test that async_client property exposes `client.aio`."""
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    # Verify client is initialized
    client = chat.client
    assert client is not None
    # Verify async_client returns client.aio
    assert chat.async_client is client.aio
    # Verify async_client has the expected async methods
    assert hasattr(chat.async_client, "models")


def test_async_client_raises_when_client_not_initialized() -> None:
    """Test that async_client raises `ValueError` if client is `None`."""
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    # Force client to None to test error handling
    chat.client = None
    with pytest.raises(ValueError, match="Client not initialized"):
        _ = chat.async_client


def test_api_endpoint_via_client_options() -> None:
    """Test that `api_endpoint` via `client_options` is used in API calls."""
    mock_generate_content = Mock()
    api_endpoint = "https://custom-endpoint.com"
    param_api_key = FAKE_API_KEY
    param_secret_api_key = SecretStr(param_api_key)

    with patch("langchain_google_genai.chat_models.Client") as mock_client_class:
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        mock_generate_content.return_value = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text="test response")]))]
        )
        mock_client_instance.models.generate_content = mock_generate_content

        chat = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            client_options={"api_endpoint": api_endpoint},
        )

        response = chat.invoke("test")
        assert response.content == "test response"
        mock_client_class.assert_called_once_with(
            api_key=param_api_key,
            http_options=ANY,
        )
        call_http_options = mock_client_class.call_args_list[0].kwargs["http_options"]
        assert call_http_options.base_url == api_endpoint
        assert "langchain-google-genai" in call_http_options.headers["User-Agent"]


async def test_async_api_endpoint_via_client_options() -> None:
    """Test that `api_endpoint` via `client_options` is used in async API calls."""
    api_endpoint = "https://async-custom-endpoint.com"
    param_api_key = FAKE_API_KEY
    param_secret_api_key = SecretStr(param_api_key)

    with patch("langchain_google_genai.chat_models.Client") as mock_client_class:
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the aio.models.generate_content method for async calls
        mock_aio = Mock()
        mock_client_instance.aio = mock_aio
        mock_aio_models = Mock()
        mock_aio.models = mock_aio_models
        mock_aio_models.generate_content = AsyncMock(
            return_value=GenerateContentResponse(
                candidates=[
                    Candidate(
                        content=Content(
                            parts=[Part(text="async custom endpoint response")]
                        )
                    )
                ]
            )
        )

        chat = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            client_options={"api_endpoint": api_endpoint},
        )

        response = await chat.ainvoke("async custom endpoint test")
        assert response.content == "async custom endpoint response"
        mock_client_class.assert_called_once_with(
            api_key=param_api_key,
            http_options=ANY,
        )
        call_http_options = mock_client_class.call_args_list[0].kwargs["http_options"]
        assert call_http_options.base_url == api_endpoint
        assert "langchain-google-genai" in call_http_options.headers["User-Agent"]


def test_default_metadata_field_alias() -> None:
    """Test `default_metadata` and `default_metadata_input` fields work correctly."""
    # Test with default_metadata_input field name (alias) - should accept None without
    # error
    # This is the main issue: LangSmith Playground passes None to default_metadata_input
    chat1 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        default_metadata_input=None,
    )
    # When None is passed to alias, it should use the default factory and be overridden
    # by validator
    assert chat1.default_metadata == ()

    # Test with empty list for default_metadata_input (should not cause validation
    # error)
    chat2 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        default_metadata_input=[],
    )
    # Empty list should be accepted and overridden by validator
    assert chat2.default_metadata == ()

    # Test with tuple for default_metadata_input (should not cause validation error)
    chat3 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        default_metadata_input=[("X-Test", "test")],
    )
    # The validator will override this with additional_headers, so it should be empty
    assert chat3.default_metadata == ()


@pytest.mark.parametrize(
    ("raw_candidate", "expected"),
    [
        (
            {"content": {"parts": [{"text": "Mike age is 30"}]}},
            AIMessage(
                content="Mike age is 30",
                additional_kwargs={},
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {"text": "Mike age is 30"},
                        {"text": "Arthur age is 30"},
                    ]
                }
            },
            AIMessage(
                content=["Mike age is 30", "Arthur age is 30"],
                additional_kwargs={},
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/bmp",
                                "data": base64.b64decode(
                                    "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
                                ),
                            }
                        }
                    ]
                }
            },
            AIMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/bmp;base64,"
                            "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
                        },
                    }
                ]
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {"text": "This is a 1x1 BMP."},
                        {
                            "inline_data": {
                                "mime_type": "image/bmp",
                                "data": base64.b64decode(
                                    "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
                                ),
                            }
                        },
                    ]
                }
            },
            AIMessage(
                content=[
                    "This is a 1x1 BMP.",
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/bmp;base64,"
                            "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
                        },
                    },
                ]
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "function_call": FunctionCall(
                                name="Information", args={"name": "Ben"}
                            )
                        }
                    ]
                }
            },
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"name": "Ben"}),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "function_call": FunctionCall(
                                name="Information",
                                args={"info": ["A", "B", "C"]},
                            )
                        }
                    ]
                }
            },
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"info": ["A", "B", "C"]}),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"info": ["A", "B", "C"]},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "function_call": FunctionCall(
                                name="Information",
                                args={
                                    "people": [
                                        {"name": "Joe", "age": 30},
                                        {"name": "Martha"},
                                    ]
                                },
                            )
                        }
                    ]
                }
            },
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps(
                            {
                                "people": [
                                    {"name": "Joe", "age": 30},
                                    {"name": "Martha"},
                                ]
                            }
                        ),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={
                            "people": [
                                {"name": "Joe", "age": 30},
                                {"name": "Martha"},
                            ]
                        },
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "function_call": FunctionCall(
                                name="Information",
                                args={"info": [[1, 2, 3], [4, 5, 6]]},
                            )
                        }
                    ]
                }
            },
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"info": [[1, 2, 3], [4, 5, 6]]}),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"info": [[1, 2, 3], [4, 5, 6]]},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {"text": "Mike age is 30"},
                        {
                            "function_call": FunctionCall(
                                name="Information", args={"name": "Ben"}
                            )
                        },
                    ]
                }
            },
            AIMessage(
                content="Mike age is 30",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"name": "Ben"}),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            {
                "content": {
                    "parts": [
                        {
                            "function_call": FunctionCall(
                                name="Information", args={"name": "Ben"}
                            )
                        },
                        {"text": "Mike age is 30"},
                    ]
                }
            },
            AIMessage(
                content="Mike age is 30",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"name": "Ben"}),
                    },
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
    ],
)
def test_parse_response_candidate(raw_candidate: dict, expected: AIMessage) -> None:
    with patch("langchain_google_genai.chat_models.uuid.uuid4") as uuid4:
        uuid4.return_value = "00000000-0000-0000-0000-00000000000"
        response_candidate = Candidate.model_validate(raw_candidate)
        result = _parse_response_candidate(response_candidate)
        assert result.content == expected.content
        assert result.tool_calls == expected.tool_calls
        for key, value in expected.additional_kwargs.items():
            if key == "function_call":
                res_fc = result.additional_kwargs[key]
                exp_fc = value
                assert res_fc["name"] == exp_fc["name"]

                assert json.loads(res_fc["arguments"]) == json.loads(
                    exp_fc["arguments"]
                )
            else:
                res_kw = result.additional_kwargs[key]
                exp_kw = value
                assert res_kw == exp_kw


def test_parse_response_candidate_includes_model_provider() -> None:
    """Test `_parse_response_candidate` has `model_provider` in `response_metadata`."""
    raw_candidate = {
        "content": {"parts": [{"text": "Hello, world!"}]},
        "finish_reason": "STOP",
        "safety_ratings": [],
    }

    response_candidate = Candidate.model_validate(raw_candidate)
    result = _parse_response_candidate(response_candidate)

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"

    # Streaming
    result = _parse_response_candidate(response_candidate, streaming=True)

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"


def test_parse_response_candidate_includes_model_name() -> None:
    """Test that `_parse_response_candidate` includes `model_name` in
    `response_metadata`."""
    raw_candidate = {
        "content": {"parts": [{"text": "Hello, world!"}]},
        "finish_reason": "STOP",
        "safety_ratings": [],
    }

    response_candidate = Candidate.model_validate(raw_candidate)
    result = _parse_response_candidate(
        response_candidate, model_name="gemini-2.5-flash"
    )

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"
    assert result.response_metadata["model_name"] == "gemini-2.5-flash"

    # No name

    result = _parse_response_candidate(response_candidate)

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"
    assert "model_name" not in result.response_metadata


def test_streaming_chunk_concatenation_no_model_name_duplication() -> None:
    """Test that `model_name` is not duplicated when streaming chunks are concatenated.

    When chunks are combined using the += operator, string values in `response_metadata`
    get concatenated. To prevent `model_name` duplication, it should only be included
    in the last chunk (when `finish_reason` exists), not in every chunk.
    """

    # Create streaming chunks - first chunk without finish_reason
    raw_chunk1 = {
        "content": {"parts": [{"text": "Hello"}]},
        "safety_ratings": [],
    }
    chunk1_candidate = Candidate.model_validate(raw_chunk1)
    response1 = GenerateContentResponse(
        candidates=[chunk1_candidate], model_version="gemini-2.5-flash"
    )

    # Second chunk without finish_reason
    raw_chunk2 = {
        "content": {"parts": [{"text": " world"}]},
        "safety_ratings": [],
    }
    chunk2_candidate = Candidate.model_validate(raw_chunk2)
    response2 = GenerateContentResponse(
        candidates=[chunk2_candidate], model_version="gemini-2.5-flash"
    )

    # Final chunk with finish_reason
    raw_chunk3 = {
        "content": {"parts": [{"text": "!"}]},
        "finish_reason": "STOP",
        "safety_ratings": [],
    }
    chunk3_candidate = Candidate.model_validate(raw_chunk3)
    response3 = GenerateContentResponse(
        candidates=[chunk3_candidate], model_version="gemini-2.5-flash"
    )

    # Convert to LangChain messages (simulating what _stream does)
    result1 = _response_to_result(response1, stream=True)
    result2 = _response_to_result(response2, stream=True)
    result3 = _response_to_result(response3, stream=True)

    msg1 = cast("AIMessageChunk", result1.generations[0].message)
    msg2 = cast("AIMessageChunk", result2.generations[0].message)
    msg3 = cast("AIMessageChunk", result3.generations[0].message)

    # First two chunks should NOT have model_name in response_metadata
    assert "model_name" not in msg1.response_metadata
    assert "model_name" not in msg2.response_metadata

    # Only the last chunk should have model_name
    assert msg3.response_metadata["model_name"] == "gemini-2.5-flash"

    # Concatenate chunks (simulating user code with +=)
    full = msg1 + msg2 + msg3

    # Verify model_name is not duplicated
    assert full.response_metadata["model_name"] == "gemini-2.5-flash"
    assert full.response_metadata["model_name"].count("gemini") == 1, (
        "model_name should not be duplicated"
    )


def test_serialize() -> None:
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key="test-key")
    serialized = dumps(llm)
    llm_loaded = loads(
        serialized,
        secrets_map={"GOOGLE_API_KEY": "test-key"},
        valid_namespaces=["langchain_google_genai"],
    )
    # Pydantic 2 equality will fail on complex attributes like clients with
    # different IDs
    llm.client = None
    llm_loaded.client = None
    assert llm == llm_loaded


@pytest.mark.parametrize(
    "tool_message",
    [
        ToolMessage(name="tool_name", content="test_content", tool_call_id="1"),
        # Legacy agent does not set `name`
        ToolMessage(
            additional_kwargs={"name": "tool_name"},
            content="test_content",
            tool_call_id="1",
        ),
    ],
)
def test__convert_tool_message_to_parts__sets_tool_name(
    tool_message: ToolMessage,
) -> None:
    parts = _convert_tool_message_to_parts(tool_message)
    assert len(parts) == 1
    part = parts[0]
    assert part.function_response is not None
    assert part.function_response.name == "tool_name"
    assert part.function_response.response == {"output": "test_content"}


def test_supports_thinking() -> None:
    """Test that `_supports_thinking` correctly identifies model capabilities."""
    llm_image_gen = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-preview-image-generation",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert not llm_image_gen._supports_thinking()
    llm_tts = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-tts",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert not llm_tts._supports_thinking()
    llm_normal = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert llm_normal._supports_thinking()
    llm_pro = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert llm_pro._supports_thinking()
    llm_15 = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert not llm_15._supports_thinking()


def test_temperature_range_pydantic_validation() -> None:
    """Test that temperature is in the range `[0.0, 2.0]`."""
    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=2.1)

    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=-0.1)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        temperature=1.5,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": MODEL_NAME,
        "ls_model_type": "chat",
        "ls_temperature": 1.5,
    }


def test_temperature_range_model_validation() -> None:
    """Test that temperature is in the range `[0.0, 2.0]`."""
    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=2.5)

    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=-0.5)


@patch("langchain_google_genai.chat_models.Client")
def test_model_kwargs(mock_client: Mock) -> None:
    """Test we can transfer unknown params to `model_kwargs`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        convert_system_message_to_human=True,
        model_kwargs={"foo": "bar"},
    )
    assert llm.model == MODEL_NAME
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}
    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
            convert_system_message_to_human=True,
            foo="bar",
        )
        assert llm.model == MODEL_NAME
        assert llm.convert_system_message_to_human is True
        assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.parametrize(
    "is_async,method_name,client_method",
    [
        (False, "_generate", "models.generate_content"),  # Sync
        (True, "_agenerate", "aio.models.generate_content"),  # Async
    ],
)
@pytest.mark.parametrize(
    "instance_timeout,call_timeout,expected_timeout_ms,should_have_timeout",
    [
        (5.0, None, 5000, True),  # Instance-level timeout (converted to ms)
        (5.0, 10.0, 10000, True),  # Call-level overrides instance (in ms)
        (None, None, None, False),  # No timeout anywhere
    ],
)
async def test_timeout_parameter_handling(
    is_async: bool,
    method_name: str,
    client_method: str,
    instance_timeout: float | None,
    call_timeout: float | None,
    expected_timeout_ms: int | None,
    should_have_timeout: bool,
) -> None:
    """Test timeout parameter handling for sync and async methods."""
    with patch(
        "langchain_google_genai.chat_models.ChatGoogleGenerativeAI.client", create=True
    ):
        # Create LLM with optional instance-level timeout
        llm_kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "google_api_key": SecretStr(FAKE_API_KEY),
        }
        if instance_timeout is not None:
            llm_kwargs["timeout"] = instance_timeout

        llm = ChatGoogleGenerativeAI(**llm_kwargs)

        # Mock the client method
        mock_response = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text="Test response")]))]
        )

        if is_async:
            mock_method = AsyncMock(return_value=mock_response)
        else:
            mock_method = Mock(return_value=mock_response)

        # Set up the mock on the client
        client_parts = client_method.split(".")
        mock_client = llm.client
        for part in client_parts[:-1]:
            mock_client = getattr(mock_client, part)
        setattr(mock_client, client_parts[-1], mock_method)

        messages: list[BaseMessage] = [HumanMessage(content="Hello")]

        # Call the appropriate method with optional call-level timeout
        method = getattr(llm, method_name)
        call_kwargs: dict[str, Any] = {}
        if call_timeout is not None:
            call_kwargs["timeout"] = call_timeout

        if is_async:
            await method(messages, **call_kwargs)
        else:
            method(messages, **call_kwargs)

        # Verify http_options were set correctly in config
        mock_method.assert_called_once()
        call_args = mock_method.call_args
        config = call_args.kwargs.get("config") or call_args[0][0].kwargs.get("config")

        if should_have_timeout:
            assert config.http_options is not None
            assert config.http_options.timeout == expected_timeout_ms
        else:
            assert config.http_options is None or config.http_options.timeout is None


@pytest.mark.parametrize(
    "instance_timeout,expected_timeout_ms,should_have_timeout",
    [
        (5.0, 5000, True),  # Instance-level timeout (converted to ms)
        (None, None, False),  # No timeout
    ],
)
def test_timeout_streaming_parameter_handling(
    instance_timeout: float | None,
    expected_timeout_ms: int | None,
    should_have_timeout: bool,
) -> None:
    """Test timeout parameter handling for streaming methods."""
    # Create LLM with optional instance-level timeout
    llm_kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "google_api_key": SecretStr(FAKE_API_KEY),
    }
    if instance_timeout is not None:
        llm_kwargs["timeout"] = instance_timeout

    llm = ChatGoogleGenerativeAI(**llm_kwargs)
    assert llm.client is not None

    # Mock the client method
    def mock_stream() -> Iterator[GenerateContentResponse]:
        yield GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text="chunk1")]))]
        )

    with patch.object(
        llm.client.models, "generate_content_stream", return_value=mock_stream()
    ):
        # Call _stream (which should include timeout in config)
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]
        request = llm._prepare_request(messages)

        # Verify timeout was set correctly in config
        config = request["config"]
        if should_have_timeout:
            assert config.http_options is not None
            assert config.http_options.timeout == expected_timeout_ms
        else:
            assert config.http_options is None or config.http_options.timeout is None


@pytest.mark.parametrize(
    "instance_max_retries,call_max_retries,expected_max_retries,should_have_max_retries",
    [
        (1, None, 1, True),  # Instance-level max_retries
        (3, 5, 5, True),  # Call-level overrides instance
        (6, None, 6, True),  # Default instance value
    ],
)
def test_max_retries_parameter_handling(
    instance_max_retries: int,
    call_max_retries: int | None,
    expected_max_retries: int,
    should_have_max_retries: bool,
) -> None:
    """Test `max_retries` handling for sync and async methods."""
    # Instance-level max_retries
    llm_kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "google_api_key": SecretStr(FAKE_API_KEY),
        "max_retries": instance_max_retries,
    }

    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    messages: list[BaseMessage] = [HumanMessage(content="Hello")]

    # Prepare request with optional call-level max_retries
    call_kwargs: dict[str, Any] = {}
    if call_max_retries is not None:
        call_kwargs["max_retries"] = call_max_retries

    request = llm._prepare_request(messages, **call_kwargs)

    # Verify max_retries was set correctly in http_options.retry_options
    config = request["config"]
    if should_have_max_retries:
        assert config.http_options is not None
        assert config.http_options.retry_options is not None
        assert config.http_options.retry_options.attempts == expected_max_retries
    else:
        assert config.http_options is None or config.http_options.retry_options is None


@pytest.mark.parametrize(
    ("raw_response", "expected_grounding_metadata"),
    [
        (
            # Case 1: Response with grounding_metadata
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Test response"}]},
                        "grounding_metadata": {
                            "grounding_chunks": [
                                {
                                    "web": {
                                        "uri": "https://example.com",
                                        "title": "Example Site",
                                    }
                                }
                            ],
                            "grounding_supports": [
                                {
                                    "segment": {
                                        "start_index": 0,
                                        "end_index": 13,
                                        "text": "Test response",
                                        "part_index": 0,
                                    },
                                    "grounding_chunk_indices": [0],
                                    "confidence_scores": [0.95],
                                }
                            ],
                            "web_search_queries": ["test query"],
                        },
                    }
                ],
                "prompt_feedback": {
                    "block_reason": "BLOCKED_REASON_UNSPECIFIED",
                    "safety_ratings": [],
                },
                "usage_metadata": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "total_token_count": 15,
                },
            },
            {
                "google_maps_widget_context_token": None,
                "grounding_chunks": [
                    {
                        "maps": None,
                        "retrieved_context": None,
                        "web": {
                            "domain": None,
                            "uri": "https://example.com",
                            "title": "Example Site",
                        },
                    }
                ],
                "grounding_supports": [
                    {
                        "segment": {
                            "start_index": 0,
                            "end_index": 13,
                            "text": "Test response",
                            "part_index": 0,
                        },
                        "grounding_chunk_indices": [0],
                        "confidence_scores": [0.95],
                    }
                ],
                "retrieval_metadata": None,
                "retrieval_queries": None,
                "search_entry_point": None,
                "source_flagging_uris": None,
                "web_search_queries": ["test query"],
            },
        ),
        (
            # Case 2: Response without grounding_metadata
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Test response"}]},
                    }
                ],
                "prompt_feedback": {
                    "block_reason": "BLOCKED_REASON_UNSPECIFIED",
                    "safety_ratings": [],
                },
                "usage_metadata": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "total_token_count": 15,
                },
            },
            {},
        ),
    ],
)
def test_response_to_result_grounding_metadata(
    raw_response: dict, expected_grounding_metadata: dict
) -> None:
    """Test that `_response_to_result` includes `grounding_metadata` in the response."""
    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    assert len(result.generations) == len(raw_response["candidates"])
    for generation in result.generations:
        grounding_metadata = (
            generation.generation_info.get("grounding_metadata", {})
            if generation.generation_info is not None
            else {}
        )
        assert grounding_metadata == expected_grounding_metadata

        # Check content format based on whether grounding metadata is present
        if expected_grounding_metadata:
            content_blocks = generation.message.content_blocks
            assert isinstance(content_blocks, list)
            assert len(content_blocks) == 1
            content_block = content_blocks[0]
            assert isinstance(content_block, dict)
            assert content_block["type"] == "text"
            assert content_block["text"] == "Test response"
            assert "annotations" in content_block
            assert len(content_block["annotations"]) == 1
            annotation = content_block["annotations"][0]
            assert annotation["type"] == "citation"
            assert annotation["cited_text"] == "Test response"
            assert annotation["url"] == "https://example.com"
            assert annotation["title"] == "Example Site"
        else:
            assert generation.message.content == "Test response"


def test_grounding_metadata_to_citations_conversion() -> None:
    """Test grounding metadata is properly converted to citations in content blocks."""
    raw_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Spain won the UEFA Euro 2024 championship by "
                                "defeating England 2-1 in the final."
                            )
                        }
                    ]
                },
                "grounding_metadata": {
                    "grounding_chunks": [
                        {
                            "web": {
                                "uri": "https://uefa.com/euro2024",
                                "title": "UEFA Euro 2024 Results",
                            }
                        },
                        {
                            "web": {
                                "uri": "https://bbc.com/sport/football",
                                "title": "BBC Sport Football",
                            }
                        },
                    ],
                    "grounding_supports": [
                        {
                            "segment": {
                                "start_index": 0,
                                "end_index": 40,
                                "text": "Spain won the UEFA Euro 2024 championship",
                                "part_index": 0,
                            },
                            "grounding_chunk_indices": [0],
                            "confidence_scores": [0.95],
                        },
                        {
                            "segment": {
                                "start_index": 41,
                                "end_index": 78,
                                "text": "by defeating England 2-1 in the final",
                                "part_index": 0,
                            },
                            "grounding_chunk_indices": [0, 1],
                            "confidence_scores": [0.92, 0.88],
                        },
                    ],
                    "web_search_queries": [
                        "UEFA Euro 2024 winner",
                        "Euro 2024 final score",
                    ],
                },
            }
        ],
        "prompt_feedback": {
            "block_reason": "BLOCKED_REASON_UNSPECIFIED",
            "safety_ratings": [],
        },
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "total_token_count": 30,
        },
    }

    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    assert len(result.generations) == 1
    message = result.generations[0].message

    assert "grounding_metadata" in message.response_metadata

    # Verify grounding metadata structure uses snake_case
    gm = message.response_metadata["grounding_metadata"]
    assert "grounding_chunks" in gm
    assert "grounding_supports" in gm
    assert "web_search_queries" in gm

    # Verify all required fields are present with correct casing
    assert len(gm["grounding_chunks"]) == 2
    assert len(gm["grounding_supports"]) == 2

    # Verify first support has all fields including start_index
    first_support = gm["grounding_supports"][0]
    assert "segment" in first_support
    assert "start_index" in first_support["segment"]
    assert first_support["segment"]["start_index"] == 0
    assert "end_index" in first_support["segment"]
    assert "grounding_chunk_indices" in first_support
    assert "confidence_scores" in first_support

    content_blocks = message.content_blocks
    text_blocks_with_citations = [
        block
        for block in content_blocks
        if block.get("type") == "text" and block.get("annotations")
    ]
    assert len(text_blocks_with_citations) > 0, "Expected citations in text blocks"

    for block in text_blocks_with_citations:
        annotations = cast("list[Any]", block.get("annotations", []))
        citations = [ann for ann in annotations if ann.get("type") == "citation"]
        assert len(citations) > 0, "Expected at least one citation"

        for citation in citations:
            assert citation.get("type") == "citation"
            assert "id" in citation
            if "url" in citation:
                assert isinstance(citation["url"], str)


def test_empty_grounding_metadata_no_citations() -> None:
    """Test that empty grounding metadata doesn't create citations."""
    raw_response = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a response without grounding."}]
                },
                "grounding_metadata": {},
            }
        ],
        "prompt_feedback": {
            "block_reason": "BLOCKED_REASON_UNSPECIFIED",
            "safety_ratings": [],
        },
        "usage_metadata": {
            "prompt_token_count": 5,
            "candidates_token_count": 8,
            "total_token_count": 13,
        },
    }

    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    message = result.generations[0].message
    content_blocks = message.content_blocks

    text_blocks_with_citations = [
        block
        for block in content_blocks
        if block.get("type") == "text" and block.get("annotations")
    ]

    assert len(text_blocks_with_citations) == 0


def test_grounding_metadata_missing_optional_fields() -> None:
    """Test handling of grounding metadata with missing optional fields."""
    raw_response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Sample text"}]},
                "grounding_metadata": {
                    "grounding_chunks": [
                        {
                            "web": {
                                "uri": "https://example.com",
                                # Missing 'title'
                            }
                        }
                    ],
                    "grounding_supports": [
                        {
                            "segment": {
                                # Missing 'text'
                                "start_index": 0,
                                "end_index": 11,
                                "part_index": 0,
                            },
                            "grounding_chunk_indices": [0],
                        }
                    ],
                    # Missing 'web_search_queries' (optional field)
                },
            }
        ],
        "prompt_feedback": {
            "block_reason": "BLOCKED_REASON_UNSPECIFIED",
            "safety_ratings": [],
        },
        "usage_metadata": {
            "prompt_token_count": 5,
            "candidates_token_count": 3,
            "total_token_count": 8,
        },
    }

    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    message = result.generations[0].message

    # Verify grounding metadata is present even with missing optional fields
    assert "grounding_metadata" in message.response_metadata
    gm = message.response_metadata["grounding_metadata"]

    # Verify structure is correct (snake_case from MessageToDict)
    assert "grounding_chunks" in gm
    assert "grounding_supports" in gm

    # Verify optional fields can be missing
    chunk = gm["grounding_chunks"][0]
    assert "web" in chunk
    assert "uri" in chunk["web"]
    # Title is missing, which is OK

    support = gm["grounding_supports"][0]
    assert "segment" in support
    # Text is missing from segment, which is OK
    assert "start_index" in support["segment"]
    assert "end_index" in support["segment"]

    # web_search_queries is optional and missing in this test


def test_grounding_metadata_multiple_parts() -> None:
    """Test grounding metadata with multiple content parts."""
    raw_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "First part. "},
                        {"text": "Second part with citation."},
                    ]
                },
                "grounding_metadata": {
                    "grounding_chunks": [
                        {"web": {"uri": "https://example.com", "title": "Example"}}
                    ],
                    "grounding_supports": [
                        {
                            "segment": {
                                "start_index": 12,  # Points to second part
                                "end_index": 38,
                                "text": "Second part with citation",
                                "part_index": 1,  # Indicates which part
                            },
                            "grounding_chunk_indices": [0],
                        }
                    ],
                },
            }
        ],
        "prompt_feedback": {
            "block_reason": "BLOCKED_REASON_UNSPECIFIED",
            "safety_ratings": [],
        },
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 10,
            "total_token_count": 20,
        },
    }

    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    message = result.generations[0].message

    # Verify grounding metadata is present
    assert "grounding_metadata" in message.response_metadata
    grounding = message.response_metadata["grounding_metadata"]
    # grounding metadata from proto.Message.to_dict() uses snake_case
    assert len(grounding["grounding_supports"]) == 1
    assert grounding["grounding_supports"][0]["segment"]["part_index"] == 1


def test_maps_grounding_content_blocks() -> None:
    """Test that `content_blocks` works with Maps grounding metadata."""
    raw_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": (
                                "Here are some great Italian restaurants near the "
                                "Eiffel Tower: Chez Pippo and La Casa di Alfio."
                            )
                        }
                    ]
                },
                "grounding_metadata": {
                    "grounding_chunks": [
                        {
                            "web": None,
                            "maps": {
                                "uri": "https://maps.google.com/?cid=8846610044005889834",
                                "title": "Chez Pippo",
                                "placeId": "places/ChIJ123",
                            },
                        },
                        {
                            "web": None,
                            "maps": {
                                "uri": "https://maps.google.com/?cid=3067710458301396100",
                                "title": "La Casa di Alfio",
                                "placeId": "places/ChIJ456",
                            },
                        },
                    ],
                    "grounding_supports": [
                        {
                            "segment": {
                                "start_index": 57,
                                "end_index": 67,
                                "text": "Chez Pippo",
                                "part_index": 0,
                            },
                            "grounding_chunk_indices": [0],
                            "confidence_scores": None,
                        },
                        {
                            "segment": {
                                "start_index": 72,
                                "end_index": 87,
                                "text": "La Casa di Alfio",
                                "part_index": 0,
                            },
                            "grounding_chunk_indices": [1],
                            "confidence_scores": None,
                        },
                    ],
                    "web_search_queries": [],
                },
            }
        ],
        "prompt_feedback": {
            "block_reason": "BLOCKED_REASON_UNSPECIFIED",
            "safety_ratings": [],
        },
        "usage_metadata": {
            "prompt_token_count": 15,
            "candidates_token_count": 25,
            "total_token_count": 40,
        },
    }

    response = GenerateContentResponse.model_validate(raw_response)
    result = _response_to_result(response, stream=False)

    message = result.generations[0].message

    assert "grounding_metadata" in message.response_metadata
    gm = message.response_metadata["grounding_metadata"]
    assert len(gm["grounding_chunks"]) == 2
    assert gm["grounding_chunks"][0]["maps"] is not None
    assert gm["grounding_chunks"][0]["web"] is None

    content_blocks = message.content_blocks
    assert isinstance(content_blocks, list)

    # Verify citations are created from maps data
    text_blocks_with_citations = [
        block
        for block in content_blocks
        if block.get("type") == "text" and block.get("annotations")
    ]
    assert len(text_blocks_with_citations) > 0, "Expected citations in text blocks"

    for block in text_blocks_with_citations:
        annotations = cast("list[Any]", block.get("annotations", []))
        citations = [ann for ann in annotations if ann.get("type") == "citation"]
        assert len(citations) > 0, "Expected at least one citation from maps"

        for citation in citations:
            assert citation.get("type") == "citation"
            assert "id" in citation
            # Maps citations should have maps.google.com URLs
            if "url" in citation:
                assert isinstance(citation["url"], str)
                assert "maps.google.com" in citation["url"]
            # Maps citations should have place_id in extras
            if "extras" in citation:
                google_ai_metadata = citation["extras"].get("google_ai_metadata", {})
                # At least one citation should have a place_id
                if "place_id" in google_ai_metadata:
                    assert google_ai_metadata["place_id"].startswith("places/")


def test_thinking_config_merging_with_generation_config() -> None:
    """Test that `thinking_config` is properly merged when passed in
    `generation_config`."""
    # Mock response with thinking content followed by regular text
    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[
                        Part(text="Let me think about this...", thought=True),
                        Part(text="There are 2 O's in Google."),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=20,
            candidates_token_count=15,
            total_token_count=35,
            cached_content_token_count=0,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert llm.client is not None

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ) as mock_client_method:
        result = llm.invoke(
            "How many O's are in Google?",
            generation_config={"thinking_config": {"include_thoughts": True}},
        )

        # Verify the call was made with merged config
        mock_client_method.assert_called_once()
        call_args = mock_client_method.call_args
        # Extract config from kwargs or positional args
        config = call_args.kwargs.get("config")
        if config is None and len(call_args.args) > 0:
            # Config might be in positional args as well
            for arg in call_args.args:
                if hasattr(arg, "thinking_config"):
                    config = arg
                    break

        assert config is not None
        assert hasattr(config, "thinking_config")
        assert config.thinking_config.include_thoughts is True

        # Verify response structure
        assert isinstance(result, AIMessage)
        content = result.content

        # Should have thinking content first
        assert isinstance(content[0], dict)
        assert content[0].get("type") == "thinking"
        assert isinstance(content[0].get("thinking"), str)
        assert content[0]["thinking"] == "Let me think about this..."

        # Should have regular text content second
        assert isinstance(content[1], str)
        assert content[1] == "There are 2 O's in Google."

        # Verify usage metadata
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] == 20
        assert result.usage_metadata["output_tokens"] == 15
        assert result.usage_metadata["total_tokens"] == 35


def test_modalities_override_in_generation_config() -> None:
    """Test response modalities in invoke `generation_config` override model-defined."""
    from langchain_google_genai import Modality

    # Mock response with both image and text content
    mock_response = Mock()
    mock_response.candidates = [
        Candidate(
            content=Content(
                parts=[
                    Part(
                        inline_data=Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(
                                "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                            ),
                        )
                    ),
                    Part(text="Meow! Here's a cat image for you."),
                ]
            ),
            finish_reason="STOP",
        )
    ]
    # Create proper usage metadata
    mock_response.usage_metadata = GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key="fake-key",
        response_modalities=[Modality.TEXT],  # Initially only TEXT
    )

    with patch.object(llm, "_generate") as mock_generate:
        mock_generate.return_value = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA"
                                        "YEEBQYFBAYGEBQYEHBWIEHCKCKJCHQDWQFXQYGBCUFHYAHSUFGHJSHB"
                                        "YWICWGIYYNKSOPGR8TMC0OMCUOKSI/2WBDAQCHBWICHKMCHMOGBYA"
                                        "KCGOKCGOKSOKCGOKCGOKCGOKCGOKCGOKSOKCGOKSOKCGOKSOKCGOKS"
                                        "OKCGOKSOKCGOKSOKCGJ/WAARICAABAAEDAISIAAHEBAEB/8QAFQABAAAAAA"
                                        "AAAAAAAAAAAAAAAV/XAAUEUAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAA"
                                        "AAAAAAAAAAAAAAAAAX/XAAUEQUAAAAAAAAAAAAAAAAAAAA/9OADAMB"
                                        "AAIRAXEAPWCDABMX/9K="
                                    )
                                },
                            },
                            "Meow! Here's a cat image for you.",
                        ],
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                )
            ]
        )

        # Invoke with generation_config that should override the model's
        # response_modalities
        result = llm.invoke(
            "Generate an image of a cat. Then, say meow!",
            config={"tags": ["meow"]},
            generation_config={
                "top_k": 2,
                "top_p": 1,
                "temperature": 0.7,
                "response_modalities": ["TEXT", "IMAGE"],  # Override to include IMAGE
            },
        )

        # Verify the response structure matches expected multimodal output
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 2

        # First item should be the image
        assert isinstance(result.content[0], dict)
        assert result.content[0].get("type") == "image_url"
        assert "url" in result.content[0].get("image_url", {})

        # Second item should be the text
        assert isinstance(result.content[1], str)
        assert not result.content[1].startswith(" ")
        assert "Meow" in result.content[1]

        # Verify usage metadata is present and valid
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] > 0
        assert result.usage_metadata["output_tokens"] > 0
        assert result.usage_metadata["total_tokens"] > 0
        assert (
            result.usage_metadata["input_tokens"]
            + result.usage_metadata["output_tokens"]
        ) == result.usage_metadata["total_tokens"]

        # Verify that the _generate method was called
        mock_generate.assert_called_once()


def test_chat_google_genai_image_content_blocks() -> None:
    """Test generating an image with mocked response and `content_blocks`
    translation."""
    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[
                        Part(text="Meow!"),
                        Part(
                            inline_data=Blob(
                                mime_type="image/png",
                                data=base64.b64decode(
                                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
                                    "FcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                                ),
                            )
                        ),
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert llm.client is not None

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ):
        result = llm.invoke(
            "Say 'meow!' and then Generate an image of a cat.",
            generation_config={
                "top_k": 2,
                "top_p": 1,
                "temperature": 0.7,
                "response_modalities": ["TEXT", "IMAGE"],
            },
        )

    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], str)
    assert isinstance(result.content[1], dict)
    assert result.content[1].get("type") == "image_url"
    assert not result.content[0].startswith(" ")

    content_blocks = result.content_blocks
    assert len(content_blocks) == 2

    text_block = content_blocks[0]
    assert text_block["type"] == "text"
    assert isinstance(text_block["text"], str)

    image_block = content_blocks[1]
    assert isinstance(image_block, dict)
    assert image_block["type"] == "image"
    assert "base64" in image_block
    assert "mime_type" in image_block
    assert image_block["mime_type"] == "image/png"
    assert image_block["base64"] == (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDw"
        "AChwGA60e6kgAAAABJRU5ErkJggg=="
    )

    # Pre-v1 we returned images using chat completions format. This translation logic
    # is already covered by unit tests in langchain core, but we add a simple test here
    # to ensure our translator function is wired up correctly
    image_blocks_v1 = _convert_to_v1_from_genai(result)
    assert len(image_blocks_v1) == 2
    assert image_blocks_v1[0] == text_block
    assert image_blocks_v1[1] == image_block


def test_content_blocks_translation_with_mixed_image_content() -> None:
    """Test converting with mixed image and text content."""
    mixed_content = [
        "Here is the image you requested:",
        {
            "type": "image_url",
            "image_url": {
                "url": base64.b64encode(
                    base64.b64decode(
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    )
                ).decode(),
            },
        },
    ]
    msg = AIMessage(content=mixed_content)  # type: ignore[arg-type]
    msg.response_metadata = {"model_provider": "google_genai"}

    content_blocks = msg.content_blocks

    assert len(content_blocks) == 2

    # First block should be text
    text_block = content_blocks[0]
    assert text_block["type"] == "text"
    assert text_block["text"] == "Here is the image you requested:"

    # Second block should be ImageContentBlock
    image_block = content_blocks[1]
    assert image_block["type"] == "image"
    assert "base64" in image_block


def test_chat_google_genai_invoke_with_audio_mocked() -> None:
    """Test generating audio with mocked response and `content_blocks` translation."""
    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                # Empty content when audio is in additional_kwargs
                content=Content(parts=[]),
                finish_reason="STOP",
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )

    wav_bytes = (  # (minimal WAV header)
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        response_modalities=[Modality.AUDIO],
    )
    assert llm.client is not None

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ):
        with patch(
            "langchain_google_genai.chat_models._parse_response_candidate"
        ) as mock_parse:
            mock_parse.return_value = AIMessage(
                content="",
                additional_kwargs={"audio": wav_bytes},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                response_metadata={"model_provider": "google_genai"},
            )

            result = llm.invoke(
                "Please say The quick brown fox jumps over the lazy dog",
            )

    assert isinstance(result, AIMessage)
    assert result.content == ""
    audio_data = result.additional_kwargs.get("audio")
    assert isinstance(audio_data, bytes)
    assert len(audio_data) >= 12
    assert audio_data[0:4] == b"RIFF"
    assert audio_data[8:12] == b"WAVE"

    # Test content_blocks translation to AudioContentBlock
    blocks = result.content_blocks
    audio_blocks = [block for block in blocks if block["type"] == "audio"]
    assert len(audio_blocks) >= 1
    audio_block = audio_blocks[0]
    assert audio_block["type"] == "audio"
    assert "base64" in audio_block
    assert audio_block["base64"] == base64.b64encode(wav_bytes).decode()


def test_auto_audio_modality_for_tts_models() -> None:
    """Test that TTS models automatically set output modality to AUDIO."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-tts",
        google_api_key=SecretStr(FAKE_API_KEY),
        # Note: NOT setting response_modalities explicitly
    )
    assert llm.client is not None

    # Mock the generate_content method to capture the config
    with patch.object(llm.client.models, "generate_content") as mock_generate:
        mock_response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(parts=[Part.from_text(text="test")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )
        mock_generate.return_value = mock_response

        llm.invoke("test message")

        # Verify that generate_content was called with AUDIO in response_modalities
        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None
        assert config.response_modalities == ["AUDIO"]


def test_explicit_modality_overrides_tts_default() -> None:
    """Test that explicitly setting `response_modalities` overrides TTS auto-config."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-tts",
        google_api_key=SecretStr(FAKE_API_KEY),
        response_modalities=[Modality.TEXT],  # Explicitly set to TEXT
    )
    assert llm.client is not None

    # Mock the generate_content method to capture the config
    with patch.object(llm.client.models, "generate_content") as mock_generate:
        mock_response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(parts=[Part.from_text(text="test")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )
        mock_generate.return_value = mock_response

        llm.invoke("test message")

        # Verify that generate_content was called with TEXT (not AUDIO)
        assert mock_generate.called
        call_kwargs = mock_generate.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None
        assert config.response_modalities == ["TEXT"]


def test_compat() -> None:
    block: types.TextContentBlock = {"type": "text", "text": "foo"}
    result = _convert_from_v1_to_generativelanguage_v1beta([block], "google_genai")
    expected = [{"text": "foo"}]
    assert result == expected

    block = {"type": "text", "text": "foo", "extras": {"signature": "bar"}}
    result = _convert_from_v1_to_generativelanguage_v1beta([block], "google_genai")
    expected = [{"text": "foo", "thought_signature": "bar"}]
    assert result == expected


def test_thought_signature_conversion() -> None:
    """Test comprehensive thought signature conversion scenarios."""

    # Test ReasoningContentBlock with signature
    reasoning_block = {
        "type": "reasoning",
        "reasoning": "Let me think about this...",
        "extras": {"signature": "signature123"},
    }
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [reasoning_block],  # type: ignore[list-item]
        "google_genai",
    )
    expected = [
        {
            "thought": True,
            "text": "Let me think about this...",
            "thought_signature": "signature123",
        }
    ]
    assert result == expected

    # Test ReasoningContentBlock without signature (should be skipped)
    reasoning_without_sig = {
        "type": "reasoning",
        "reasoning": "Thinking without signature...",
        "extras": {},
    }
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [reasoning_without_sig],  # type: ignore[list-item]
        "google_genai",
    )
    assert result == []

    # Without an extras key (should be skipped)
    reasoning_no_extras = {
        "type": "reasoning",
        "reasoning": "Thinking without extras...",
    }
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [reasoning_no_extras],  # type: ignore[list-item]
        "google_genai",
    )
    assert result == []

    # Test TextContentBlock
    text_no_sig = {"type": "text", "text": "Hello world"}
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [text_no_sig],  # type: ignore[list-item]
        "google_genai",
    )
    expected = [{"text": "Hello world"}]
    assert result == expected

    # Test TextContentBlock with empty signature
    text_empty_sig = {"type": "text", "text": "Hello", "extras": {"signature": ""}}
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [text_empty_sig],  # type: ignore[list-item]
        "google_genai",
    )
    expected = [{"text": "Hello"}]
    assert result == expected

    # Test non-google_genai provider ignores signatures
    text_with_sig_other_provider = {
        "type": "text",
        "text": "foo",
        "extras": {"signature": "bar"},
    }
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [text_with_sig_other_provider],  # type: ignore[list-item]
        "other_provider",
    )
    expected = [{"text": "foo"}]
    assert result == expected

    reasoning_other_provider = {
        "type": "reasoning",
        "reasoning": "thinking...",
        "extras": {"signature": "sig123"},
    }
    result = _convert_from_v1_to_generativelanguage_v1beta(
        [reasoning_other_provider],  # type: ignore[list-item]
        "other_provider",
    )
    assert result == []


def test_thought_signature_extraction_from_response() -> None:
    """Test thought signature extraction from API response Parts."""

    # Test thought part with signature
    binary_signature = b"test_signature_data"
    thought_part = Part(
        text="I need to think about this...",
        thought=True,
        thought_signature=binary_signature,
    )

    candidate = Candidate(content=Content(parts=[thought_part]))

    # Parse candidate (the function will extract signature if present)
    result = _parse_response_candidate(candidate, streaming=False)

    # Check that signature was extracted and base64 encoded
    assert isinstance(result.content, list)
    thinking_blocks = [
        b for b in result.content if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    assert len(thinking_blocks) == 1
    assert "signature" in thinking_blocks[0]

    # Verify signature is base64 encoded
    extracted_sig = thinking_blocks[0]["signature"]
    assert extracted_sig == base64.b64encode(binary_signature).decode("ascii")

    # Test text part with signature
    text_part_with_sig = Part(
        text="Final answer here", thought_signature=binary_signature
    )

    candidate_text = Candidate(content=Content(parts=[text_part_with_sig]))
    result_text = _parse_response_candidate(candidate_text, streaming=False)

    assert isinstance(result_text.content, list)
    text_blocks = [
        b
        for b in result_text.content
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    assert len(text_blocks) == 1
    assert "extras" in text_blocks[0]
    assert "signature" in text_blocks[0]["extras"]
    assert text_blocks[0]["extras"]["signature"] == base64.b64encode(
        binary_signature
    ).decode("ascii")

    # Test part without signature
    regular_part = Part(text="Regular text without signature")
    candidate_regular = Candidate(content=Content(parts=[regular_part]))
    result_regular = _parse_response_candidate(candidate_regular, streaming=False)

    # Should be simple string content without signatures
    assert isinstance(result_regular.content, str)
    assert result_regular.content == "Regular text without signature"

    # Test empty signature handling
    empty_sig_part = Part(text="Text with empty signature", thought_signature=b"")
    candidate_empty = Candidate(content=Content(parts=[empty_sig_part]))
    result_empty = _parse_response_candidate(candidate_empty, streaming=False)

    # Empty signature should be ignored
    assert isinstance(result_empty.content, str)
    assert result_empty.content == "Text with empty signature"

    # Test invalid signature handling (None bytes)
    invalid_sig_part = Part(text="Text with None signature", thought_signature=None)
    candidate_invalid = Candidate(content=Content(parts=[invalid_sig_part]))
    result_invalid = _parse_response_candidate(candidate_invalid, streaming=False)

    # None signature should be ignored
    assert isinstance(result_invalid.content, str)
    assert result_invalid.content == "Text with None signature"


def test_signature_round_trip_conversion() -> None:
    """Test complete round-trip signature handling in conversation context."""

    # Create a mock response with thought signature
    binary_sig = b"test_sig_data"
    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[
                        Part(
                            text="I need to think...",
                            thought=True,
                            thought_signature=binary_sig,
                        ),
                        Part(text="Final answer", thought_signature=binary_sig),
                    ]
                )
            )
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        output_version="v1",
        include_thoughts=True,
    )
    assert llm.client is not None

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ):
        # First call - get response with signatures
        result = llm.invoke("Test message")

        # Verify signatures were extracted
        assert isinstance(result.content, list)

        # Find blocks with signatures
        sig_blocks = []
        for block in result.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning" and "signature" in block:
                    sig_blocks.append(block)
                elif block.get("extras") and "signature" in block["extras"]:
                    sig_blocks.append(block)

        assert len(sig_blocks) >= 1, (
            f"Expected signature blocks, got content: {result.content}"
        )

        # Now simulate passing this result back in a conversation
        with patch(
            "langchain_google_genai.chat_models._convert_from_v1_to_generativelanguage_v1beta"
        ) as mock_convert:
            from langchain_google_genai._compat import (
                _convert_from_v1_to_generativelanguage_v1beta as real_convert,
            )

            mock_convert.side_effect = real_convert

            # Create conversation with the signature-containing message
            conversation = [
                HumanMessage(content="First message"),
                result,  # This contains signatures
                HumanMessage(content="Follow up"),
            ]

            # Set up mock for the follow-up response
            follow_up_response = GenerateContentResponse(
                candidates=[
                    Candidate(content=Content(parts=[Part(text="Follow up response")]))
                ]
            )

            with patch.object(
                llm.client.models, "generate_content", return_value=follow_up_response
            ):
                follow_up = llm.invoke(conversation)

            # Verify conversion was called
            assert mock_convert.call_count >= 1

            # Find calls with signatures
            calls_with_signatures = []
            for call in mock_convert.call_args_list:
                content_blocks, model_provider = call[0]
                if model_provider == "google_genai":
                    for block in content_blocks:
                        if isinstance(block, dict):
                            if block.get("type") == "reasoning" and block.get(
                                "extras", {}
                            ).get("signature"):
                                calls_with_signatures.append(call)
                                break
                            if block.get("type") == "text" and block.get(
                                "extras", {}
                            ).get("signature"):
                                calls_with_signatures.append(call)
                                break

            assert len(calls_with_signatures) >= 1, (
                "Expected at least one call to convert signatures"
            )

            # Verify follow-up succeeded
            assert isinstance(follow_up, AIMessage)
            assert follow_up.content is not None


def test_parse_response_candidate_adds_index_to_signature() -> None:
    """Test _parse_response_candidate adds index to function_call_signature blocks."""
    # Mock a candidate with thinking and function call with signature
    part1 = Part(text="Thinking...", thought=True)

    # Signature must be bytes
    sig = b"mysig"
    part2 = Part(
        function_call=FunctionCall(name="tool", args={}), thought_signature=sig
    )

    candidate = Candidate(content=Content(parts=[part1, part2]))

    msg = _parse_response_candidate(candidate)
    function_call_map = msg.additional_kwargs[
        "__gemini_function_call_thought_signatures__"
    ]
    tool_call_id = msg.tool_calls[0]["id"]
    assert function_call_map[tool_call_id] == base64.b64encode(sig).decode("ascii")


def test_parse_chat_history_uses_index_for_signature() -> None:
    """Test _parse_chat_history uses the index field to map signatures to tool calls."""
    sig_bytes = b"dummy_signature"
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

    # Content with thinking block (index 0) and signature block (index 1)
    # The signature block points to tool call index 0
    content = [{"type": "thinking", "thinking": "I should use the tool."}]

    tool_calls = [{"name": "my_tool", "args": {"param": "value"}, "id": "call_1"}]

    message = AIMessage(
        content=content,  # type: ignore[arg-type]
        tool_calls=tool_calls,
        additional_kwargs={
            "__gemini_function_call_thought_signatures__": {"call_1": sig_b64}
        },
    )

    # Parse the history
    _, formatted_messages = _parse_chat_history([message])

    # Check the result
    model_content = formatted_messages[0]
    assert model_content.role == "model"
    assert model_content.parts is not None
    assert len(model_content.parts) == 2

    # First part should be the thinking text (thinking blocks come first)
    thinking_part = model_content.parts[0]
    assert thinking_part.thought is True
    assert thinking_part.text == "I should use the tool."

    # Second part should be the function call with signature
    function_part = model_content.parts[1]
    assert function_part.function_call is not None
    assert function_part.function_call.name == "my_tool"
    assert function_part.thought_signature == sig_bytes


def test_system_message_only_raises_error() -> None:
    """Test that invoking with only a `SystemMessage` raises a helpful error."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )

    # Should raise ValueError when only SystemMessage is provided
    with pytest.raises(
        ValueError,
        match=r"contents are required\.",
    ):
        llm.invoke([SystemMessage(content="You are a helpful assistant")])


def test_convert_to_parts_text_only() -> None:
    """Test `_convert_to_parts` with text content."""
    # Test single string
    result = _convert_to_parts("Hello, world!")
    assert len(result) == 1
    assert result[0].text == "Hello, world!"
    assert result[0].inline_data is None
    # Test list of strings
    result = _convert_to_parts(["Hello", "world", "!"])
    assert len(result) == 3
    assert result[0].text == "Hello"
    assert result[1].text == "world"
    assert result[2].text == "!"


def test_convert_to_parts_text_content_block() -> None:
    """Test `_convert_to_parts` with text content blocks."""
    content = [{"type": "text", "text": "Hello, world!"}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].text == "Hello, world!"


def test_convert_to_parts_image_url() -> None:
    """Test `_convert_to_parts` with `image_url` content blocks."""
    content = [{"type": "image_url", "image_url": {"url": SMALL_VIEWABLE_BASE64_IMAGE}}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "image/png"


def test_convert_to_parts_image_url_string() -> None:
    """Test `_convert_to_parts` with `image_url` as string."""
    content = [{"type": "image_url", "image_url": SMALL_VIEWABLE_BASE64_IMAGE}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "image/png"


def test_convert_to_parts_file_data_url() -> None:
    """Test `_convert_to_parts` with file data URL."""
    content = [
        {
            "type": "file",
            "source_type": "url",
            "url": "https://example.com/image.jpg",
            "mime_type": "image/jpeg",
        }
    ]
    with patch("langchain_google_genai.chat_models.ImageBytesLoader") as mock_loader:
        mock_loader_instance = Mock()
        mock_loader_instance._bytes_from_url.return_value = b"fake_image_data"
        mock_loader.return_value = mock_loader_instance
        result = _convert_to_parts(content)
        assert len(result) == 1
        assert result[0].inline_data is not None
        assert result[0].inline_data.mime_type == "image/jpeg"
        assert result[0].inline_data.data == b"fake_image_data"


def test_convert_to_parts_file_data_base64() -> None:
    """Test `_convert_to_parts` with file data base64."""
    content = [
        {
            "type": "file",
            "source_type": "base64",
            "data": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
            "mime_type": "text/plain",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "text/plain"
    assert result[0].inline_data.data == b"Hello World"


def test_convert_to_parts_file_data_auto_mime_type() -> None:
    """Test `_convert_to_parts` with auto-detected mime type."""
    content = [
        {
            "type": "file",
            "source_type": "base64",
            "data": "SGVsbG8gV29ybGQ=",
            # No mime_type specified, should be auto-detected
        }
    ]
    with patch("langchain_google_genai.chat_models.mimetypes.guess_type") as mock_guess:
        mock_guess.return_value = ("text/plain", None)
        result = _convert_to_parts(content)
        assert len(result) == 1
        assert result[0].inline_data is not None
        assert result[0].inline_data.mime_type == "text/plain"


def test_convert_to_parts_file_with_file_id() -> None:
    """Test `_convert_to_parts` with `FileContentBlock` containing `file_id`."""
    content = [
        {
            "type": "file",
            "file_id": "files/abc123xyz",
            "mime_type": "application/pdf",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].file_data is not None
    assert result[0].file_data.file_uri == "files/abc123xyz"
    assert result[0].file_data.mime_type == "application/pdf"


def test_convert_to_parts_file_with_file_id_default_mime_type() -> None:
    """Test `_convert_to_parts` with `file_id` but no `mime_type` specified."""
    content = [
        {
            "type": "file",
            "file_id": "files/xyz789",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].file_data is not None
    assert result[0].file_data.file_uri == "files/xyz789"
    assert result[0].file_data.mime_type == "application/octet-stream"


def test_convert_to_parts_media_with_data() -> None:
    """Test `_convert_to_parts` with media type containing data."""
    content = [{"type": "media", "mime_type": "video/mp4", "data": b"fake_video_data"}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "video/mp4"
    assert result[0].inline_data.data == b"fake_video_data"


def test_convert_to_parts_media_with_file_uri() -> None:
    """Test `_convert_to_parts` with media type containing file_uri."""
    content = [
        {
            "type": "media",
            "mime_type": "application/pdf",
            "file_uri": "gs://bucket/file.pdf",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].file_data is not None
    assert result[0].file_data.mime_type == "application/pdf"
    assert result[0].file_data.file_uri == "gs://bucket/file.pdf"


def test_convert_to_parts_media_with_video_metadata() -> None:
    """Test `_convert_to_parts` with media type containing video metadata."""
    content = [
        {
            "type": "media",
            "mime_type": "video/mp4",
            "file_uri": "gs://bucket/video.mp4",
            "video_metadata": {"start_offset": "10s", "end_offset": "20s"},
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].file_data is not None
    assert result[0].video_metadata is not None
    assert result[0].video_metadata.start_offset == "10s"
    assert result[0].video_metadata.end_offset == "20s"


def test_convert_to_parts_executable_code() -> None:
    """Test `_convert_to_parts` with executable code."""
    content = [
        {
            "type": "executable_code",
            "language": "python",
            "executable_code": "print('Hello, World!')",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].executable_code is not None
    assert result[0].executable_code.language == Language.PYTHON
    assert result[0].executable_code.code == "print('Hello, World!')"


def test_convert_to_parts_code_execution_result() -> None:
    """Test `_convert_to_parts` with code execution result."""
    content = [
        {
            "type": "code_execution_result",
            "code_execution_result": "Hello, World!",
            "outcome": CodeExecutionResultOutcome.OUTCOME_OK,
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].code_execution_result is not None
    assert result[0].code_execution_result.output == "Hello, World!"
    assert (
        result[0].code_execution_result.outcome == CodeExecutionResultOutcome.OUTCOME_OK
    )


def test_convert_to_parts_code_execution_result_backward_compatibility() -> None:
    """Test `_convert_to_parts` with code execution result without outcome (compat)."""
    content = [
        {
            "type": "code_execution_result",
            "code_execution_result": "Hello, World!",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].code_execution_result is not None
    assert result[0].code_execution_result.output == "Hello, World!"
    assert (
        result[0].code_execution_result.outcome == CodeExecutionResultOutcome.OUTCOME_OK
    )


def test_convert_to_parts_thinking() -> None:
    """Test `_convert_to_parts` with thinking content."""
    content = [{"type": "thinking", "thinking": "I need to think about this..."}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].text == "I need to think about this..."
    assert result[0].thought is True


def test_convert_to_parts_mixed_content() -> None:
    """Test `_convert_to_parts` with mixed content types."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
        {"type": "image_url", "image_url": {"url": SMALL_VIEWABLE_BASE64_IMAGE}},
    ]
    result = _convert_to_parts(content)
    assert len(result) == 3
    assert result[0].text == "Hello"
    assert result[1].text == "World"
    assert result[2].inline_data is not None


def test_convert_to_parts_invalid_type() -> None:
    """Test `_convert_to_parts` with invalid source_type."""
    content = [
        {
            "type": "file",
            "source_type": "invalid",
            "data": "some_data",
        }
    ]
    with pytest.raises(ValueError, match="Unrecognized message part type: file"):
        _convert_to_parts(content)


def test_convert_to_parts_invalid_source_type() -> None:
    """Test `_convert_to_parts` with invalid source_type."""
    content = [
        {
            "type": "media",
            "source_type": "invalid",
            "data": "some_data",
            "mime_type": "text/plain",
        }
    ]
    with pytest.raises(ValueError, match="Data should be valid base64"):
        _convert_to_parts(content)


def test_convert_to_parts_invalid_image_url_format() -> None:
    """Test `_convert_to_parts` with invalid `image_url` format."""
    content = [{"type": "image_url", "image_url": {"invalid_key": "value"}}]
    with pytest.raises(ValueError, match="Unrecognized message image format"):
        _convert_to_parts(content)


def test_convert_to_parts_missing_mime_type_in_media() -> None:
    """Test `_convert_to_parts` with missing `mime_type` in media."""
    content = [
        {
            "type": "media",
            "file_uri": "gs://bucket/file.pdf",
            # Missing mime_type
        }
    ]
    with pytest.raises(ValueError, match="Missing mime_type in media part"):
        _convert_to_parts(content)


def test_convert_to_parts_media_missing_data_and_file_uri() -> None:
    """Test `_convert_to_parts` with media missing both data and `file_uri`."""
    content = [
        {
            "type": "media",
            "mime_type": "application/pdf",
            # Missing both data and file_uri
        }
    ]
    with pytest.raises(
        ValueError, match="Media part must have either data or file_uri"
    ):
        _convert_to_parts(content)


def test_convert_to_parts_missing_executable_code_keys() -> None:
    """Test `_convert_to_parts` with missing keys in `executable_code`."""
    content = [
        {
            "type": "executable_code",
            "language": "python",
            # Missing executable_code key
        }
    ]
    with pytest.raises(
        ValueError, match="Executable code part must have 'code' and 'language'"
    ):
        _convert_to_parts(content)


def test_convert_to_parts_missing_code_execution_result_key() -> None:
    """Test `_convert_to_parts` with missing `code_execution_result` key."""
    content = [
        {
            "type": "code_execution_result"
            # Missing code_execution_result key
        }
    ]
    with pytest.raises(
        ValueError, match="Code execution result part must have 'code_execution_result'"
    ):
        _convert_to_parts(content)


def test_convert_to_parts_unrecognized_type() -> None:
    """Test `_convert_to_parts` with unrecognized type."""
    content = [{"type": "unrecognized_type", "data": "some_data"}]
    with pytest.raises(ValueError, match="Unrecognized message part type"):
        _convert_to_parts(content)


def test_convert_to_parts_non_dict_mapping() -> None:
    """Test `_convert_to_parts` with non-dict mapping."""
    content = [123]  # Not a string or dict
    with pytest.raises(
        ChatGoogleGenerativeAIError,
        match="Unknown error occurred while converting LC message content to parts",
    ):
        _convert_to_parts(content)  # type: ignore[arg-type]


def test_convert_to_parts_unrecognized_format_warning() -> None:
    """Test `_convert_to_parts` with unrecognized format triggers warning."""
    content = [{"some_key": "some_value"}]  # Not a recognized format
    with patch("langchain_google_genai.chat_models.logger.warning") as mock_warning:
        result = _convert_to_parts(content)
        mock_warning.assert_called_once()
        assert "Unrecognized message part format" in mock_warning.call_args[0][0]
        assert len(result) == 1
        assert result[0].text == "{'some_key': 'some_value'}"


def test_convert_tool_message_to_parts_string_content() -> None:
    """Test `_convert_tool_message_to_parts` with string content."""
    message = ToolMessage(name="test_tool", content="test_result", tool_call_id="123")
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_tool"
    assert result[0].function_response.response == {"output": "test_result"}


def test_convert_tool_message_to_parts_json_content() -> None:
    """Test `_convert_tool_message_to_parts` with JSON string content."""
    message = ToolMessage(
        name="test_tool",
        content='{"result": "success", "data": [1, 2, 3]}',
        tool_call_id="123",
    )
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_tool"
    assert result[0].function_response.response == {
        "result": "success",
        "data": [1, 2, 3],
    }


def test_convert_tool_message_to_parts_dict_content() -> None:
    """Test `_convert_tool_message_to_parts` with `dict` content."""
    message = ToolMessage(  # type: ignore[call-overload]
        name="test_tool",
        content={"result": "success", "data": [1, 2, 3]},
        tool_call_id="123",
    )
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_tool"
    assert result[0].function_response.response == {
        "output": str({"result": "success", "data": [1, 2, 3]})
    }


def test_convert_tool_message_to_parts_list_content_with_media() -> None:
    """Test `_convert_tool_message_to_parts` with `list` content containing media."""
    message = ToolMessage(
        name="test_tool",
        content=[
            "Text response",
            {"type": "image_url", "image_url": {"url": SMALL_VIEWABLE_BASE64_IMAGE}},
        ],
        tool_call_id="123",
    )
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 2
    # First part should be the media (image)
    assert result[0].inline_data is not None
    # Second part should be the function response
    assert result[1].function_response is not None
    assert result[1].function_response.name == "test_tool"
    assert result[1].function_response.response == {"output": ["Text response"]}


def test_convert_tool_message_to_parts_with_name_parameter() -> None:
    """Test `_convert_tool_message_to_parts` with explicit name parameter."""
    message = ToolMessage(
        content="test_result",
        tool_call_id="123",
        # No name in message
    )
    result = _convert_tool_message_to_parts(message, name="explicit_tool_name")
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "explicit_tool_name"


def test_convert_tool_message_to_parts_legacy_name_in_kwargs() -> None:
    """Test `_convert_tool_message_to_parts` with legacy name in `additional_kwargs`."""
    message = ToolMessage(
        content="test_result",
        tool_call_id="123",
        additional_kwargs={"name": "legacy_tool_name"},
    )
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "legacy_tool_name"


def test_convert_tool_message_to_parts_function_message() -> None:
    """Test `_convert_tool_message_to_parts` with `FunctionMessage`."""
    message = FunctionMessage(name="test_function", content="function_result")
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_function"
    assert result[0].function_response.response == {"output": "function_result"}


def test_convert_tool_message_to_parts_invalid_json_fallback() -> None:
    """Test `_convert_tool_message_to_parts` with invalid JSON falls back to string."""
    message = ToolMessage(
        name="test_tool",
        content='{"invalid": json}',  # Invalid JSON
        tool_call_id="123",
    )
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.response == {"output": '{"invalid": json}'}


def test_get_ai_message_tool_messages_parts_basic() -> None:
    """Test `_get_ai_message_tool_messages_parts` with basic tool messages."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}},
            {"id": "call_2", "name": "tool_2", "args": {"arg2": "value2"}},
        ],
    )
    tool_messages = [
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1"),
        ToolMessage(name="tool_2", content="result_2", tool_call_id="call_2"),
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 2
    # Check first tool response
    assert result[0].function_response is not None
    assert result[0].function_response.name == "tool_1"
    assert result[0].function_response.response == {"output": "result_1"}
    # Check second tool response
    assert result[1].function_response is not None
    assert result[1].function_response.name == "tool_2"
    assert result[1].function_response.response == {"output": "result_2"}


def test_get_ai_message_tool_messages_parts_partial_matches() -> None:
    """Test `_get_ai_message_tool_messages_parts` with partial tool message matches."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}},
            {"id": "call_2", "name": "tool_2", "args": {"arg2": "value2"}},
        ],
    )
    tool_messages = [
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1"),
        # Missing tool_2 response
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 1
    # Only tool_1 response should be included
    assert result[0].function_response is not None
    assert result[0].function_response.name == "tool_1"
    assert result[0].function_response.response == {"output": "result_1"}


def test_get_ai_message_tool_messages_parts_no_matches() -> None:
    """Test `_get_ai_message_tool_messages_parts` with no matching tool messages."""
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}}],
    )
    tool_messages = [
        ToolMessage(name="tool_2", content="result_2", tool_call_id="call_2"),
        ToolMessage(name="tool_3", content="result_3", tool_call_id="call_3"),
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 0


def test_get_ai_message_tool_messages_parts_empty_tool_calls() -> None:
    """Test `_get_ai_message_tool_messages_parts` with empty tool calls."""
    ai_message = AIMessage(content="No tool calls")
    tool_messages = [
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1")
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 0


def test_get_ai_message_tool_messages_parts_empty_tool_messages() -> None:
    """Test `_get_ai_message_tool_messages_parts` with empty tool messages."""
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}}],
    )
    result = _get_ai_message_tool_messages_parts([], ai_message)
    assert len(result) == 0


def test_get_ai_message_tool_messages_parts_duplicate_tool_calls() -> None:
    """Test `_get_ai_message_tool_messages_parts` handles duplicate tool call IDs."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}},
            {
                "id": "call_1",
                "name": "tool_1",
                "args": {"arg1": "value1"},
            },  # Duplicate ID
        ],
    )
    tool_messages = [
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1")
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 1  # Should only process the first match
    assert result[0].function_response is not None
    assert result[0].function_response.name == "tool_1"


def test_get_ai_message_tool_messages_parts_order_preserved() -> None:
    """Test `_get_ai_message_tool_messages_parts` preserves order of tool messages."""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}},
            {"id": "call_2", "name": "tool_2", "args": {"arg2": "value2"}},
        ],
    )
    tool_messages = [
        ToolMessage(name="tool_2", content="result_2", tool_call_id="call_2"),
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1"),
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 2
    # Order should be preserved based on tool_messages order, not tool_calls order
    assert result[0].function_response is not None
    assert result[0].function_response.name == "tool_2"
    assert result[1].function_response is not None
    assert result[1].function_response.name == "tool_1"


def test_get_ai_message_tool_messages_parts_with_name_from_tool_call() -> None:
    """Test `_get_ai_message_tool_messages_parts` uses name from tool call"""
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "tool_from_call", "args": {"arg1": "value1"}}
        ],
    )
    tool_messages = [
        ToolMessage(content="result_1", tool_call_id="call_1")  # No name in message
    ]
    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert (
        result[0].function_response.name == "tool_from_call"
    )  # Should use name from tool call


def test_with_structured_output_json_schema_alias() -> None:
    """Test that `json_schema` (preferred) method works as alias for `json_mode`
    (old)."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        age: int

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key="fake-key")

    structured_llm = llm.with_structured_output(TestModel, method="json_schema")
    assert structured_llm is not None

    schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
    structured_llm_dict = llm.with_structured_output(schema_dict, method="json_schema")
    assert structured_llm_dict is not None


def test_response_json_schema_parameter() -> None:
    """Test that `response_json_schema` is properly set via `bind`."""

    class TestModel(BaseModel):
        name: str
        age: int

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    llm_with_json_schema = llm.bind(
        response_mime_type="application/json", response_json_schema=schema_dict
    )
    bound_kwargs = cast("Any", llm_with_json_schema).kwargs
    assert bound_kwargs["response_mime_type"] == "application/json"
    assert bound_kwargs["response_json_schema"] == schema_dict


def test_response_json_schema_param_mapping() -> None:
    """Test both `response_schema` and `response_json_schema` map correctly to
    `response_json_schema` in `GenerationConfig`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    # Test response_schema parameter maps to response_json_schema in gen_config
    gen_config_1 = llm._prepare_params(
        stop=None, response_mime_type="application/json", response_schema=schema_dict
    )
    assert gen_config_1.response_json_schema == schema_dict

    # Test response_json_schema parameter maps directly to response_json_schema in
    # gen_config
    gen_config_2 = llm._prepare_params(
        stop=None,
        response_mime_type="application/json",
        response_json_schema=schema_dict,
    )
    assert gen_config_2.response_json_schema == schema_dict

    # Test that response_json_schema takes precedence over response_schema
    different_schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
    }

    gen_config_3 = llm._prepare_params(
        stop=None,
        response_mime_type="application/json",
        response_schema=schema_dict,
        response_json_schema=different_schema,
    )
    assert (
        gen_config_3.response_json_schema == different_schema
    )  # response_json_schema takes precedence


def test_with_struct_out() -> None:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }

    structured_llm = llm.with_structured_output(schema, method="json_schema")
    assert structured_llm is not None

    structured_llm_mode = llm.with_structured_output(schema, method="json_mode")  # Old
    assert structured_llm_mode is not None


def test_json_schema_dict_support() -> None:
    """Test `json_schema` with dictionary schemas."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    dict_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    structured_llm_dict = llm.with_structured_output(dict_schema, method="json_schema")
    assert structured_llm_dict is not None


def test_ref_preservation() -> None:
    class RecursiveModel(BaseModel):
        name: str
        children: list["RecursiveModel"] | None = None

    RecursiveModel.model_rebuild()

    # Get the raw schema with $defs
    raw_schema = RecursiveModel.model_json_schema()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    structured = llm.with_structured_output(RecursiveModel, method="json_schema")
    llm = cast("Any", structured).first

    schema = llm.kwargs["response_json_schema"]

    assert "$defs" in schema, "json_schema should preserve $defs definitions"
    assert schema == raw_schema, "json_schema should preserve raw schema exactly"


def test_recursive_schema_support() -> None:
    """Test support for recursive schemas using `$ref`."""

    class TreeNode(BaseModel):
        value: str
        children: list["TreeNode"] | None = None

    TreeNode.model_rebuild()  # Rebuild to resolve forward references

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    structured_llm = llm.with_structured_output(TreeNode, method="json_schema")
    assert structured_llm is not None

    recursive_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "children": {
                "type": "array",
                "items": {"$ref": "#"},  # Reference to root schema
            },
        },
    }

    # json_schema should handle $ref properly
    structured_llm_dict = llm.with_structured_output(
        recursive_schema, method="json_schema"
    )
    assert structured_llm_dict is not None


def test_union_schema_with_anyof() -> None:
    """Test that `anyOf` schemas are properly handled."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    # Schema with anyOf for union support
    union_schema = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "text"},
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "number"},
                    "value": {"type": "number"},
                },
                "required": ["type", "value"],
            },
        ]
    }
    structured_llm = llm.with_structured_output(union_schema, method="json_schema")
    assert structured_llm is not None

    # Verify anyOf schemas work with previous (json_schema) method too
    structured_llm_legacy = llm.with_structured_output(
        union_schema, method="json_schema"
    )
    assert structured_llm_legacy is not None


def test_union_schema_support() -> None:
    """Test that `Union` types work correctly with both `json_schema` methods.

    This addresses a bug where `json_schema` method would fail with `KeyError`
    when processing `Union` types that generate `anyOf` arrays with `$ref` entries.
    """

    class SpamDetails(BaseModel):
        """Details for content classified as spam."""

        reason: str = Field(
            description="The reason why the content is considered spam."
        )
        spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"] = (
            Field(description="The type of spam.")
        )

    class NotSpamDetails(BaseModel):
        """Details for content classified as not spam."""

        summary: str = Field(description="A brief summary of the content.")
        is_safe: bool = Field(
            description="Whether the content is safe for all audiences."
        )

    class ModerationResult(BaseModel):
        """The result of content moderation."""

        decision: SpamDetails | NotSpamDetails

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    structured = llm.with_structured_output(ModerationResult, method="json_schema")

    llm = cast("Any", structured).first

    assert "response_json_schema" in llm.kwargs


def test_response_schema_mime_type_validation() -> None:
    """Test that `response_schema` requires correct MIME type."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key=SecretStr(FAKE_API_KEY)
    )

    schema = {"type": "object", "properties": {"field": {"type": "string"}}}

    # Test response_schema validation - error happens during _prepare_params
    with pytest.raises(
        ValueError, match=r"JSON schema structured output is only supported when"
    ):
        llm._prepare_params(
            stop=None, response_schema=schema, response_mime_type="text/plain"
        )

    # Test that binding succeeds (validation happens later during generation)
    llm_with_schema = llm.bind(
        response_schema=schema, response_mime_type="application/json"
    )
    assert llm_with_schema is not None

    llm_with_json_schema = llm.bind(
        response_json_schema=schema, response_mime_type="application/json"
    )
    assert llm_with_json_schema is not None


def test_thinking_budget_preserved_with_structured_output() -> None:
    """Test that `thinking_budget` is preserved when using `with_structured_output`."""

    class TestSchema(BaseModel):
        name: str
        value: int

    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text='{"name": "test", "value": 42}')]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_budget=0,
    )
    assert llm.client is not None

    structured_llm = llm.with_structured_output(TestSchema, method="json_schema")

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ) as mock_client_method:
        structured_llm.invoke("test input")

        mock_client_method.assert_called_once()

        call_args = mock_client_method.call_args
        config = call_args.kwargs.get("config")

        if config is None and len(call_args.args) > 0:
            for arg in call_args.args:
                if hasattr(arg, "thinking_config"):
                    config = arg
                    break

        assert config is not None, "Config should be present in API call"
        assert hasattr(config, "thinking_config"), "thinking_config should be present"
        assert config.thinking_config is not None, "thinking_config should not be None"
        assert config.thinking_config.thinking_budget == 0, (
            f"thinking_budget should be 0, got {config.thinking_config.thinking_budget}"
        )


def test_thinking_level_preserved_with_structured_output() -> None:
    """Test that `thinking_level` is preserved when using `with_structured_output`."""

    class TestSchema(BaseModel):
        name: str
        value: int

    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text='{"name": "test", "value": 42}')]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_level="low",
    )
    assert llm.client is not None

    structured_llm = llm.with_structured_output(TestSchema, method="json_schema")

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ) as mock_client_method:
        structured_llm.invoke("test input")

        mock_client_method.assert_called_once()
        call_args = mock_client_method.call_args

        config = call_args.kwargs.get("config")
        if config is None and len(call_args.args) > 0:
            for arg in call_args.args:
                if hasattr(arg, "thinking_config"):
                    config = arg
                    break

        # Verify thinking_level is in the config
        assert config is not None, "Config should be present in API call"
        assert hasattr(config, "thinking_config"), "thinking_config should be present"
        assert config.thinking_config is not None, "thinking_config should not be None"
        assert config.thinking_config.thinking_level == ThinkingLevel.LOW, (
            f"thinking_level should be LOW, got {config.thinking_config.thinking_level}"
        )


def test_include_thoughts_preserved_with_structured_output() -> None:
    """Test that `include_thoughts` is preserved when using `with_structured_output`."""

    class TestSchema(BaseModel):
        name: str
        value: int

    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    parts=[
                        Part(text="Let me think...", thought=True),
                        Part(text='{"name": "test", "value": 42}'),
                    ]
                ),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        include_thoughts=True,
    )
    assert llm.client is not None

    structured_llm = llm.with_structured_output(TestSchema, method="json_schema")

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ) as mock_client_method:
        structured_llm.invoke("test input")

        mock_client_method.assert_called_once()
        call_args = mock_client_method.call_args

        config = call_args.kwargs.get("config")
        if config is None and len(call_args.args) > 0:
            for arg in call_args.args:
                if hasattr(arg, "thinking_config"):
                    config = arg
                    break

        assert config is not None, "Config should be present in API call"
        assert hasattr(config, "thinking_config"), "thinking_config should be present"
        assert config.thinking_config is not None, "thinking_config should not be None"
        msg = (
            f"include_thoughts should be True, "
            f"got {config.thinking_config.include_thoughts}"
        )
        assert config.thinking_config.include_thoughts is True, msg


def test_thinking_budget_and_include_thoughts_with_structured_output() -> None:
    """Test that both `thinking_budget` and `include_thoughts` are preserved."""

    class TestSchema(BaseModel):
        name: str
        value: int

    mock_response = GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(parts=[Part(text='{"name": "test", "value": 42}')]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        ),
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_budget=0,
        include_thoughts=False,
    )
    assert llm.client is not None

    structured_llm = llm.with_structured_output(TestSchema, method="json_schema")

    with patch.object(
        llm.client.models, "generate_content", return_value=mock_response
    ) as mock_client_method:
        structured_llm.invoke("test input")

        mock_client_method.assert_called_once()
        call_args = mock_client_method.call_args

        config = call_args.kwargs.get("config")
        if config is None and len(call_args.args) > 0:
            for arg in call_args.args:
                if hasattr(arg, "thinking_config"):
                    config = arg
                    break

        assert config is not None, "Config should be present in API call"
        assert hasattr(config, "thinking_config"), "thinking_config should be present"
        assert config.thinking_config is not None, "thinking_config should not be None"
        assert config.thinking_config.thinking_budget == 0, (
            f"thinking_budget should be 0, got {config.thinking_config.thinking_budget}"
        )
        msg = (
            f"include_thoughts should be False, "
            f"got {config.thinking_config.include_thoughts}"
        )
        assert config.thinking_config.include_thoughts is False, msg


def test_is_new_gemini_model() -> None:
    assert _is_gemini_3_or_later("gemini-3.0-pro") is True
    assert _is_gemini_3_or_later("gemini-2.5-pro") is False
    assert _is_gemini_3_or_later("gemini-2.5-flash") is False
    assert _is_gemini_3_or_later("gemini-1.5-pro") is False
    assert _is_gemini_3_or_later("gemini-1.0-pro") is False
    assert _is_gemini_3_or_later("") is False
    assert _is_gemini_3_or_later(None) is False  # type: ignore


def test_is_gemini_25_model() -> None:
    """Test the _is_gemini_25_model function."""
    assert _is_gemini_25_model("gemini-2.5-pro") is True
    assert _is_gemini_25_model("gemini-2.5-flash") is True
    assert _is_gemini_25_model("gemini-2.5-flash-lite") is True
    assert _is_gemini_25_model("models/gemini-2.5-pro") is True
    assert _is_gemini_25_model("GEMINI-2.5-FLASH") is True
    assert _is_gemini_25_model("gemini-3.0-pro") is False
    assert _is_gemini_25_model("gemini-1.5-pro") is False
    assert _is_gemini_25_model("gemini-pro-latest") is False
    assert _is_gemini_25_model("") is False
    assert _is_gemini_25_model(None) is False  # type: ignore


def test_per_part_media_resolution_warning_gemini_25() -> None:
    """Test that per-part `media_resolution` warns for Gemini 2.5 models."""
    content_with_media_resolution = [
        {
            "type": "media",
            "mime_type": "image/jpeg",
            "data": base64.b64encode(b"fake_image_data").decode(),
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        },
        {"type": "text", "text": "Describe this image"},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convert_to_parts(content_with_media_resolution, model="gemini-2.5-flash")

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        expected_msg = (
            "Setting per-part media resolution requests to Gemini 2.5 models "
            "and older is not supported"
        )
        assert expected_msg in str(w[0].message)


def test_per_part_media_resolution_no_warning_new_models() -> None:
    """Test that per-part `media_resolution` does not warn for new models."""
    content_with_media_resolution = [
        {
            "type": "media",
            "mime_type": "image/jpeg",
            "data": base64.b64encode(b"fake_image_data").decode(),
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        },
        {"type": "text", "text": "Describe this image"},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convert_to_parts(content_with_media_resolution, model="gemini-3.0-pro")

        assert len(w) == 0


def test_per_part_media_resolution_no_warning_old_models() -> None:
    """Test that per-part `media_resolution` does not warn for old models."""
    content_with_media_resolution = [
        {
            "type": "media",
            "mime_type": "image/jpeg",
            "data": base64.b64encode(b"fake_image_data").decode(),
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        },
        {"type": "text", "text": "Describe this image"},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convert_to_parts(content_with_media_resolution, model="gemini-1.5-pro")

        assert len(w) == 0


def test_per_part_media_resolution_warning_gemini_25_data_block() -> None:
    """Test that per-part media_resolution warns for Gemini 2.5 models with data content
    blocks."""
    content_with_media_resolution = [
        {
            "type": "image",
            "base64": base64.b64encode(b"fake_image_data").decode(),
            "mime_type": "image/jpeg",
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        },
        {"type": "text", "text": "Describe this image"},
    ]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convert_to_parts(content_with_media_resolution, model="gemini-2.5-pro")

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert (
            "Setting per-part media resolution requests to Gemini 2.5 models and older "
            "is not supported" in str(w[0].message)
        )


def test_thinking_level_parameter() -> None:
    """Test that `thinking_level` is properly handled."""
    # Test with thinking_level only
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_level="low",
    )
    config = llm._prepare_params(stop=None)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == ThinkingLevel.LOW
    # Pydantic models define all fields; check value is None rather than hasattr
    assert config.thinking_config.thinking_budget is None

    # Test with thinking_level="high"
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_level="high",
    )
    config = llm._prepare_params(stop=None)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == ThinkingLevel.HIGH


def test_thinking_level_takes_precedence_over_thinking_budget() -> None:
    """Test that `thinking_level` takes precedence when both are provided."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
            thinking_level="low",
            thinking_budget=128,
        )
        config = llm._prepare_params(stop=None)

        # Check that warning was issued
        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, UserWarning)
        assert "thinking_level' takes precedence" in str(warning_list[0].message)

        # Check that thinking_level is used and thinking_budget is ignored
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == ThinkingLevel.LOW
        # Pydantic models define all fields; check value is None rather than hasattr
        assert config.thinking_config.thinking_budget is None


def test_thinking_budget_alone_still_works() -> None:
    """Test that `thinking_budget` still works when `thinking_level` is not provided."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_budget=64,
    )
    config = llm._prepare_params(stop=None)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 64
    # Pydantic models define all fields; check value is None rather than hasattr
    assert config.thinking_config.thinking_level is None


def test_kwargs_override_max_output_tokens() -> None:
    """Test that max_output_tokens can be overridden via kwargs."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        max_output_tokens=100,
    )

    config = llm._prepare_params(stop=None, max_output_tokens=500)
    assert config.max_output_tokens == 500


def test_kwargs_override_thinking_budget() -> None:
    """Test that thinking_budget can be overridden via kwargs."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_budget=64,
    )

    config = llm._prepare_params(stop=None, thinking_budget=128)
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 128


def test_kwargs_override_thinking_level() -> None:
    """Test that thinking_level can be overridden via kwargs."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        thinking_level="low",
    )

    config = llm._prepare_params(stop=None, thinking_level="high")
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_level == ThinkingLevel.HIGH


def test_client_error_raises_descriptive_error() -> None:
    """Test `ClientError` from the API is properly converted to a descriptive error."""
    invalid_model_name = "gemini-invalid-model-name"
    mock_client = Mock()
    mock_models = Mock()
    mock_generate_content = Mock()

    # Simulate a NOT_FOUND error from the API (what happens with invalid model names)
    mock_generate_content.side_effect = ClientError(
        code=404,
        response_json={
            "error": {
                "message": f"models/{invalid_model_name} is not found",
                "status": "NOT_FOUND",
            }
        },
        response=None,
    )
    mock_models.generate_content = mock_generate_content
    mock_client.return_value.models = mock_models

    with patch("langchain_google_genai.chat_models.Client", mock_client):
        chat = ChatGoogleGenerativeAI(
            model=invalid_model_name,
            google_api_key=SecretStr(FAKE_API_KEY),
            max_retries=0,  # Disable retries for faster test
        )

        with pytest.raises(
            ChatGoogleGenerativeAIError,
            match=rf"Error calling model '{invalid_model_name}' \(NOT_FOUND\)",
        ):
            chat.invoke("test")


def test_kwargs_override_response_modalities() -> None:
    """Test that `response_modalities` can be overridden via kwargs."""
    from langchain_core.messages import HumanMessage

    from langchain_google_genai import Modality

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )

    # Test passing response_modalities as kwarg to _prepare_request
    msg = HumanMessage(content="test")
    request = llm._prepare_request(
        [msg],
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    )

    # Verify response_modalities is set correctly in the config
    assert request["config"].response_modalities == ["TEXT", "IMAGE"]


def test_response_modalities_set_on_instance() -> None:
    """Test that `response_modalities` can be set on the instance."""
    from langchain_core.messages import HumanMessage

    from langchain_google_genai import Modality

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        response_modalities=[Modality.TEXT],
    )

    msg = HumanMessage(content="test")
    request = llm._prepare_request([msg])

    assert request["config"].response_modalities == ["TEXT"]


def test_kwargs_response_modalities_overrides_instance() -> None:
    """Test that kwarg `response_modalities` overrides instance value."""
    from langchain_core.messages import HumanMessage

    from langchain_google_genai import Modality

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        response_modalities=[Modality.TEXT],
    )

    msg = HumanMessage(content="test")
    request = llm._prepare_request(
        [msg],
        response_modalities=[Modality.TEXT, Modality.IMAGE],
    )

    # Kwarg should override instance value
    assert request["config"].response_modalities == ["TEXT", "IMAGE"]


# Test for protobuf integer/float conversion fix in chat models.


def test_parse_response_candidate_corrects_integer_like_floats() -> None:
    """Test that `_parse_response_candidate` correctly handles integer-like floats.

    Handling in tool call arguments from the Gemini API response.

    This test addresses a bug where `proto.Message.to_dict()` converts integers
    to floats, causing downstream type casting errors.
    """
    # Create a mock Protobuf Struct for the arguments with problematic float values
    args_struct = Struct()
    args_struct.update(
        {
            "entity_type": "table",
            "upstream_depth": 3.0,  # The problematic float value that should be int
            "downstream_depth": 5.0,  # Another problematic float value
            "fqn": "test.table.name",
            "valid_float": 3.14,  # This should remain as float
            "string_param": "test_string",  # This should remain as string
            "bool_param": True,  # This should remain as boolean
        }
    )

    # Create the mock API response candidate
    candidate = Candidate(
        content=Content(
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="get_entity_lineage",
                        args=args_struct,
                    )
                )
            ]
        )
    )

    # Call the function we are testing
    result_message = _parse_response_candidate(candidate)

    # Assert that the parsed tool_calls have the correct integer types
    assert len(result_message.tool_calls) == 1
    tool_call = result_message.tool_calls[0]
    assert tool_call["name"] == "get_entity_lineage"
    assert tool_call["args"]["upstream_depth"] == 3
    assert tool_call["args"]["downstream_depth"] == 5
    assert isinstance(tool_call["args"]["upstream_depth"], int)
    assert isinstance(tool_call["args"]["downstream_depth"], int)

    # Assert that non-integer values are preserved correctly
    assert tool_call["args"]["valid_float"] == 3.14
    assert isinstance(tool_call["args"]["valid_float"], float)
    assert tool_call["args"]["string_param"] == "test_string"
    assert isinstance(tool_call["args"]["string_param"], str)
    assert tool_call["args"]["bool_param"] is True
    assert isinstance(tool_call["args"]["bool_param"], bool)

    # Assert that the additional_kwargs also contains corrected JSON
    function_call_args = json.loads(
        result_message.additional_kwargs["function_call"]["arguments"]
    )
    assert function_call_args["upstream_depth"] == 3
    assert function_call_args["downstream_depth"] == 5
    assert isinstance(function_call_args["upstream_depth"], int)
    assert isinstance(function_call_args["downstream_depth"], int)

    # Assert that non-integer values are preserved in additional_kwargs too
    assert function_call_args["valid_float"] == 3.14
    assert isinstance(function_call_args["valid_float"], float)


def test_parse_response_candidate_handles_no_function_call() -> None:
    """Test that the function works correctly when there's no function call."""
    candidate = Candidate(
        content=Content(
            parts=[Part(text="This is a regular text response without function calls")]
        )
    )

    result_message = _parse_response_candidate(candidate)

    assert (
        result_message.content
        == "This is a regular text response without function calls"
    )
    assert len(result_message.tool_calls) == 0
    assert "function_call" not in result_message.additional_kwargs


def test_parse_response_candidate_handles_empty_args() -> None:
    """Test that the function works correctly with empty function call arguments."""
    args_struct = Struct()
    # Empty struct - no arguments

    candidate = Candidate(
        content=Content(
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="no_args_function",
                        args=args_struct,
                    )
                )
            ]
        )
    )

    result_message = _parse_response_candidate(candidate)

    assert len(result_message.tool_calls) == 1
    tool_call = result_message.tool_calls[0]
    assert tool_call["name"] == "no_args_function"
    assert tool_call["args"] == {}

    function_call_args = json.loads(
        result_message.additional_kwargs["function_call"]["arguments"]
    )
    assert function_call_args == {}


def test_backend_detection_default() -> None:
    """Test default backend detection (Gemini Developer API)."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
    )
    assert llm._use_vertexai is False  # type: ignore[attr-defined]


def test_backend_detection_explicit_vertexai_true() -> None:
    """Test explicit `vertexai=True` forces Vertex AI backend."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-project",
        vertexai=True,
    )
    assert llm._use_vertexai is True  # type: ignore[attr-defined]


def test_backend_detection_explicit_vertexai_false() -> None:
    """Test explicit `vertexai=False` forces Gemini Developer API."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-project",
        vertexai=False,
    )
    assert llm._use_vertexai is False  # type: ignore[attr-defined]


def test_backend_detection_project_auto_detects_vertexai() -> None:
    """Test that providing project parameter auto-detects Vertex AI."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=FAKE_API_KEY,
        project="test-project",
    )
    assert llm._use_vertexai is True  # type: ignore[attr-defined]


def test_backend_detection_credentials_auto_detects_vertexai() -> None:
    """Test that providing credentials parameter auto-detects Vertex AI."""
    from unittest.mock import Mock

    fake_credentials = Mock()
    fake_credentials.project_id = "test-project"

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        credentials=fake_credentials,
        project="test-project",
    )
    assert llm._use_vertexai is True  # type: ignore[attr-defined]


def test_backend_detection_env_var_vertexai_true() -> None:
    """Test `GOOGLE_GENAI_USE_VERTEXAI=true` forces Vertex AI."""
    original_env = os.environ.copy()
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
        )
        assert llm._use_vertexai is True  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_backend_detection_env_var_vertexai_false() -> None:
    """Test `GOOGLE_GENAI_USE_VERTEXAI=false` forces Gemini Developer API."""
    original_env = os.environ.copy()
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
            project="test-project",  # Would normally trigger Vertex AI
        )
        assert llm._use_vertexai is False  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_backend_detection_env_var_variations() -> None:
    """Test various values for `GOOGLE_GENAI_USE_VERTEXAI` env var."""
    original_env = os.environ.copy()

    # Test "1" as true
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
        )
        assert llm._use_vertexai is True  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    # Test "yes" as true
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "yes"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
        )
        assert llm._use_vertexai is True  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    # Test "0" as false
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
            project="test-project",
        )
        assert llm._use_vertexai is False  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_backend_detection_priority_explicit_over_env() -> None:
    """Test that explicit vertexai parameter overrides env var."""

    original_env = os.environ.copy()
    try:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

        # Explicit False should override env var True
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
            vertexai=False,
        )
        assert llm._use_vertexai is False  # type: ignore[attr-defined]

        # Explicit True should also work
        llm2 = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            api_key=FAKE_API_KEY,
            vertexai=True,
        )
        assert llm2._use_vertexai is True  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_model_name_normalization_for_vertexai() -> None:
    """Test that model names with `'models/'` prefix are normalized for Vertex AI."""
    original_env = os.environ.copy()
    try:
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"

        # Test with models/ prefix for Vertex AI - should be stripped
        llm_vertex = ChatGoogleGenerativeAI(
            model=f"models/{MODEL_NAME}",
            api_key=FAKE_API_KEY,
            vertexai=True,
        )
        assert llm_vertex.model == "gemini-2.5-flash"
        assert llm_vertex._use_vertexai is True  # type: ignore[attr-defined]

        # Test with models/ prefix for Google AI - should remain unchanged
        llm_google_ai = ChatGoogleGenerativeAI(
            model=f"models/{MODEL_NAME}",
            api_key=FAKE_API_KEY,
            vertexai=False,
        )
        assert llm_google_ai.model == "models/gemini-2.5-flash"
        assert llm_google_ai._use_vertexai is False  # type: ignore[attr-defined]

        # Test without models/ prefix for Vertex AI - should remain unchanged
        llm_vertex_no_prefix = ChatGoogleGenerativeAI(
            model=f"models/{MODEL_NAME}",
            api_key=FAKE_API_KEY,
            vertexai=True,
        )
        assert llm_vertex_no_prefix.model == "gemini-2.5-flash"
        assert llm_vertex_no_prefix._use_vertexai is True  # type: ignore[attr-defined]
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_max_retries_configuration_for_500_errors() -> None:
    """Test that `max_retries` is properly configured for handling 500 errors.

    This test verifies `max_retries` parameter is properly passed to the SDK's
    `HttpRetryOptions`.

    The actual retry logic for transient errors (including 500 INTERNAL errors) is
    handled by the `google-genai` SDK at the HTTP transport level via
    `HttpRetryOptions`.
    """

    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        max_retries=5,  # Configure retries for transient errors
    )

    # Prepare a request to inspect the configuration
    messages: list[BaseMessage] = [HumanMessage(content="test")]
    request = chat._prepare_request(messages)

    # Verify HttpRetryOptions is configured with correct attempts
    config = request["config"]
    assert config.http_options is not None, "HttpOptions should be configured"
    assert config.http_options.retry_options is not None, (
        "HttpRetryOptions should be configured"
    )
    assert config.http_options.retry_options.attempts == 5, (
        "Retry attempts should match max_retries"
    )


def test_max_retries_can_be_overridden_per_call() -> None:
    """Test that `max_retries` can be overridden on a per-call basis.

    Ensures users can adjust retry behavior for specific calls, which is useful
    for handling different scenarios (e.g., interactive vs batch processing).
    """
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        max_retries=3,  # Default
    )

    messages: list[BaseMessage] = [HumanMessage(content="test")]

    # Test with call-level override
    request = chat._prepare_request(messages, max_retries=10)
    config = request["config"]

    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.attempts == 10, (
        "Call-level max_retries should override instance default"
    )


def test_default_max_retries_value() -> None:
    """Test that `max_retries` has a sensible default value."""
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        # Don't specify max_retries - use default
    )

    messages: list[BaseMessage] = [HumanMessage(content="test")]
    request = chat._prepare_request(messages)
    config = request["config"]

    # Verify a reasonable default is set (should be >= 2 for transient errors)
    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.attempts >= 2, (
        "Default max_retries should handle transient failures"
    )


def test_client_error_with_500_raises_descriptive_error() -> None:
    """Test that 500 INTERNAL errors are properly reported when retries are exhausted.

    This ensures that after the SDK exhausts all retry attempts, the error message
    is clear and informative for debugging.
    """
    mock_client = Mock()
    mock_models = Mock()
    mock_generate_content = Mock()

    # Simulate persistent 500 error (all retries exhausted)
    mock_generate_content.side_effect = ClientError(
        code=500,
        response_json={
            "error": {
                "message": (
                    "Unable to submit request because the service is temporarily "
                    "unavailable."
                ),
                "status": "INTERNAL",
            }
        },
        response=None,
    )

    mock_models.generate_content = mock_generate_content
    mock_client.return_value.models = mock_models

    with patch("langchain_google_genai.chat_models.Client", mock_client):
        chat = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr(FAKE_API_KEY),
            max_retries=1,  # Minimize test time
        )

        # Should raise descriptive error after retries exhausted
        with pytest.raises(
            ChatGoogleGenerativeAIError,
            match=r"Error calling model .* \(INTERNAL\).*temporarily unavailable",
        ):
            chat.invoke("test message")


def test_finish_reason_as_integer() -> None:
    """Test that finish_reason as an integer doesn't crash.

    Reproduces issue #1232 where the API returns finish_reason as a raw integer
    (e.g., 15 for unknown enum values) instead of an enum object with .name attribute.

    This can happen when the server returns a new/unknown finish reason value that
    isn't in the local enum definition yet.
    """
    # Create a mock candidate where finish_reason is a raw integer
    mock_candidate = Mock(spec=Candidate)
    mock_candidate.finish_reason = 15  # Raw integer, not an enum
    mock_candidate.safety_ratings = []
    mock_candidate.content = Content(
        parts=[Part.from_text(text="Test response")],
        role="model",
    )

    # Create a mock response with the candidate
    mock_response = Mock(spec=GenerateContentResponse)
    mock_response.candidates = [mock_candidate]
    mock_response.model_version = "gemini-2.5-flash-image"
    mock_response.usage_metadata = None
    mock_response.prompt_feedback = None

    # This should not raise an error, and should convert the integer to a string
    result = _response_to_result(
        mock_response,
        stream=False,
    )
    assert result.generations[0].generation_info["finish_reason"] == "UNKNOWN_15"  # type: ignore[index]


def test_image_config_in_init() -> None:
    """Test that `image_config` is properly initialized."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "16:9", "image_size": "2K"},
    )
    assert llm.image_config == {"aspect_ratio": "16:9", "image_size": "2K"}


def test_image_config_in_identifying_params() -> None:
    """Test that `image_config` is included in identifying parameters."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "16:9", "image_size": "2K"},
    )
    params = llm._identifying_params
    assert "image_config" in params
    assert params["image_config"] == {"aspect_ratio": "16:9", "image_size": "2K"}


def test_image_config_passed_to_generate_content_config() -> None:
    """Test that `image_config` is properly passed to `GenerateContentConfig`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "16:9", "image_size": "2K"},
    )
    messages: list[BaseMessage] = [HumanMessage(content="Generate an image of a cat")]
    request = llm._prepare_request(messages)
    config = request["config"]

    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "16:9"
    assert config.image_config.image_size == "2K"


def test_image_config_none_by_default() -> None:
    """Test that `image_config` is `None` by default."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )
    assert llm.image_config is None

    messages: list[BaseMessage] = [HumanMessage(content="Hello")]
    request = llm._prepare_request(messages)
    config = request["config"]

    assert config.image_config is None


def test_image_config_override_in_invoke() -> None:
    """Test that `image_config` can be overridden in `invoke()`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "1:1"},
    )

    # Override with different config
    messages: list[BaseMessage] = [HumanMessage(content="Generate an image")]
    request = llm._prepare_request(
        messages, image_config={"aspect_ratio": "16:9", "image_size": "2K"}
    )
    config = request["config"]

    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "16:9"
    assert config.image_config.image_size == "2K"


def test_image_config_override_with_none() -> None:
    """Test that `image_config` defaults to instance config when `None` is passed."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "1:1"},
    )

    # Don't pass image_config parameter (defaults to None)
    messages: list[BaseMessage] = [HumanMessage(content="Generate an image")]
    request = llm._prepare_request(messages)
    config = request["config"]

    # Should use instance-level config
    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "1:1"


def test_image_config_override_with_empty_dict() -> None:
    """Test that passing empty `dict` creates empty `ImageConfig`."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
        image_config={"aspect_ratio": "1:1"},
    )

    # Pass empty dict - should create ImageConfig with no fields set
    messages: list[BaseMessage] = [HumanMessage(content="Generate an image")]
    request = llm._prepare_request(messages, image_config={})
    config = request["config"]

    # Empty dict creates an ImageConfig with all fields None
    assert config.image_config is not None
    assert config.image_config.aspect_ratio is None
    assert config.image_config.image_size is None


def test_image_config_no_instance_config_with_override() -> None:
    """Test that `image_config` works when not set at instance level."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr(FAKE_API_KEY),
    )

    # No instance config, but override in invoke
    messages: list[BaseMessage] = [HumanMessage(content="Generate an image")]
    request = llm._prepare_request(messages, image_config={"aspect_ratio": "16:9"})
    config = request["config"]

    assert config.image_config is not None
    assert config.image_config.aspect_ratio == "16:9"
