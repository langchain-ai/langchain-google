"""Test chat model integration."""

import asyncio
import base64
import json
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
from unittest.mock import ANY, Mock, patch

import google.ai.generativelanguage as glm
import pytest
from google.ai.generativelanguage_v1beta.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)
from google.api_core.exceptions import ResourceExhausted
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.block_translators.google_genai import (
    _convert_to_v1_from_genai,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import SecretStr
from pydantic_core._pydantic_core import ValidationError

from langchain_google_genai import HarmBlockThreshold, HarmCategory, Modality
from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _chat_with_retry,
    _convert_tool_message_to_parts,
    _parse_chat_history,
    _parse_response_candidate,
    _response_to_result,
)

MODEL_NAME = "gemini-flash-lite-latest"


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("..."),
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
        google_api_key=SecretStr("..."),
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
        api_key=SecretStr("..."),
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
                google_api_key=SecretStr("..."),
                safety_setting={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
                },  # Invalid arg
            )
        assert llm.model == f"models/{MODEL_NAME}"
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


def test_safety_settings_initialization() -> None:
    """Test chat model initialization with safety_settings parameter."""
    safety_settings: dict[HarmCategory, HarmBlockThreshold] = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE  # type: ignore[dict-item]
    }

    # Test initialization with safety_settings
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("test-key"),
        temperature=0.7,
        safety_settings=safety_settings,
    )

    # Verify the safety_settings are stored correctly
    assert llm.safety_settings == safety_settings
    assert llm.temperature == 0.7
    assert llm.model == f"models/{MODEL_NAME}"


def test_initialization_inside_threadpool() -> None:
    # new threads don't have a running event loop,
    # thread pool executor easiest way to create one
    with ThreadPoolExecutor() as executor:
        executor.submit(
            ChatGoogleGenerativeAI,
            model=MODEL_NAME,
            google_api_key=SecretStr("secret-api-key"),
        ).result()


def test_client_transport() -> None:
    """Test client transport configuration."""
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key="fake-key")
    assert model.client.transport.kind == "grpc"

    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME, google_api_key="fake-key", transport="rest"
    )
    assert model.client.transport.kind == "rest"

    async def check_async_client() -> None:
        model = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key="fake-key")
        assert model.async_client.transport.kind == "grpc_asyncio"

        # Test auto conversion of transport to "grpc_asyncio" from "rest"
        model = ChatGoogleGenerativeAI(
            model=MODEL_NAME, google_api_key="fake-key", transport="rest"
        )
        assert model.async_client.transport.kind == "grpc_asyncio"

    asyncio.run(check_async_client())


def test_initalization_without_async() -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("secret-api-key"),
    )
    assert chat.async_client is None


def test_initialization_with_async() -> None:
    async def initialize_chat_with_async_client() -> ChatGoogleGenerativeAI:
        model = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr("secret-api-key"),
        )
        _ = model.async_client
        return model

    chat = asyncio.run(initialize_chat_with_async_client())
    assert chat.async_client is not None


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("secret-api-key"),
    )
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: pytest.CaptureFixture,
) -> None:
    chat = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("secret-api-key"),
    )
    print(chat.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


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
    assert history[0] == glm.Content(role="user", parts=[glm.Part(text=text_question1)])
    assert history[1] == glm.Content(
        role="model",
        parts=[
            glm.Part(
                function_call=glm.FunctionCall(
                    {
                        "name": "calculator",
                        "args": function_call_1["args"],
                    }
                )
            )
        ],
    )
    assert history[2] == glm.Content(
        role="user",
        parts=[
            glm.Part(
                function_response=glm.FunctionResponse(
                    {
                        "name": "calculator",
                        "response": {"result": 4},
                    }
                )
            )
        ],
    )
    assert history[3] == glm.Content(
        role="model",
        parts=[
            glm.Part(
                function_call=glm.FunctionCall(
                    {
                        "name": "calculator",
                        "args": json.loads(function_call_2["arguments"]),
                    }
                )
            )
        ],
    )
    assert history[4] == glm.Content(
        role="user",
        parts=[
            glm.Part(
                function_response=glm.FunctionResponse(
                    {
                        "name": "calculator",
                        "response": {"result": 4},
                    }
                )
            )
        ],
    )
    assert history[5] == glm.Content(
        role="model",
        parts=[
            glm.Part(
                function_call=glm.FunctionCall(
                    {
                        "name": "calculator",
                        "args": function_call_3["args"],
                    }
                )
            ),
            glm.Part(
                function_call=glm.FunctionCall(
                    {
                        "name": "calculator",
                        "args": function_call_4["args"],
                    }
                )
            ),
        ],
    )
    assert history[6] == glm.Content(
        role="user",
        parts=[
            glm.Part(
                function_response=glm.FunctionResponse(
                    {
                        "name": "calculator",
                        "response": {"result": 4},
                    }
                )
            ),
            glm.Part(
                function_response=glm.FunctionResponse(
                    {
                        "name": "calculator",
                        "response": {"result": 6},
                    }
                )
            ),
        ],
    )
    assert history[7] == glm.Content(role="model", parts=[glm.Part(text=text_answer1)])
    if convert_system_message_to_human:
        assert system_instruction is None
    else:
        assert system_instruction == glm.Content(parts=[glm.Part(text=system_input)])


@pytest.mark.parametrize("content", ['["a"]', '{"a":"b"}', "function output"])
def test_parse_function_history(content: Union[str, list[Union[str, dict]]]) -> None:
    function_message = FunctionMessage(name="search_tool", content=content)
    _parse_chat_history([function_message])


@pytest.mark.parametrize(
    "headers", [None, {}, {"X-User-Header": "Coco", "X-User-Header2": "Jamboo"}]
)
def test_additional_headers_support(headers: Optional[dict[str, str]]) -> None:
    mock_client = Mock()
    mock_generate_content = Mock()
    mock_generate_content.return_value = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="test response")]))]
    )
    mock_client.return_value.generate_content = mock_generate_content
    api_endpoint = "http://127.0.0.1:8000/ai"
    param_api_key = "[secret]"
    param_secret_api_key = SecretStr(param_api_key)
    param_client_options = {"api_endpoint": api_endpoint}
    param_transport = "rest"

    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient",
        mock_client,
    ):
        chat = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            client_options=param_client_options,
            transport=param_transport,
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
        transport=param_transport,
        client_options=ANY,
        client_info=ANY,
    )
    call_client_options = mock_client.call_args_list[0].kwargs["client_options"]
    assert call_client_options.api_key == param_api_key
    assert call_client_options.api_endpoint == api_endpoint
    call_client_info = mock_client.call_args_list[0].kwargs["client_info"]
    assert "langchain-google-genai" in call_client_info.user_agent
    assert "ChatGoogleGenerativeAI" in call_client_info.user_agent


def test_default_metadata_field_alias() -> None:
    """Test 'default_metadata' and 'default_metadata_input' fields work correctly."""
    # Test with default_metadata_input field name (alias) - should accept None without
    # error
    # This is the main issue: LangSmith Playground passes None to default_metadata_input
    chat1 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("test-key"),
        default_metadata_input=None,
    )
    # When None is passed to alias, it should use the default factory and be overridden
    # by validator
    assert chat1.default_metadata == ()

    # Test with empty list for default_metadata_input (should not cause validation
    # error)
    chat2 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("test-key"),
        default_metadata_input=[],
    )
    # Empty list should be accepted and overridden by validator
    assert chat2.default_metadata == ()

    # Test with tuple for default_metadata_input (should not cause validation error)
    chat3 = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("test-key"),
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
                            "function_call": glm.FunctionCall(
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
                            "function_call": glm.FunctionCall(
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
                            "function_call": glm.FunctionCall(
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
                            "function_call": glm.FunctionCall(
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
                            "function_call": glm.FunctionCall(
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
                            "function_call": glm.FunctionCall(
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
        response_candidate = glm.Candidate(raw_candidate)
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
        "finish_reason": 1,
        "safety_ratings": [],
    }

    response_candidate = glm.Candidate(raw_candidate)
    result = _parse_response_candidate(response_candidate)

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"

    # Streaming
    result = _parse_response_candidate(response_candidate, streaming=True)

    assert hasattr(result, "response_metadata")
    assert result.response_metadata["model_provider"] == "google_genai"


def test_parse_response_candidate_includes_model_name() -> None:
    """Test that _parse_response_candidate includes model_name in response_metadata."""
    raw_candidate = {
        "content": {"parts": [{"text": "Hello, world!"}]},
        "finish_reason": 1,
        "safety_ratings": [],
    }

    response_candidate = glm.Candidate(raw_candidate)
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
    assert part.function_response.name == "tool_name"
    assert part.function_response.response == {"output": "test_content"}


def test_temperature_range_pydantic_validation() -> None:
    """Test that temperature is in the range [0.0, 2.0]."""
    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=2.1)

    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=-0.1)

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=SecretStr("..."),
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
    """Test that temperature is in the range [0.0, 2.0]."""
    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=2.5)

    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=-0.5)


def test_model_kwargs() -> None:
    """Test we can transfer unknown params to model_kwargs."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        convert_system_message_to_human=True,
        model_kwargs={"foo": "bar"},
    )
    assert llm.model == f"models/{MODEL_NAME}"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            convert_system_message_to_human=True,
            foo="bar",
        )
    assert llm.model == f"models/{MODEL_NAME}"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}


def test_retry_decorator_with_custom_parameters() -> None:
    # Mock the generation method
    mock_generation_method = Mock()
    # TODO: remove ignore once google-auth has types.
    mock_generation_method.side_effect = ResourceExhausted("Quota exceeded")  # type: ignore[no-untyped-call]

    # Call the function with custom retry parameters
    with pytest.raises(ResourceExhausted):
        _chat_with_retry(
            generation_method=mock_generation_method,
            max_retries=3,
            wait_exponential_multiplier=1.5,
            wait_exponential_min=2.0,
            wait_exponential_max=30.0,
        )

    # Verify that the retry mechanism used the custom parameters
    assert mock_generation_method.call_count == 3


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
                "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
                "usage_metadata": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 5,
                    "total_token_count": 15,
                },
            },
            {
                "grounding_chunks": [
                    {"web": {"uri": "https://example.com", "title": "Example Site"}}
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
        ),
        (
            # Case 2: Response without grounding_metadata
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Test response"}]},
                    }
                ],
                "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
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
    """Test that _response_to_result includes grounding_metadata in the response."""
    response = GenerateContentResponse(raw_response)
    result = _response_to_result(response, stream=False)

    assert len(result.generations) == len(raw_response["candidates"])
    for generation in result.generations:
        assert generation.message.content == "Test response"
        grounding_metadata = (
            generation.generation_info.get("grounding_metadata", {})
            if generation.generation_info is not None
            else {}
        )
        assert grounding_metadata == expected_grounding_metadata


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
        "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "total_token_count": 30,
        },
    }

    response = GenerateContentResponse(raw_response)
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
        annotations = block.get("annotations", [])
        citations = [ann for ann in annotations if ann.get("type") == "citation"]  # type: ignore[attr-defined]
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
        "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
        "usage_metadata": {
            "prompt_token_count": 5,
            "candidates_token_count": 8,
            "total_token_count": 13,
        },
    }

    response = GenerateContentResponse(raw_response)
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
        "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
        "usage_metadata": {
            "prompt_token_count": 5,
            "candidates_token_count": 3,
            "total_token_count": 8,
        },
    }

    response = GenerateContentResponse(raw_response)
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
        "prompt_feedback": {"block_reason": 0, "safety_ratings": []},
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 10,
            "total_token_count": 20,
        },
    }

    response = GenerateContentResponse(raw_response)
    result = _response_to_result(response, stream=False)

    message = result.generations[0].message

    # Verify grounding metadata is present
    assert "grounding_metadata" in message.response_metadata
    grounding = message.response_metadata["grounding_metadata"]
    # grounding metadata from proto.Message.to_dict() uses snake_case
    assert len(grounding["grounding_supports"]) == 1
    assert grounding["grounding_supports"][0]["segment"]["part_index"] == 1


@pytest.mark.parametrize(
    "is_async,mock_target,method_name",
    [
        (False, "_chat_with_retry", "_generate"),  # Sync
        (True, "_achat_with_retry", "_agenerate"),  # Async
    ],
)
@pytest.mark.parametrize(
    "instance_timeout,call_timeout,expected_timeout,should_have_timeout",
    [
        (5.0, None, 5.0, True),  # Instance-level timeout
        (5.0, 10.0, 10.0, True),  # Call-level overrides instance
        (None, None, None, False),  # No timeout anywhere
    ],
)
async def test_timeout_parameter_handling(
    is_async: bool,
    mock_target: str,
    method_name: str,
    instance_timeout: Optional[float],
    call_timeout: Optional[float],
    expected_timeout: Optional[float],
    should_have_timeout: bool,
) -> None:
    """Test timeout parameter handling for sync and async methods."""
    with patch(f"langchain_google_genai.chat_models.{mock_target}") as mock_retry:
        mock_retry.return_value = GenerateContentResponse(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Test response"}]},
                        "finish_reason": "STOP",
                    }
                ]
            }
        )

        # Create LLM with optional instance-level timeout
        llm_kwargs = {
            "model": "gemini-2.5-flash",
            "google_api_key": SecretStr("test-key"),
        }
        if instance_timeout is not None:
            llm_kwargs["timeout"] = instance_timeout

        llm = ChatGoogleGenerativeAI(**llm_kwargs)
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]

        # Call the appropriate method with optional call-level timeout
        method = getattr(llm, method_name)
        call_kwargs = {}
        if call_timeout is not None:
            call_kwargs["timeout"] = call_timeout

        if is_async:
            await method(messages, **call_kwargs)
        else:
            method(messages, **call_kwargs)

        # Verify timeout was passed correctly
        mock_retry.assert_called_once()
        call_kwargs_actual = mock_retry.call_args[1]

        if should_have_timeout:
            assert "timeout" in call_kwargs_actual
            assert call_kwargs_actual["timeout"] == expected_timeout
        else:
            assert "timeout" not in call_kwargs_actual


@pytest.mark.parametrize(
    "instance_timeout,expected_timeout,should_have_timeout",
    [
        (5.0, 5.0, True),  # Instance-level timeout
        (None, None, False),  # No timeout
    ],
)
@patch("langchain_google_genai.chat_models._chat_with_retry")
def test_timeout_streaming_parameter_handling(
    mock_retry: Mock,
    instance_timeout: Optional[float],
    expected_timeout: Optional[float],
    should_have_timeout: bool,
) -> None:
    """Test timeout parameter handling for streaming methods."""

    # Mock the return value for _chat_with_retry to return an iterator
    def mock_stream() -> Iterator[GenerateContentResponse]:
        yield GenerateContentResponse(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "chunk1"}]},
                        "finish_reason": "STOP",
                    }
                ]
            }
        )

    mock_retry.return_value = mock_stream()

    # Create LLM with optional instance-level timeout
    llm_kwargs = {
        "model": "gemini-2.5-flash",
        "google_api_key": SecretStr("test-key"),
    }
    if instance_timeout is not None:
        llm_kwargs["timeout"] = instance_timeout

    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    # Call _stream (which should pass timeout to _chat_with_retry)
    messages: list[BaseMessage] = [HumanMessage(content="Hello")]
    list(llm._stream(messages))  # Convert generator to list to trigger execution

    # Verify timeout was passed correctly
    mock_retry.assert_called_once()
    call_kwargs = mock_retry.call_args[1]

    if should_have_timeout:
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == expected_timeout
    else:
        assert "timeout" not in call_kwargs


@pytest.mark.parametrize(
    "is_async,mock_target,method_name",
    [
        (False, "_chat_with_retry", "_generate"),  # Sync
        (True, "_achat_with_retry", "_agenerate"),  # Async
    ],
)
@pytest.mark.parametrize(
    "instance_max_retries,call_max_retries,expected_max_retries,should_have_max_retries",
    [
        (1, None, 1, True),  # Instance-level max_retries
        (3, 5, 5, True),  # Call-level overrides instance
        (6, None, 6, True),  # Default instance value
    ],
)
async def test_max_retries_parameter_handling(
    is_async: bool,
    mock_target: str,
    method_name: str,
    instance_max_retries: int,
    call_max_retries: Optional[int],
    expected_max_retries: int,
    should_have_max_retries: bool,
) -> None:
    """Test max_retries parameter handling for sync and async methods."""
    with patch(f"langchain_google_genai.chat_models.{mock_target}") as mock_retry:
        mock_retry.return_value = GenerateContentResponse(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Test response"}]},
                        "finish_reason": "STOP",
                    }
                ]
            }
        )

        # Instance-level max_retries
        llm_kwargs = {
            "model": "gemini-2.5-flash",
            "google_api_key": SecretStr("test-key"),
            "max_retries": instance_max_retries,
        }

        llm = ChatGoogleGenerativeAI(**llm_kwargs)
        messages: list[BaseMessage] = [HumanMessage(content="Hello")]

        # Call the appropriate method with optional call-level max_retries
        method = getattr(llm, method_name)
        call_kwargs = {}
        if call_max_retries is not None:
            call_kwargs["max_retries"] = call_max_retries

        if is_async:
            await method(messages, **call_kwargs)
        else:
            method(messages, **call_kwargs)

        # Verify max_retries was passed correctly
        mock_retry.assert_called_once()
        call_kwargs_actual = mock_retry.call_args[1]

        if should_have_max_retries:
            assert "max_retries" in call_kwargs_actual
            assert call_kwargs_actual["max_retries"] == expected_max_retries
        else:
            assert "max_retries" not in call_kwargs_actual


def test_thinking_config_merging_with_generation_config() -> None:
    """Test that thinking_config is properly merged when passed in generation_config."""
    with patch("langchain_google_genai.chat_models._chat_with_retry") as mock_retry:
        # Mock response with thinking content followed by regular text
        mock_response = GenerateContentResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                Part(text="Let me think about this...", thought=True),
                                Part(text="There are 2 O's in Google."),
                            ]
                        },
                        "finish_reason": "STOP",
                    }
                ],
                "usage_metadata": {
                    "prompt_token_count": 20,
                    "candidates_token_count": 15,
                    "total_token_count": 35,
                    "cached_content_token_count": 0,
                },
            }
        )
        mock_retry.return_value = mock_response

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=SecretStr("test-key"),
        )

        result = llm.invoke(
            "How many O's are in Google?",
            generation_config={"thinking_config": {"include_thoughts": True}},
        )

        # Verify the call was made with merged config
        mock_retry.assert_called_once()
        call_args = mock_retry.call_args
        request = call_args.kwargs["request"]
        assert hasattr(request, "generation_config")
        assert hasattr(request.generation_config, "thinking_config")
        assert request.generation_config.thinking_config.include_thoughts is True

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


def test_with_structured_output_json_schema_alias() -> None:
    """Test that json_schema method works as alias for json_mode."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        age: int

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="fake-key")

    structured_llm = llm.with_structured_output(TestModel, method="json_schema")
    assert structured_llm is not None

    schema_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
    structured_llm_dict = llm.with_structured_output(schema_dict, method="json_schema")
    assert structured_llm_dict is not None


def test_modalities_override_in_generation_config() -> None:
    """Test response modalities in invoke generation_config override model-defined."""
    from langchain_google_genai import Modality

    # Mock response with both image and text content
    mock_response = Mock()
    mock_response.candidates = [
        Candidate(
            content=Content(
                parts=[
                    Part(
                        inline_data=glm.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(
                                "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                            ),
                        )
                    ),
                    Part(text="Meow! Here's a cat image for you."),
                ]
            ),
            finish_reason=Candidate.FinishReason.STOP,
        )
    ]
    # Create proper usage metadata using dict approach
    from google.ai.generativelanguage_v1beta.types import UsageMetadata

    mock_response.usage_metadata = UsageMetadata(
        {
            "prompt_token_count": 10,
            "response_token_count": 5,
            "total_token_count": 15,
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp-image-generation",
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
    """Test generating an image with mocked response and content_blocks translation."""
    mock_response = GenerateContentResponse(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Meow!"},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64.b64decode(
                                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
                                        "FcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                                    ),
                                }
                            },
                        ]
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=SecretStr("test-key"),
    )

    with patch.object(llm.client, "generate_content", return_value=mock_response):
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
    """Test generating audio with mocked response and content_blocks translation."""
    mock_response = GenerateContentResponse(
        {
            "candidates": [
                {
                    # Empty content when audio is in additional_kwargs
                    "content": {"parts": []},
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }
    )

    wav_bytes = (  # (minimal WAV header)
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-tts",
        google_api_key=SecretStr("test-key"),
        response_modalities=[Modality.AUDIO],
    )

    with patch.object(llm.client, "generate_content", return_value=mock_response):
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
