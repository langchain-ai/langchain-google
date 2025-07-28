"""Test chat model integration."""

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from unittest.mock import ANY, Mock, patch

import google.ai.generativelanguage as glm
import pytest
from google.ai.generativelanguage_v1beta.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from pydantic import SecretStr
from pydantic_core._pydantic_core import ValidationError
from pytest import CaptureFixture

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _convert_tool_message_to_parts,
    _parse_chat_history,
    _parse_response_candidate,
    _response_to_result,
)


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
        top_k=2,
        top_p=1,
        temperature=0.7,
        n=2,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": "gemini-nano",
        "ls_model_type": "chat",
        "ls_temperature": 0.7,
    }

    llm = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
        max_output_tokens=10,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": "gemini-nano",
        "ls_model_type": "chat",
        "ls_temperature": 0.7,
        "ls_max_tokens": 10,
    }

    ChatGoogleGenerativeAI(
        model="gemini-nano",
        api_key=SecretStr("..."),
        top_k=2,
        top_p=1,
        temperature=0.7,
    )

    # test initialization with an invalid argument to check warning
    with patch("langchain_google_genai.chat_models.logger.warning") as mock_warning:
        llm = ChatGoogleGenerativeAI(
            model="gemini-nano",
            google_api_key=SecretStr("..."),  # type: ignore[call-arg]
            safety_setting={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
            },  # Invalid arg
        )
        assert llm.model == "models/gemini-nano"
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


def test_initialization_inside_threadpool() -> None:
    # new threads don't have a running event loop,
    # thread pool executor easiest way to create one
    with ThreadPoolExecutor() as executor:
        executor.submit(
            ChatGoogleGenerativeAI,
            model="gemini-nano",
            google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
        ).result()


def test_initalization_without_async() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
    )
    assert chat.async_client is None


def test_initialization_with_async() -> None:
    async def initialize_chat_with_async_client() -> ChatGoogleGenerativeAI:
        model = ChatGoogleGenerativeAI(
            model="gemini-nano",
            google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
        )
        _ = model.async_client
        return model

    loop = asyncio.get_event_loop()
    chat = loop.run_until_complete(initialize_chat_with_async_client())
    assert chat.async_client is not None


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
    )
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
    )
    print(chat.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.parametrize("convert_system_message_to_human", [False, True])
def test_parse_history(convert_system_message_to_human: bool) -> None:
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
    system_instruction, history = _parse_chat_history(
        messages, convert_system_message_to_human=convert_system_message_to_human
    )
    assert len(history) == 8
    if convert_system_message_to_human:
        assert history[0] == glm.Content(
            role="user",
            parts=[glm.Part(text=system_input), glm.Part(text=text_question1)],
        )
    else:
        assert history[0] == glm.Content(
            role="user", parts=[glm.Part(text=text_question1)]
        )
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
def test_parse_function_history(content: Union[str, List[Union[str, Dict]]]) -> None:
    function_message = FunctionMessage(name="search_tool", content=content)
    _parse_chat_history([function_message], convert_system_message_to_human=True)


@pytest.mark.parametrize(
    "headers", (None, {}, {"X-User-Header": "Coco", "X-User-Header2": "Jamboo"})
)
def test_additional_headers_support(headers: Optional[Dict[str, str]]) -> None:
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
            model="gemini-pro",
            google_api_key=param_secret_api_key,  # type: ignore[call-arg]
            client_options=param_client_options,
            transport=param_transport,
            additional_headers=headers,
        )

    expected_default_metadata: tuple = ()
    if not headers:
        assert chat.additional_headers == headers
    else:
        assert chat.additional_headers
        assert all(header in chat.additional_headers for header in headers.keys())
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


@pytest.mark.parametrize(
    "raw_candidate, expected",
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
                            + "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
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
                            + "Qk0eAAAAAAAAABoAAAAMAAAAAQABAAEAGAAAAP8A"
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
def test_parse_response_candidate(raw_candidate: Dict, expected: AIMessage) -> None:
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


def test_serialize() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-pro-1.5", google_api_key="test-key")  # type: ignore[call-arg]
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
    """Test that temperature is in the range [0.0, 2.0]"""

    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=2.1)

    with pytest.raises(ValidationError):
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=-0.1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
        temperature=1.5,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": "gemini-2.0-flash",
        "ls_model_type": "chat",
        "ls_temperature": 1.5,
    }


def test_temperature_range_model_validation() -> None:
    """Test that temperature is in the range [0.0, 2.0]"""

    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=2.5)

    with pytest.raises(ValueError):
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=-0.5)


def test_model_kwargs() -> None:
    """Test we can transfer unknown params to model_kwargs."""
    llm = ChatGoogleGenerativeAI(
        model="my-model",
        convert_system_message_to_human=True,
        model_kwargs={"foo": "bar"},
    )
    assert llm.model == "models/my-model"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatGoogleGenerativeAI(
            model="my-model",
            convert_system_message_to_human=True,
            foo="bar",
        )
    assert llm.model == "models/my-model"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.parametrize(
    "raw_response, expected_grounding_metadata",
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
    raw_response: Dict, expected_grounding_metadata: Dict
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
