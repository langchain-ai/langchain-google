"""Test chat model integration."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Type, Union
from unittest.mock import ANY, Mock, patch

import google.ai.generativelanguage as glm
import pytest
from google.ai.generativelanguage_v1beta.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumps, loads
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.pydantic_v1 import SecretStr
from langchain_standard_tests.unit_tests import ChatModelUnitTests
from pytest import CaptureFixture

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _parse_chat_history,
    _parse_response_candidate,
)


class GoogleGenerativeAIStandardTests(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGoogleGenerativeAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gemini-1.5-pro"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-nano",
        api_key=SecretStr("..."),
        top_k=2,
        top_p=1,
        temperature=0.7,
        n=2,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": "models/gemini-nano",
        "ls_model_type": "chat",
        "ls_temperature": 0.7,
    }

    llm = ChatGoogleGenerativeAI(
        model="gemini-nano",
        api_key=SecretStr("..."),
        max_output_tokens=10,
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_name": "models/gemini-nano",
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


def test_initialization_inside_threadpool() -> None:
    # new threads don't have a running event loop,
    # thread pool executor easiest way to create one
    with ThreadPoolExecutor() as executor:
        executor.submit(
            ChatGoogleGenerativeAI,
            model="gemini-nano",
            api_key=SecretStr("secret-api-key"),
        ).result()


def test_initalization_without_async() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano", api_key=SecretStr("secret-api-key")
    )
    assert chat.async_client is None


def test_initialization_with_async() -> None:
    async def initialize_chat_with_async_client() -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-nano", api_key=SecretStr("secret-api-key")
        )

    loop = asyncio.get_event_loop()
    chat = loop.run_until_complete(initialize_chat_with_async_client())
    assert chat.async_client is not None


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano", api_key=SecretStr("secret-api-key")
    )
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano", api_key=SecretStr("secret-api-key")
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
        "arguments": json.dumps({"arg1": "2", "arg2": "2", "op": "+"}),
    }
    function_answer1 = json.dumps({"result": 4})
    function_call_2 = {
        "name": function_name,
        "arguments": json.dumps({"arg1": "2", "arg2": "2", "op": "*"}),
    }
    function_answer2 = json.dumps({"result": 4})
    text_answer1 = "They are same"

    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(
        content="",
        additional_kwargs={
            "function_call": function_call_1,
        },
    )
    message3 = ToolMessage(
        name="calculator", content=function_answer1, tool_call_id="1"
    )
    message4 = AIMessage(
        content="",
        additional_kwargs={
            "function_call": function_call_2,
        },
    )
    message5 = FunctionMessage(name="calculator", content=function_answer2)
    message6 = AIMessage(content=text_answer1)
    messages = [
        system_message,
        message1,
        message2,
        message3,
        message4,
        message5,
        message6,
    ]
    system_instruction, history = _parse_chat_history(
        messages, convert_system_message_to_human=convert_system_message_to_human
    )
    assert len(history) == 6
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
                        "args": json.loads(function_call_1["arguments"]),
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
    assert history[5] == glm.Content(role="model", parts=[glm.Part(text=text_answer1)])
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
            api_key=param_secret_api_key,
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
    llm = ChatGoogleGenerativeAI(model="gemini-pro-1.5", google_api_key="test-key")
    serialized = dumps(llm)
    llm_loaded = loads(
        serialized,
        secrets_map={"GOOGLE_API_KEY": "test-key"},
        valid_namespaces=["langchain_google_genai"],
    )
    assert llm == llm_loaded
