"""Test chat model integration."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from unittest.mock import ANY, Mock, patch

import pytest
from google.api_core.exceptions import ResourceExhausted
from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    FunctionResponse,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    Language,
    Part,
)
from google.genai.types import (
    Outcome as CodeExecutionResultOutcome,
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

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _chat_with_retry,
    _convert_to_parts,
    _convert_tool_message_to_parts,
    _get_ai_message_tool_messages_parts,
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


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
    )
    assert isinstance(chat.google_api_key, SecretStr)


def test_base_url_set_in_constructor() -> None:
    chat = ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
        base_url="http://localhost:8000",
    )
    assert chat.base_url == "http://localhost:8000"


def test_base_url_passed_to_client() -> None:
    with patch("langchain_google_genai.chat_models.Client") as mock_client:
        ChatGoogleGenerativeAI(
            model="gemini-nano",
            google_api_key=SecretStr("secret-api-key"),  # type: ignore[call-arg]
            base_url="http://localhost:8000",
        )
        mock_client.assert_called_once_with(
            api_key="secret-api-key",
            http_options=HttpOptions(base_url="http://localhost:8000", headers={}),
        )


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
        assert history[0] == Content(
            role="user",
            parts=[Part(text=system_input), Part(text=text_question1)],
        )
    else:
        assert history[0] == Content(role="user", parts=[Part(text=text_question1)])
    assert history[1] == Content(
        role="model",
        parts=[
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=function_call_1["args"],  # type: ignore[arg-type]
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
                    args=function_call_3["args"],  # type: ignore[arg-type]
                )
            ),
            Part(
                function_call=FunctionCall(
                    name="calculator",
                    args=function_call_4["args"],  # type: ignore[arg-type]
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
def test_parse_function_history(content: Union[str, List[Union[str, Dict]]]) -> None:
    function_message = FunctionMessage(name="search_tool", content=content)
    _parse_chat_history([function_message], convert_system_message_to_human=True)


@pytest.mark.parametrize(
    "headers", (None, {}, {"X-User-Header": "Coco", "X-User-Header2": "Jamboo"})
)
def test_additional_headers_support(headers: Optional[Dict[str, str]]) -> None:
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
    param_api_key = "[secret]"
    param_secret_api_key = SecretStr(param_api_key)
    param_client_options = {"api_endpoint": api_endpoint}
    param_transport = "rest"

    with patch(
        "langchain_google_genai.chat_models.Client",
        mock_client,
    ):
        chat = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=param_secret_api_key,  # type: ignore[call-arg]
            base_url=api_endpoint,
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
        api_key=param_api_key,
        http_options=ANY,
    )
    call_http_options = mock_client.call_args_list[0].kwargs["http_options"]
    assert call_http_options.base_url == api_endpoint
    if headers:
        assert call_http_options.headers == headers
    else:
        assert call_http_options.headers == {}


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
                                name="Information",
                                args={"name": "Ben"},
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
                                name="Information",
                                args={"name": "Ben"},
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
    assert part.function_response is not None
    assert part.function_response.name == "tool_name"
    assert part.function_response.response == {"output": "test_content"}


def test_supports_thinking() -> None:
    """Test that _supports_thinking correctly identifies model capabilities."""
    # Test models that don't support thinking
    llm_image_gen = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-preview-image-generation",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
    )
    assert not llm_image_gen._supports_thinking()

    llm_tts = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-tts",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
    )
    assert not llm_tts._supports_thinking()

    # Test models that do support thinking
    llm_normal = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
    )
    assert llm_normal._supports_thinking()

    llm_15 = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        google_api_key=SecretStr("..."),  # type: ignore[call-arg]
    )
    assert llm_15._supports_thinking()


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


@patch("langchain_google_genai.chat_models.Client")
def test_model_kwargs(mock_client: Mock) -> None:
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


def test_retry_decorator_with_custom_parameters() -> None:
    # Mock the generation method
    mock_generation_method = Mock()
    mock_generation_method.side_effect = ResourceExhausted("Quota exceeded")

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
                "grounding_chunks": [
                    {
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
                            "part_index": None,
                        },
                        "grounding_chunk_indices": [0],
                        "confidence_scores": [0.95],
                    }
                ],
                "retrieval_metadata": None,
                "retrieval_queries": None,
                "search_entry_point": None,
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
    raw_response: Dict, expected_grounding_metadata: Dict
) -> None:
    """Test that _response_to_result includes grounding_metadata in the response."""
    response = GenerateContentResponse.model_validate(raw_response)
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


def test_convert_to_parts_text_only() -> None:
    """Test _convert_to_parts with text content."""
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
    """Test _convert_to_parts with text content blocks."""
    content = [{"type": "text", "text": "Hello, world!"}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].text == "Hello, world!"


def test_convert_to_parts_image_url() -> None:
    """Test _convert_to_parts with image_url content blocks."""
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
            },
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "image/jpeg"


def test_convert_to_parts_image_url_string() -> None:
    """Test _convert_to_parts with image_url as string."""
    content = [
        {
            "type": "image_url",
            "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k=",
        }
    ]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "image/jpeg"


def test_convert_to_parts_file_data_url() -> None:
    """Test _convert_to_parts with file data URL."""
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
    """Test _convert_to_parts with file data base64."""
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
    """Test _convert_to_parts with auto-detected mime type."""
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


def test_convert_to_parts_media_with_data() -> None:
    """Test _convert_to_parts with media type containing data."""
    content = [{"type": "media", "mime_type": "video/mp4", "data": b"fake_video_data"}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].inline_data is not None
    assert result[0].inline_data.mime_type == "video/mp4"
    assert result[0].inline_data.data == b"fake_video_data"


def test_convert_to_parts_media_with_file_uri() -> None:
    """Test _convert_to_parts with media type containing file_uri."""
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
    """Test _convert_to_parts with media type containing video metadata."""
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
    """Test _convert_to_parts with executable code."""
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
    """Test _convert_to_parts with code execution result."""
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
    """Test _convert_to_parts with code execution result without outcome (backward compatibility)."""
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
    """Test _convert_to_parts with thinking content."""
    content = [{"type": "thinking", "thinking": "I need to think about this..."}]
    result = _convert_to_parts(content)
    assert len(result) == 1
    assert result[0].text == "I need to think about this..."
    assert result[0].thought is True


def test_convert_to_parts_mixed_content() -> None:
    """Test _convert_to_parts with mixed content types."""
    content = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
            },
        },
    ]
    result = _convert_to_parts(content)
    assert len(result) == 3
    assert result[0].text == "Hello"
    assert result[1].text == "World"
    assert result[2].inline_data is not None


def test_convert_to_parts_invalid_type() -> None:
    """Test _convert_to_parts with invalid source_type."""
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
    """Test _convert_to_parts with invalid source_type."""
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
    """Test _convert_to_parts with invalid image_url format."""
    content = [{"type": "image_url", "image_url": {"invalid_key": "value"}}]
    with pytest.raises(ValueError, match="Unrecognized message image format"):
        _convert_to_parts(content)


def test_convert_to_parts_missing_mime_type_in_media() -> None:
    """Test _convert_to_parts with missing mime_type in media."""
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
    """Test _convert_to_parts with media missing both data and file_uri."""
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
    """Test _convert_to_parts with missing keys in executable_code."""
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
    """Test _convert_to_parts with missing code_execution_result key."""
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
    """Test _convert_to_parts with unrecognized type."""
    content = [{"type": "unrecognized_type", "data": "some_data"}]
    with pytest.raises(ValueError, match="Unrecognized message part type"):
        _convert_to_parts(content)


def test_convert_to_parts_non_dict_mapping() -> None:
    """Test _convert_to_parts with non-dict mapping."""
    content = [123]  # Not a string or dict
    with pytest.raises(
        Exception, match="Gemini only supports text and inline_data parts"
    ):
        _convert_to_parts(content)


def test_convert_to_parts_unrecognized_format_warning() -> None:
    """Test _convert_to_parts with unrecognized format triggers warning."""
    content = [{"some_key": "some_value"}]  # Not a recognized format
    with patch("langchain_google_genai.chat_models.logger.warning") as mock_warning:
        result = _convert_to_parts(content)
        mock_warning.assert_called_once()
        assert "Unrecognized message part format" in mock_warning.call_args[0][0]
        assert len(result) == 1
        assert result[0].text == "{'some_key': 'some_value'}"


def test_convert_tool_message_to_parts_string_content() -> None:
    """Test _convert_tool_message_to_parts with string content."""
    message = ToolMessage(name="test_tool", content="test_result", tool_call_id="123")
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_tool"
    assert result[0].function_response.response == {"output": "test_result"}


def test_convert_tool_message_to_parts_json_content() -> None:
    """Test _convert_tool_message_to_parts with JSON string content."""
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
    """Test _convert_tool_message_to_parts with dict content."""
    message = ToolMessage(
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
    """Test _convert_tool_message_to_parts with list content containing media."""
    message = ToolMessage(
        name="test_tool",
        content=[
            "Text response",
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
                },
            },
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
    assert result[1].function_response.response == {"output": [str("Text response")]}


def test_convert_tool_message_to_parts_with_name_parameter() -> None:
    """Test _convert_tool_message_to_parts with explicit name parameter."""
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
    """Test _convert_tool_message_to_parts with legacy name in additional_kwargs."""
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
    """Test _convert_tool_message_to_parts with FunctionMessage."""
    message = FunctionMessage(name="test_function", content="function_result")
    result = _convert_tool_message_to_parts(message)
    assert len(result) == 1
    assert result[0].function_response is not None
    assert result[0].function_response.name == "test_function"
    assert result[0].function_response.response == {"output": "function_result"}


def test_convert_tool_message_to_parts_invalid_json_fallback() -> None:
    """Test _convert_tool_message_to_parts with invalid JSON falls back to string."""
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
    """Test _get_ai_message_tool_messages_parts with basic tool messages."""
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
    """Test _get_ai_message_tool_messages_parts with partial tool message matches."""
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
    """Test _get_ai_message_tool_messages_parts with no matching tool messages."""
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
    """Test _get_ai_message_tool_messages_parts with empty tool calls."""
    ai_message = AIMessage(content="No tool calls")
    tool_messages = [
        ToolMessage(name="tool_1", content="result_1", tool_call_id="call_1")
    ]

    result = _get_ai_message_tool_messages_parts(tool_messages, ai_message)
    assert len(result) == 0


def test_get_ai_message_tool_messages_parts_empty_tool_messages() -> None:
    """Test _get_ai_message_tool_messages_parts with empty tool messages."""
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "tool_1", "args": {"arg1": "value1"}}],
    )

    result = _get_ai_message_tool_messages_parts([], ai_message)
    assert len(result) == 0


def test_get_ai_message_tool_messages_parts_duplicate_tool_calls() -> None:
    """Test _get_ai_message_tool_messages_parts handles duplicate tool call IDs correctly."""
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
    """Test _get_ai_message_tool_messages_parts preserves order of tool messages."""
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
    assert result[0].function_response.name == "tool_2"
    assert result[1].function_response.name == "tool_1"


def test_get_ai_message_tool_messages_parts_with_name_from_tool_call() -> None:
    """Test _get_ai_message_tool_messages_parts uses name from tool call when available."""
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
