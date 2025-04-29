"""Test chat model integration."""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from google.cloud.aiplatform_v1beta1.types import (
    Candidate,
    Content,
    FunctionCall,
    FunctionResponse,
    GenerateContentResponse,
    GenerationConfig,
    HarmCategory,
    Part,
    SafetySetting,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.output_parsers.openai_tools import (
    PydanticToolsParser,
)
from pydantic import BaseModel
from vertexai.generative_models import (  # type: ignore
    SafetySetting as VertexSafetySetting,
)
from vertexai.language_models import (  # type: ignore
    ChatMessage,
    InputOutputTextPair,
)

from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _parse_chat_history,
    _parse_chat_history_gemini,
    _parse_examples,
    _parse_response_candidate,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex


def test_init() -> None:
    for llm in [
        ChatVertexAI(
            model_name="gemini-pro",
            project="test-project",
            max_output_tokens=10,
            stop=["bar"],
            location="moon-dark1",
        ),
        ChatVertexAI(
            model="gemini-pro",
            max_tokens=10,
            stop_sequences=["bar"],
            location="moon-dark1",
            project="test-proj",
        ),
    ]:
        assert llm.model_name == "gemini-pro"
        assert llm.max_output_tokens == 10
        assert llm.stop == ["bar"]

        ls_params = llm._get_ls_params()
        assert ls_params == {
            "ls_provider": "google_vertexai",
            "ls_model_name": "gemini-pro",
            "ls_model_type": "chat",
            "ls_temperature": None,
            "ls_max_tokens": 10,
            "ls_stop": ["bar"],
        }

    # test initialization with an invalid argument to check warning
    with patch("langchain_google_vertexai.chat_models.logger.warning") as mock_warning:
        llm = ChatVertexAI(
            model_name="gemini-pro",
            project="test-project",
            safety_setting={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
            },  # Invalid arg
        )
        assert llm.model_name == "gemini-pro"
        assert llm.project == "test-project"
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


@pytest.mark.parametrize(
    "model,location",
    [
        (
            "gemini-1.0-pro-001",
            "moon-dark1",
        ),
        ("publishers/google/models/gemini-1.0-pro-001", "moon-dark2"),
    ],
)
def test_init_client(model: str, location: str) -> None:
    config = {"model": model, "location": location}
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._base.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_info = mock_prediction_service.call_args.kwargs["client_info"]
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        assert client_options.api_endpoint == f"{location}-aiplatform.googleapis.com"
        assert "langchain-google-vertexai" in client_info.user_agent
        assert "ChatVertexAI" in client_info.user_agent
        assert "langchain-google-vertexai" in client_info.client_library_version
        assert "ChatVertexAI" in client_info.client_library_version
        assert llm.full_model_name == (
            f"projects/test-proj/locations/{location}/publishers/google/models/gemini-1.0-pro-001"
        )


@pytest.mark.parametrize(
    "model,location",
    [
        (
            "gemini-1.0-pro-001",
            "moon-dark1",
        ),
    ],
)
def test_model_name_presence_in_chat_results(model: str, location: str) -> None:
    config = {"model": model, "location": location}
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._base.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm_response = llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        assert len(llm_response.generations) != 0
        assert isinstance(llm_response.generations[0].message, AIMessage)
        assert (
            llm_response.generations[0].message.response_metadata["model_name"]
            == "gemini-1.0-pro-001"
        )


def test_tuned_model_name() -> None:
    llm = ChatVertexAI(
        model_name="gemini-pro",
        project="test-project",
        tuned_model_name="projects/123/locations/europe-west4/endpoints/456",
        max_tokens=500,
    )
    assert llm.model_name == "gemini-pro"
    assert llm.tuned_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.full_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.max_output_tokens == 500
    assert llm.max_tokens == 500


def test_parse_examples_correct() -> None:
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    text_answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    answer = AIMessage(content=text_answer)
    examples = _parse_examples([question, answer, question, answer])
    assert len(examples) == 2
    assert examples == [
        InputOutputTextPair(input_text=text_question, output_text=text_answer),
        InputOutputTextPair(input_text=text_question, output_text=text_answer),
    ]


def test_parse_examples_failes_wrong_sequence() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = _parse_examples([AIMessage(content="a")])
    assert (
        str(exc_info.value)
        == "Expect examples to have an even amount of messages, got 1."
    )


@dataclass
class StubTextChatResponse:
    """Stub text-chat response from VertexAI for testing."""

    text: str


@pytest.mark.parametrize("stop", [None, "stop1"])
def test_vertexai_args_passed(stop: Optional[str]) -> None:
    response_text = "Goodbye"
    user_prompt = "Hello"
    prompt_params: Dict[str, Any] = {
        "max_output_tokens": 1,
        "temperature": 10000.0,
        "top_k": 10,
        "top_p": 0.5,
    }

    # Mock the library to ensure the args are passed correctly
    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as mg:
        mock_response = MagicMock()
        mock_response.candidates = [StubTextChatResponse(text=response_text)]
        mock_chat = MagicMock()
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        mock_model = MagicMock()
        mock_start_chat = MagicMock(return_value=mock_chat)
        mock_model.start_chat = mock_start_chat
        mg.return_value = mock_model

        model = ChatVertexAI(**prompt_params, project="test-proj")
        message = HumanMessage(content=user_prompt)
        if stop:
            response = model([message], stop=[stop])
        else:
            response = model([message])

        assert response.content == response_text
        mock_send_message.assert_called_once_with(
            message=user_prompt, candidate_count=1
        )
        expected_stop_sequence = [stop] if stop else None
        mock_start_chat.assert_called_once_with(
            message_history=[], **prompt_params, stop_sequences=expected_stop_sequence
        )


def test_parse_chat_history_correct() -> None:
    text_context = (
        "My name is Peter. You are my personal assistant. My "
        "favorite movies are Lord of the Rings and Hobbit."
    )
    context = SystemMessage(content=text_context)
    text_question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    question = HumanMessage(content=text_question)
    text_answer = (
        "Sure, You might enjoy The Lord of the Rings: The Fellowship of the Ring "
        "(2001): This is the first movie in the Lord of the Rings trilogy."
    )
    answer = AIMessage(content=text_answer)
    history = _parse_chat_history([context, question, answer, question, answer])
    assert history.context == context.content
    assert len(history.history) == 4
    assert history.history == [
        ChatMessage(content=text_question, author="user"),
        ChatMessage(content=text_answer, author="bot"),
        ChatMessage(content=text_question, author="user"),
        ChatMessage(content=text_answer, author="bot"),
    ]


def test_parse_history_gemini() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    messages = [system_message, message1, message2, message3]
    image_bytes_loader = ImageBytesLoader()
    system_instructions, history = _parse_chat_history_gemini(
        messages, image_bytes_loader
    )
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1
    assert history[1].role == "model"
    assert history[1].parts[0].text == text_answer1
    assert system_instructions and system_instructions.parts[0].text == system_input


def test_parse_history_gemini_converted_message() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    messages = [system_message, message1, message2, message3]
    image_bytes_loader = ImageBytesLoader()
    _, history = _parse_chat_history_gemini(
        messages, image_bytes_loader, convert_system_message_to_human=True
    )
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[0].parts[0].text == system_input
    assert history[0].parts[1].text == text_question1
    assert history[1].role == "model"
    assert history[1].parts[0].text == text_answer1


def test_parse_history_gemini_function() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1 = "Which is bigger 2+2, 3*3 or 4-4?"
    fn_name_1 = "add"
    fn_name_2 = "multiply"
    fn_name_3 = "subtract"
    text_answer1 = "3*3 is bigger than 2+2 and 4-4"
    tool_call_1 = create_tool_call(
        name=fn_name_1,
        id="1",
        args={
            "arg1": "2",
            "arg2": "2",
        },
    )
    tool_call_2 = create_tool_call(
        name=fn_name_2,
        id="2",
        args={
            "arg1": "3",
            "arg2": "3",
        },
    )
    tool_call_3 = create_tool_call(
        name=fn_name_3,
        id="3",
        args={
            "arg1": "4",
            "arg2": "4",
        },
    )

    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(
        content="",
        tool_calls=[
            tool_call_1,
            tool_call_2,
        ],
    )
    message3 = ToolMessage(content="2", tool_call_id="1")
    message4 = FunctionMessage(name=fn_name_2, content="9")
    message5 = AIMessage(content="", tool_calls=[tool_call_3])
    message6 = ToolMessage(content="0", tool_call_id="3")
    message7 = AIMessage(content=text_answer1)
    messages = [
        system_message,
        message1,
        message2,
        message3,
        message4,
        message5,
        message6,
        message7,
    ]
    image_bytes_loader = ImageBytesLoader()
    system_instructions, history = _parse_chat_history_gemini(
        messages, image_bytes_loader
    )
    assert len(history) == 6
    assert system_instructions and system_instructions.parts[0].text == system_input
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1

    assert history[1].role == "model"
    assert history[1].parts[0].function_call == FunctionCall(
        name=tool_call_1["name"], args=tool_call_1["args"]
    )
    assert history[1].parts[1].function_call == FunctionCall(
        name=tool_call_2["name"], args=tool_call_2["args"]
    )

    assert history[2].role == "function"
    assert history[2].parts[0].function_response == FunctionResponse(
        name=fn_name_1,
        response={"content": message3.content},
    )
    assert history[2].parts[1].function_response == FunctionResponse(
        name=message4.name,
        response={"content": message4.content},
    )

    assert history[3].role == "model"
    assert history[3].parts[0].function_call == FunctionCall(
        name=tool_call_3["name"], args=tool_call_3["args"]
    )

    assert history[4].role == "function"
    assert history[4].parts[0].function_response == FunctionResponse(
        name=fn_name_3,
        response={"content": message6.content},
    )
    assert history[5].parts[0].text == text_answer1


@pytest.mark.parametrize(
    "source_history, expected_sm, expected_history",
    [
        (
            [
                AIMessage(
                    content="Mike age is 30",
                ),
                SystemMessage(content="test1"),
                SystemMessage(content="test2"),
            ],
            Content(role="system", parts=[Part(text="test1"), Part(text="test2")]),
            [Content(role="model", parts=[Part(text="Mike age is 30")])],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30", "Arthur age is 30"],
                ),
            ],
            None,
            [
                Content(
                    role="model",
                    parts=[
                        Part(text="Mike age is 30"),
                        Part(text="Arthur age is 30"),
                    ],
                ),
            ],
        ),
        (
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            )
                        )
                    ],
                )
            ],
        ),
        (
            [
                AIMessage(
                    content="Mike age is 30",
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                Content(
                    role="model",
                    parts=[
                        Part(text="Mike age is 30"),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            )
                        ),
                    ],
                )
            ],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30", "Arthur age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                Content(
                    role="model",
                    parts=[
                        Part(text="Mike age is 30"),
                        Part(text="Arthur age is 30"),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            )
                        ),
                    ],
                )
            ],
        ),
        (
            [
                AIMessage(
                    content=["Mike age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Rob"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
                AIMessage(
                    content=["Arthur age is 30"],
                    tool_calls=[
                        create_tool_call(
                            name="Information",
                            args={"name": "Ben"},
                            id="00000000-0000-0000-0000-00000000000",
                        ),
                    ],
                ),
            ],
            None,
            [
                Content(
                    role="model",
                    parts=[
                        Part(text="Mike age is 30"),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Rob"},
                            )
                        ),
                        Part(text="Arthur age is 30"),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            )
                        ),
                    ],
                )
            ],
        ),
    ],
)
def test_parse_history_gemini_multi(
    source_history, expected_sm, expected_history
) -> None:
    image_bytes_loader = ImageBytesLoader()
    sm, result_history = _parse_chat_history_gemini(
        history=source_history, imageBytesLoader=image_bytes_loader
    )

    for result, expected in zip(result_history, expected_history):
        assert result == expected
    assert sm == expected_sm


def test_default_params_palm() -> None:
    user_prompt = "Hello"

    with patch("vertexai._model_garden._model_garden_models._from_pretrained") as mg:
        mock_response = MagicMock()
        mock_response.candidates = [StubTextChatResponse(text="Goodbye")]
        mock_chat = MagicMock()
        mock_send_message = MagicMock(return_value=mock_response)
        mock_chat.send_message = mock_send_message

        mock_model = MagicMock()
        mock_start_chat = MagicMock(return_value=mock_chat)
        mock_model.start_chat = mock_start_chat
        mg.return_value = mock_model

        model = ChatVertexAI(model_name="text-bison@001")
        message = HumanMessage(content=user_prompt)
        _ = model([message])
        mock_start_chat.assert_called_once_with(
            message_history=[],
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
            stop_sequences=None,
            temperature=0.0,
        )


def test_default_params_gemini() -> None:
    user_prompt = "Hello"

    with patch("langchain_google_vertexai._base.v1beta1PredictionServiceClient") as mc:
        response = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text="Hi")]))]
        )
        mock_generate_content = MagicMock(return_value=response)
        mc.return_value.generate_content = mock_generate_content

        model = ChatVertexAI(model_name="gemini-pro", project="test-project")
        message = HumanMessage(content=user_prompt)
        _ = model.invoke([message])
        mock_generate_content.assert_called_once()
        assert (
            mock_generate_content.call_args.kwargs["request"].contents[0].role == "user"
        )
        assert (
            mock_generate_content.call_args.kwargs["request"].contents[0].parts[0].text
            == "Hello"
        )
        expected = GenerationConfig(
            candidate_count=1,
        )
        assert (
            mock_generate_content.call_args.kwargs["request"].generation_config
            == expected
        )
        assert mock_generate_content.call_args.kwargs["request"].tools == []
        assert not mock_generate_content.call_args.kwargs["request"].tool_config
        assert not mock_generate_content.call_args.kwargs["request"].safety_settings


@pytest.mark.parametrize(
    "raw_candidate, expected",
    [
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text="Mike age is 30",
                        )
                    ],
                )
            ),
            AIMessage(
                content="Mike age is 30",
                additional_kwargs={},
            ),
        ),
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text="Mike age is 30",
                        ),
                        Part(
                            text="Arthur age is 30",
                        ),
                    ],
                )
            ),
            AIMessage(
                content=["Mike age is 30", "Arthur age is 30"],
                additional_kwargs={},
            ),
        ),
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            ),
                        )
                    ],
                )
            ),
            AIMessage(
                content="",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"info": ["A", "B", "C"]},
                            ),
                        )
                    ],
                )
            ),
            AIMessage(
                content="",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={
                                    "people": [
                                        {"name": "Joe", "age": 30},
                                        {"name": "Martha"},
                                    ]
                                },
                            ),
                        )
                    ],
                )
            ),
            AIMessage(
                content="",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"info": [[1, 2, 3], [4, 5, 6]]},
                            ),
                        )
                    ],
                )
            ),
            AIMessage(
                content="",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(text="Mike age is 30"),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            ),
                        ),
                    ],
                )
            ),
            AIMessage(
                content="Mike age is 30",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            ),
                        ),
                        Part(text="Mike age is 30"),
                    ],
                )
            ),
            AIMessage(
                content="Mike age is 30",
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
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Ben"},
                            ),
                        ),
                        Part(
                            function_call=FunctionCall(
                                name="Information",
                                args={"name": "Mike"},
                            ),
                        ),
                    ],
                )
            ),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"name": "Mike"}),
                    }
                },
                tool_calls=[
                    create_tool_call(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                    create_tool_call(
                        name="Information",
                        args={"name": "Mike"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
    ],
)
def test_parse_response_candidate(raw_candidate, expected) -> None:
    with patch("langchain_google_vertexai.chat_models.uuid.uuid4") as uuid4:
        uuid4.return_value = "00000000-0000-0000-0000-00000000000"
        response_candidate = raw_candidate
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


def test_parser_multiple_tools():
    class Add(BaseModel):
        arg1: int
        arg2: int

    class Multiply(BaseModel):
        arg1: int
        arg2: int

    with patch("langchain_google_vertexai._base.v1beta1PredictionServiceClient") as mc:
        response = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=FunctionCall(
                                    name="Add",
                                    args={"arg1": "1", "arg2": "2"},
                                )
                            ),
                            Part(
                                function_call=FunctionCall(
                                    name="Multiply",
                                    args={"arg1": "3", "arg2": "3"},
                                )
                            ),
                        ],
                    )
                )
            ]
        )
        mock_generate_content = MagicMock(return_value=response)
        mc.return_value.generate_content = mock_generate_content

        model = ChatVertexAI(model_name="gemini-1.5-pro", project="test-project")
        message = HumanMessage(content="Hello")
        parser = PydanticToolsParser(tools=[Add, Multiply])
        llm = model | parser
        result = llm.invoke([message])
        mock_generate_content.assert_called_once()
        assert isinstance(result, list)
        assert isinstance(result[0], Add)
        assert result[0] == Add(arg1=1, arg2=2)
        assert isinstance(result[1], Multiply)
        assert result[1] == Multiply(arg1=3, arg2=3)


def test_generation_config_gemini() -> None:
    model = ChatVertexAI(
        model_name="gemini-pro",
        temperature=0.2,
        top_k=3,
        frequency_penalty=0.2,
        presence_penalty=0.6,
    )
    generation_config = model._generation_config_gemini(
        temperature=0.3,
        stop=["stop"],
        candidate_count=2,
        frequency_penalty=0.9,
        presence_penalty=0.8,
    )
    expected = GenerationConfig(
        stop_sequences=["stop"],
        temperature=0.3,
        top_k=3,
        candidate_count=2,
        frequency_penalty=0.9,
        presence_penalty=0.8,
    )
    assert generation_config == expected


def test_safety_settings_gemini() -> None:
    model = ChatVertexAI(
        model_name="gemini-pro", temperature=0.2, top_k=3, project="test-project"
    )
    expected_safety_setting = SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    )
    safety_settings = model._safety_settings_gemini([expected_safety_setting])
    assert safety_settings == [expected_safety_setting]
    safety_settings = model._safety_settings_gemini(
        {"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"}
    )
    assert safety_settings == [expected_safety_setting]
    safety_settings = model._safety_settings_gemini({2: 1})
    assert safety_settings == [expected_safety_setting]
    threshold = SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    safety_settings = model._safety_settings_gemini(
        {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold}
    )
    assert safety_settings == [expected_safety_setting]


def test_safety_settings_gemini_init() -> None:
    expected_safety_setting = [
        VertexSafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            method=SafetySetting.HarmBlockMethod.SEVERITY,
        )
    ]
    model = ChatVertexAI(
        model_name="gemini-pro",
        temperature=0.2,
        top_k=3,
        project="test-project",
        safety_settings=expected_safety_setting,
    )
    safety_settings = model._safety_settings_gemini(None)
    assert safety_settings == expected_safety_setting


def test_multiple_fc() -> None:
    prompt = (
        "I'm trying to decide whether to go to London or Zurich this weekend. How "
        "hot are those cities? How about Singapore? Or maybe Tokyo. I want to go "
        "somewhere not that cold but not too hot either. Suggest me."
    )
    raw_history = [
        HumanMessage(content=prompt),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"location": "Munich"}, "id": "1"},
                {"name": "get_weather", "args": {"location": "London"}, "id": "2"},
                {"name": "get_weather", "args": {"location": "Berlin"}, "id": "3"},
            ],
        ),
        ToolMessage(
            name="get_weather",
            tool_call_id="1",
            content='{"condition": "sunny", "temp_c": -23.9}',
        ),
        ToolMessage(
            name="get_weather",
            tool_call_id="2",
            content='{"condition": "sunny", "temp_c": -30.0}',
        ),
        ToolMessage(
            name="get_weather",
            tool_call_id="3",
            content='{"condition": "rainy", "temp_c": 25.2}',
        ),
    ]
    image_bytes_loader = ImageBytesLoader()
    _, history = _parse_chat_history_gemini(raw_history, image_bytes_loader)
    expected = [
        Content(
            parts=[Part(text=prompt)],
            role="user",
        ),
        Content(
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="get_weather", args={"location": "Munich"}
                    )
                ),
                Part(
                    function_call=FunctionCall(
                        name="get_weather", args={"location": "London"}
                    )
                ),
                Part(
                    function_call=FunctionCall(
                        name="get_weather", args={"location": "Berlin"}
                    )
                ),
            ],
            role="model",
        ),
        Content(
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="get_weather",
                        response={"condition": "sunny", "temp_c": -23.9},
                    )
                ),
                Part(
                    function_response=FunctionResponse(
                        name="get_weather",
                        response={"condition": "sunny", "temp_c": -30.0},
                    )
                ),
                Part(
                    function_response=FunctionResponse(
                        name="get_weather",
                        response={"condition": "rainy", "temp_c": 25.2},
                    )
                ),
            ],
            role="function",
        ),
    ]
    assert history == expected


def test_parse_chat_history_gemini_with_literal_eval() -> None:
    instruction = "Describe the attached media in 5 words."
    text_message = {"type": "text", "text": instruction}
    message = str([text_message])
    history: list[BaseMessage] = [HumanMessage(content=message)]
    image_bytes_loader = ImageBytesLoader()
    _, response = _parse_chat_history_gemini(
        history=history,
        imageBytesLoader=image_bytes_loader,
        perform_literal_eval_on_string_raw_content=True,
    )
    parts = [
        Part(text=instruction),
    ]
    expected = [Content(role="user", parts=parts)]
    assert expected == response


def test_parse_chat_history_gemini_without_literal_eval() -> None:
    instruction = "Describe the attached media in 5 words."
    text_message = {"type": "text", "text": instruction}
    message = str([text_message])
    history: list[BaseMessage] = [HumanMessage(content=message)]
    image_bytes_loader = ImageBytesLoader()
    _, response = _parse_chat_history_gemini(
        history=history,
        imageBytesLoader=image_bytes_loader,
        perform_literal_eval_on_string_raw_content=False,
    )
    parts = [
        Part(text=message),
    ]
    expected = [Content(role="user", parts=parts)]
    assert expected == response


def test_init_client_with_custom_api_endpoint() -> None:
    config = {
        "model": "gemini-1.5-pro",
        "api_endpoint": "https://example.com",
        "api_transport": "rest",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._base.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        transport = mock_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://example.com"
        assert transport == "rest"


def test_init_client_with_custom_base_url() -> None:
    config = {
        "model": "gemini-1.5-pro",
        "base_url": "https://example.com",
        "api_transport": "rest",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._base.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        transport = mock_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://example.com"
        assert transport == "rest"


def test_init_client_with_custom_model_kwargs() -> None:
    llm = ChatAnthropicVertex(
        project="test-project",
        location="test-location",
        model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    assert llm.model_kwargs == {"thinking": {"type": "enabled", "budget_tokens": 1024}}

    default_params = llm._default_params
    assert default_params["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_model_kwargs_chat_vertex() -> None:
    """Test we can transfer unknown params to model_kwargs."""
    llm = ChatVertexAI(
        model="my-model",
        convert_system_message_to_human=True,
        model_kwargs={"foo": "bar"},
    )
    assert llm.model_name == "my-model"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatVertexAI(
            model="my-model",
            convert_system_message_to_human=True,
            foo="bar",
        )
    assert llm.model_name == "my-model"
    assert llm.convert_system_message_to_human is True
    assert llm.model_kwargs == {"foo": "bar"}


def test_anthropic_format_output() -> None:
    """Test format output handles different content structures correctly."""

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int
        cache_creation_input_tokens: Optional[int]
        cache_read_input_tokens: Optional[int]

    @dataclass
    class Message:
        def model_dump(self):
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "123",
                        "name": "calculator",
                        "input": {"number": 42},
                    }
                ],
                "model": "baz",
                "role": "assistant",
                "usage": Usage(
                    input_tokens=2,
                    output_tokens=1,
                    cache_creation_input_tokens=1,
                    cache_read_input_tokens=1,
                ),
                "type": "message",
            }

        usage: Usage

    test_msg = Message(
        usage=Usage(
            input_tokens=2,
            output_tokens=1,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=1,
        )
    )

    model = ChatAnthropicVertex(project="test-project", location="test-location")
    result = model._format_output(test_msg)

    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.content == test_msg.model_dump()["content"]
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["name"] == "calculator"
    assert message.tool_calls[0]["args"] == {"number": 42}
    assert message.usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 1,
    }


def test_anthropic_format_output_with_chain_of_thoughts() -> None:
    """Test format output handles chain of thoughts correctly."""

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int
        cache_creation_input_tokens: Optional[int]
        cache_read_input_tokens: Optional[int]

    @dataclass
    class Message:
        def model_dump(self):
            return {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Thoughts of the model...",
                        "signature": "thought-signatire",
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "redacted-thoughts-data",
                    },
                    {
                        "type": "text",
                        "text": "Final output the model",
                    },
                ],
                "model": "baz",
                "role": "assistant",
                "usage": Usage(
                    input_tokens=2,
                    output_tokens=1,
                    cache_creation_input_tokens=1,
                    cache_read_input_tokens=1,
                ),
                "type": "message",
            }

        usage: Usage

    test_msg = Message(
        usage=Usage(
            input_tokens=2,
            output_tokens=1,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=1,
        )
    )

    model = ChatAnthropicVertex(project="test-project", location="test-location")
    result = model._format_output(test_msg)

    message = result.generations[0].message
    print(message)
    assert isinstance(message, AIMessage)
    assert len(message.content) == 3
    assert message.content == test_msg.model_dump()["content"]
    assert message.usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
        "cache_creation_input_tokens": 1,
        "cache_read_input_tokens": 1,
    }
