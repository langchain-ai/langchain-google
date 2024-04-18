"""Test chat model integration."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from google.cloud.aiplatform_v1beta1.types import (
    Content as Content,
)
from google.cloud.aiplatform_v1beta1.types import (
    FunctionCall,
    FunctionResponse,
)
from google.cloud.aiplatform_v1beta1.types import (
    Part as Part,
)
from google.cloud.aiplatform_v1beta1.types import (
    content as gapic_content_types,
)
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
)
from vertexai.generative_models import (  # type: ignore
    Candidate,
)
from vertexai.language_models import (  # type: ignore
    ChatMessage,
    InputOutputTextPair,
)

from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _parse_chat_history,
    _parse_chat_history_gemini,
    _parse_examples,
    _parse_response_candidate,
)


def test_model_name() -> None:
    for llm in [
        ChatVertexAI(model_name="gemini-pro", project="test-project"),
        ChatVertexAI(model="gemini-pro", project="test-project"),  # type: ignore[call-arg]
    ]:
        assert llm.model_name == "gemini-pro"


def test_tuned_model_name() -> None:
    llm = ChatVertexAI(
        model_name="gemini-pro",
        project="test-project",
        tuned_model_name="projects/123/locations/europe-west4/endpoints/456",
    )
    assert llm.model_name == "gemini-pro"
    assert llm.tuned_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.client._model_name == "projects/123/locations/europe-west4/endpoints/456"


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

        model = ChatVertexAI(**prompt_params)
        message = HumanMessage(content=user_prompt)
        if stop:
            response = model([message], stop=[stop])
        else:
            response = model([message])

        assert response.content == response_text
        mock_send_message.assert_called_once_with(user_prompt, candidate_count=1)
        expected_stop_sequence = [stop] if stop else None
        mock_start_chat.assert_called_once_with(
            context=None,
            message_history=[],
            **prompt_params,
            stop_sequences=expected_stop_sequence,
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
    system_instructions, history = _parse_chat_history_gemini(messages)
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
    _, history = _parse_chat_history_gemini(
        messages, convert_system_message_to_human=True
    )
    assert len(history) == 3
    assert history[0].role == "user"
    assert history[0].parts[0].text == system_input
    assert history[0].parts[1].text == text_question1
    assert history[1].role == "model"
    assert history[1].parts[0].text == text_answer1


def test_parse_history_gemini_function() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1 = "Which is bigger 2+2 or 2*2?"
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
    system_instructions, history = _parse_chat_history_gemini(messages)
    assert len(history) == 6
    assert system_instructions and system_instructions.parts[0].text == system_input
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1

    assert history[1].role == "model"
    assert history[1].parts[0].function_call == FunctionCall(
        name=function_call_1["name"], args=json.loads(function_call_1["arguments"])
    )

    assert history[2].role == "function"
    assert history[2].parts[0].function_response == FunctionResponse(
        name=function_call_1["name"],
        response={"content": function_answer1},
    )

    assert history[3].role == "model"
    assert history[3].parts[0].function_call == FunctionCall(
        name=function_call_2["name"], args=json.loads(function_call_2["arguments"])
    )

    assert history[4].role == "function"
    assert history[2].parts[0].function_response == FunctionResponse(
        name=function_call_2["name"],
        response={"content": function_answer2},
    )

    assert history[5].role == "model"
    assert history[5].parts[0].text == text_answer1


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
            context=None,
            message_history=[],
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
            stop_sequences=None,
            temperature=0.0,
        )


@dataclass
class StubGeminiResponse:
    """Stub gemini response from VertexAI for testing."""

    text: str
    content: Any
    citation_metadata: Any
    safety_ratings: List[Any] = field(default_factory=list)


def test_default_params_gemini() -> None:
    user_prompt = "Hello"

    with patch("langchain_google_vertexai.chat_models.GenerativeModel") as gm:
        mock_response = MagicMock()
        mock_response.candidates = [
            StubGeminiResponse(
                text="Goodbye",
                content=Mock(parts=[Mock(function_call=None)]),
                citation_metadata=None,
            )
        ]
        mock_generate_content = MagicMock(return_value=mock_response)
        mock_model = MagicMock()
        mock_model.generate_content = mock_generate_content
        gm.return_value = mock_model

        model = ChatVertexAI(model_name="gemini-pro")
        message = HumanMessage(content=user_prompt)
        _ = model.invoke([message])
        mock_generate_content.assert_called_once()
        assert mock_generate_content.call_args.args[0][0].parts[0].text == user_prompt


@pytest.mark.parametrize(
    "raw_candidate, expected",
    [
        (
            gapic_content_types.Candidate(
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
            gapic_content_types.Candidate(
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
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"name": "Ben"}),
                    },
                },
                tool_calls=[
                    ToolCall(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            gapic_content_types.Candidate(
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
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps({"info": ["A", "B", "C"]}),
                    },
                },
                tool_calls=[
                    ToolCall(
                        name="Information",
                        args={"info": ["A", "B", "C"]},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            gapic_content_types.Candidate(
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
                    ToolCall(
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
            gapic_content_types.Candidate(
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
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps(
                            {"info": [[1, 2, 3], [4, 5, 6]]}
                        ),
                    },
                },
                tool_calls=[
                    ToolCall(
                        name="Information",
                        args={"info": [[1, 2, 3], [4, 5, 6]]},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            gapic_content_types.Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text="Mike age is 30"
                        ),
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
                content="Mike age is 30",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps(
                            {"name": "Ben"}
                        ),
                    },
                },
                tool_calls=[
                    ToolCall(
                        name="Information",
                        args={"name": "Ben"},
                        id="00000000-0000-0000-0000-00000000000",
                    ),
                ],
            ),
        ),
        (
            gapic_content_types.Candidate(
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
                            text="Mike age is 30"
                        ),
                    ],
                )
            ),
            AIMessage(
                content="Mike age is 30",
                additional_kwargs={
                    "function_call": {
                        "name": "Information",
                        "arguments": json.dumps(
                            {"name": "Ben"}
                        ),
                    },
                },
                tool_calls=[
                    ToolCall(
                        name="Information",
                        args={"name": "Ben"},
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
        response_candidate = Candidate._from_gapic(raw_candidate)
        result = _parse_response_candidate(response_candidate)
        assert result.content == expected.content
        assert result.tool_calls == expected.tool_calls
        for key, value in expected.additional_kwargs.items():
          if key == "function_call":
            res_fc = result.additional_kwargs[key]
            exp_fc = value
            assert res_fc["name"] == exp_fc["name"]

            assert json.loads(res_fc["arguments"]) == json.loads(exp_fc["arguments"])
          else:
            res_kw = result.additional_kwargs[key]
            exp_kw = value
            assert res_kw == exp_kw

