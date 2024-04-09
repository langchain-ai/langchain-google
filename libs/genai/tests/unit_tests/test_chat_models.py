"""Test chat model integration."""

from typing import Dict, List, Optional, Union
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_google_genai.chat_models import (
    ChatGoogleGenerativeAI,
    _parse_chat_history,
)


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key="...",
        top_k=2,
        top_p=1,
        temperature=0.7,
        n=2,
    )
    ChatGoogleGenerativeAI(
        model="gemini-nano",
        google_api_key="...",
        top_k=2,
        top_p=1,
        temperature=0.7,
        candidate_count=2,
    )


def test_api_key_is_string() -> None:
    chat = ChatGoogleGenerativeAI(model="gemini-nano", google_api_key="secret-api-key")
    assert isinstance(chat.google_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(capsys: CaptureFixture) -> None:
    chat = ChatGoogleGenerativeAI(model="gemini-nano", google_api_key="secret-api-key")
    print(chat.google_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_parse_history() -> None:
    system_input = "You're supposed to answer math questions."
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    messages = [system_message, message1, message2, message3]
    system_instruction, history = _parse_chat_history(
        messages, convert_system_message_to_human=True
    )
    assert len(history) == 3
    assert history[0] == {
        "role": "user",
        "parts": [{"text": text_question1}],
    }
    assert history[1] == {"role": "model", "parts": [{"text": text_answer1}]}
    assert system_instruction == [{"text": system_input}]


@pytest.mark.parametrize("content", ['["a"]', '{"a":"b"}', "function output"])
def test_parse_function_history(content: Union[str, List[Union[str, Dict]]]) -> None:
    function_message = FunctionMessage(name="search_tool", content=content)
    _parse_chat_history([function_message], convert_system_message_to_human=True)


@pytest.mark.parametrize(
    "headers", (None, {}, {"X-User-Header": "Coco", "X-User-Header2": "Jamboo"})
)
def test_additional_headers_support(headers: Optional[Dict[str, str]]) -> None:
    mock_configure = Mock()
    params = {
        "google_api_key": "[secret]",
        "client_options": {"api_endpoint": "http://127.0.0.1:8000/ai"},
        "transport": "rest",
        "additional_headers": headers,
    }

    with patch("langchain_google_genai.chat_models.genai.configure", mock_configure):
        chat = ChatGoogleGenerativeAI(model="gemini-pro", **params)

    expected_default_metadata: tuple = ()
    if not headers:
        assert chat.additional_headers == headers
    else:
        assert chat.additional_headers
        assert all(header in chat.additional_headers for header in headers.keys())
        expected_default_metadata = tuple(headers.items())

    mock_configure.assert_called_once_with(
        api_key=params["google_api_key"],
        transport=params["transport"],
        client_options=params["client_options"],
        default_metadata=expected_default_metadata,
    )
