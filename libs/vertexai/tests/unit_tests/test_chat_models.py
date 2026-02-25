"""Chat model unit tests."""

import base64
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.cloud.aiplatform_v1beta1.types import (
    Blob,
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
from vertexai.generative_models import (
    SafetySetting as VertexSafetySetting,  # TODO: migrate to google-genai
)
from vertexai.language_models import (
    InputOutputTextPair,
)

from langchain_google_vertexai._base import _get_prediction_client
from langchain_google_vertexai._compat import _convert_from_v1_to_vertex
from langchain_google_vertexai._image_utils import ImageBytesLoader
from langchain_google_vertexai.chat_models import (
    ChatVertexAI,
    _bytes_to_base64,
    _parse_chat_history_gemini,
    _parse_examples,
    _parse_response_candidate,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from tests.integration_tests.conftest import (
    _DEFAULT_MODEL_NAME,
)


@pytest.fixture
def clear_prediction_client_cache() -> None:
    # Clear the prediction client cache so we can mock varied calls to
    # PredictionServiceClient
    _get_prediction_client.cache_clear()


def test_init() -> None:
    """Test initialization of `ChatVertexAI` with different parameter names.

    Done since we have aliasing of some parameters for consistency with other LLMs.
    """
    for llm in [
        ChatVertexAI(
            model_name="gemini-2-5-flash",
            project="test-project",
            max_output_tokens=10,
            stop=["bar"],
            location="moon-dark1",
        ),
        ChatVertexAI(
            model="gemini-2-5-flash",
            project="test-proj",
            max_tokens=10,
            stop_sequences=["bar"],
            location="moon-dark1",
        ),
    ]:
        assert llm.model_name == "gemini-2-5-flash"
        assert llm.max_output_tokens == 10
        assert llm.stop == ["bar"]

        ls_params = llm._get_ls_params()
        assert ls_params == {
            "ls_provider": "google_vertexai",
            "ls_model_name": "gemini-2-5-flash",
            "ls_model_type": "chat",
            "ls_temperature": None,
            "ls_max_tokens": 10,
            "ls_stop": ["bar"],
        }

    # Test initialization with an invalid argument to check warning
    with patch("langchain_google_vertexai.chat_models.logger.warning") as mock_warning:
        # Suppress UserWarning during test execution - we're testing the warning
        # mechanism via logger mock assertions, not via pytest's warning system
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            llm = ChatVertexAI(
                model_name="gemini-2-5-flash",
                project="test-project",
                safety_setting={
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
                },  # Invalid arg
            )
        assert llm.model_name == "gemini-2-5-flash"
        assert llm.project == "test-project"
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


@pytest.mark.parametrize(
    ("model", "location"),
    [
        (
            "gemini-2-5-flash",
            "moon-dark1",
        ),
        ("publishers/google/models/gemini-2-5-flash", "moon-dark2"),
    ],
)
def test_init_client(model: str, location: str) -> None:
    """Test initialization of `ChatVertexAI` with different models and locations.

    Ensure the user agent is set correctly and the full model name is constructed using
    the provided project and location.
    """
    config = {"model": model, "location": location}
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
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
            f"projects/test-proj/locations/{location}/publishers/google/models/gemini-2-5-flash"
        )


def test_init_client_with_custom_api_endpoint() -> None:
    """Test that custom API endpoint and transport are set correctly."""
    config = {
        "model": "gemini-2.5-pro",
        "api_endpoint": "https://example.com",
        "api_transport": "rest",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        transport = mock_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://example.com"
        assert transport == "rest"


def test_init_client_with_custom_base_url(clear_prediction_client_cache: Any) -> None:
    """Test that `base_url` alias is preserved and used in API calls."""
    config = {
        "model": "gemini-2.5-pro",
        "base_url": "https://example.com",
        "api_transport": "rest",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )

    assert llm.api_endpoint == "https://example.com"

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        transport = mock_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://example.com"
        assert transport == "rest"


def test_api_endpoint_preservation(clear_prediction_client_cache: Any) -> None:
    """Test that `api_endpoint` field is preserved and used in API calls."""
    config = {
        "model": "gemini-2.5-pro",
        "api_endpoint": "https://direct-endpoint.com",
        "api_transport": "rest",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )

    assert llm.api_endpoint == "https://direct-endpoint.com"

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        client_options = mock_prediction_service.call_args.kwargs["client_options"]
        assert client_options.api_endpoint == "https://direct-endpoint.com"


async def test_async_base_url_support(clear_prediction_client_cache: Any) -> None:
    """Test that `base_url` is properly used in async API calls."""
    config = {
        "model": "gemini-2.5-pro",
        "base_url": "https://async-example.com",
        "api_transport": "grpc_asyncio",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )

    assert llm.api_endpoint == "https://async-example.com"

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceAsyncClient"
    ) as mock_async_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_generate_content = AsyncMock(return_value=response)
        mock_async_prediction_service.return_value.generate_content = (
            mock_generate_content
        )

        await llm._agenerate_gemini(messages=[])
        mock_async_prediction_service.assert_called_once()
        client_options = mock_async_prediction_service.call_args.kwargs[
            "client_options"
        ]
        transport = mock_async_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://async-example.com"
        assert transport == "grpc_asyncio"


async def test_async_api_endpoint_support(clear_prediction_client_cache: Any) -> None:
    """Test that `api_endpoint` is properly used in async API calls."""
    config = {
        "model": "gemini-2.5-pro",
        "api_endpoint": "https://async-direct-endpoint.com",
        "api_transport": "grpc_asyncio",
    }
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )

    assert llm.api_endpoint == "https://async-direct-endpoint.com"

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceAsyncClient"
    ) as mock_async_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_generate_content = AsyncMock(return_value=response)
        mock_async_prediction_service.return_value.generate_content = (
            mock_generate_content
        )

        await llm._agenerate_gemini(messages=[])
        mock_async_prediction_service.assert_called_once()
        client_options = mock_async_prediction_service.call_args.kwargs[
            "client_options"
        ]
        transport = mock_async_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://async-direct-endpoint.com"
        assert transport == "grpc_asyncio"


async def test_async_api_endpoint_alias_behavior(
    clear_prediction_client_cache: Any,
) -> None:
    """Test that `api_endpoint` and `base_url` are aliases in async calls."""
    # Test 1: Only api_endpoint specified
    llm1 = ChatVertexAI(
        model="gemini-2.5-pro",
        project="test-proj",
        api_endpoint="https://api-endpoint-only.com",
        api_transport="grpc_asyncio",
    )
    assert llm1.api_endpoint == "https://api-endpoint-only.com"

    # Test 2: Only base_url specified (should be aliased to api_endpoint)
    llm2 = ChatVertexAI(
        model="gemini-2.5-pro",
        project="test-proj",
        base_url="https://base-url-only.com",
        api_transport="grpc_asyncio",
    )
    assert llm2.api_endpoint == "https://base-url-only.com"

    # Test async call with base_url
    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceAsyncClient"
    ) as mock_async_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_generate_content = AsyncMock(return_value=response)
        mock_async_prediction_service.return_value.generate_content = (
            mock_generate_content
        )

        await llm2._agenerate_gemini(messages=[])
        mock_async_prediction_service.assert_called_once()
        client_options = mock_async_prediction_service.call_args.kwargs[
            "client_options"
        ]
        transport = mock_async_prediction_service.call_args.kwargs["transport"]
        assert client_options.api_endpoint == "https://base-url-only.com"
        assert transport == "grpc_asyncio"


def test_init_client_with_custom_model_kwargs() -> None:
    """Test that custom model kwargs are set correctly."""
    llm = ChatAnthropicVertex(
        project="test-project",
        location="test-location",
        model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    assert llm.model_kwargs == {"thinking": {"type": "enabled", "budget_tokens": 1024}}

    default_params = llm._default_params
    assert default_params["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_profile() -> None:
    model = ChatVertexAI(
        model="gemini-2.0-flash", project="test-project", location="moon-dark1"
    )
    assert model.profile
    assert not model.profile["reasoning_output"]

    model = ChatVertexAI(
        model="gemini-2.5-flash", project="test-project", location="moon-dark1"
    )
    assert model.profile
    assert model.profile["reasoning_output"]

    model = ChatVertexAI(model="foo", project="test-project", location="moon-dark1")
    assert model.profile == {}


@pytest.mark.parametrize(
    ("model", "location"),
    [
        (
            "gemini-2-5-flash",
            "moon-dark1",
        ),
    ],
)
def test_model_name_presence_in_chat_results(
    model: str, location: str, clear_prediction_client_cache: Any
) -> None:
    """Ensure the model name is present in the response metadata of messages."""
    config = {"model": model, "location": location}
    llm = ChatVertexAI(
        **{k: v for k, v in config.items() if v is not None}, project="test-proj"
    )
    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_prediction_service.return_value.generate_content.return_value = response

        llm_response = llm._generate_gemini(messages=[])
        mock_prediction_service.assert_called_once()
        assert len(llm_response.generations) != 0
        assert isinstance(llm_response.generations[0].message, AIMessage)
        assert (
            llm_response.generations[0].message.response_metadata["model_name"]
            == "gemini-2-5-flash"
        )


def test_tuned_model_name() -> None:
    """Test initialization of `ChatVertexAI` with a tuned model name.

    Tuned models must be specified using the full resource name.
    """
    llm = ChatVertexAI(
        model_name="gemini-2-5-flash",
        project="test-project",
        tuned_model_name="projects/123/locations/europe-west4/endpoints/456",
        max_tokens=500,
    )
    assert llm.model_name == "gemini-2-5-flash"
    assert llm.tuned_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.full_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.max_output_tokens == 500
    assert llm.max_tokens == 500  # Alias


def test_default_params_gemini() -> None:
    """Test that default parameters are set correctly in the request."""
    user_prompt = "Hello"

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mc:
        response = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text="Hi")]))]
        )
        mock_generate_content = MagicMock(return_value=response)
        mc.return_value.generate_content = mock_generate_content

        model = ChatVertexAI(model_name="gemini-2-5-flash", project="test-project")
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


def test_generation_config_gemini() -> None:
    """Test that generation config is set correctly in the request when overridden."""
    model = ChatVertexAI(
        model_name="gemini-2-5-flash",
        project="test-project",
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
    # Note merged settings, with method args taking precedence (e.g. temperature)
    expected = GenerationConfig(
        stop_sequences=["stop"],
        temperature=0.3,
        top_k=3,
        candidate_count=2,
        frequency_penalty=0.9,
        presence_penalty=0.8,
    )
    assert generation_config == expected


def test_safety_settings_gemini_init() -> None:
    """Test that safety settings are set correctly when provided at init."""
    expected_safety_setting = [
        VertexSafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            method=SafetySetting.HarmBlockMethod.SEVERITY,
        )
    ]
    model = ChatVertexAI(
        model_name="gemini-2-5-flash",
        temperature=0.2,
        top_k=3,
        project="test-project",
        safety_settings=expected_safety_setting,
    )
    safety_settings = model._safety_settings_gemini(None)
    assert safety_settings == expected_safety_setting


def test_safety_settings_gemini() -> None:
    """Test that safety settings are set correctly in the request."""
    model = ChatVertexAI(
        model_name="gemini-2-5-flash", temperature=0.2, top_k=3, project="test-project"
    )
    expected_safety_setting = SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    )
    safety_settings = model._safety_settings_gemini([expected_safety_setting])
    assert safety_settings == [expected_safety_setting]
    # Ignores for tests that intentionally use invalid dict types
    safety_settings = model._safety_settings_gemini(
        # Ignore since testing string conversion
        {"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"}  # type: ignore[dict-item]
    )
    assert safety_settings == [expected_safety_setting]
    # Ignore since testing int conversion
    safety_settings = model._safety_settings_gemini({2: 1})  # type: ignore[dict-item]
    assert safety_settings == [expected_safety_setting]
    threshold = SafetySetting.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    safety_settings = model._safety_settings_gemini(
        # Ignore since testing enum conversion
        {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold}  # type: ignore[dict-item]
    )
    assert safety_settings == [expected_safety_setting]


def test_parse_examples_correct() -> None:
    """Test parsing of message examples into `InputOutputTextPair` objects."""
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
    """Out of order messages should raise an error.

    Ensure that the _parsed_examples function raises an error if the messages are not
    in the correct Human-AI alternating sequence.
    """
    with pytest.raises(ValueError) as exc_info:
        _ = _parse_examples([AIMessage(content="a")])
    assert (
        str(exc_info.value)
        == "Expect examples to have an even amount of messages, got 1."
    )


def test_parse_history_gemini() -> None:
    """Test parsing of chat history into Gemini format."""
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
    assert system_instructions
    assert system_instructions.parts[0].text == system_input


def test_parse_history_gemini_number() -> None:
    """Ensure that numeric strings are parsed correctly as text.

    TODO: remove this? Seems like an edge case of limited value.
    """
    system_input = "You're supposed to answer math questions."
    text_question1 = "54321.1"
    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    messages = [system_message, message1]
    image_bytes_loader = ImageBytesLoader()
    system_instructions, history = _parse_chat_history_gemini(
        messages, image_bytes_loader, perform_literal_eval_on_string_raw_content=True
    )
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1
    assert system_instructions
    assert system_instructions.parts[0].text == system_input


def test_parse_history_gemini_function_empty_list() -> None:
    """Test parsing of chat history with an empty tool call response.

    If the model calls a function/tool but the response is an empty list, it should
    still be parsed correctly (meaning the function call is recorded, and the function
    response is recorded as an empty content).
    """
    system_input = "You're supposed to answer math questions."
    text_question1 = "Solve the following equation. x^2+16=0"
    fn_name_1 = "root"

    tool_call_1 = create_tool_call(
        name=fn_name_1,
        id="1",
        args={
            "arg1": "-10",
            "arg2": "10",
        },
    )

    system_message = SystemMessage(content=system_input)
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(
        content="",
        tool_calls=[
            tool_call_1,
        ],
    )
    message3 = ToolMessage(content=[], tool_call_id="1")
    messages = [
        system_message,
        message1,
        message2,
        message3,
    ]
    image_bytes_loader = ImageBytesLoader()
    system_instructions, history = _parse_chat_history_gemini(
        messages, image_bytes_loader
    )
    assert len(history) == 3
    assert system_instructions
    assert system_instructions.parts[0].text == system_input
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1

    assert history[1].role == "model"
    assert history[1].parts[0].function_call == FunctionCall(
        name=tool_call_1["name"], args=tool_call_1["args"]
    )

    assert history[2].role == "user"
    assert history[2].parts[0].function_response == FunctionResponse(
        name=fn_name_1,
        response={"content": ""},
    )


def test_parse_history_gemini_function() -> None:
    """Test parsing of chat history with multiple tool calls and responses."""
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
    assert system_instructions
    assert system_instructions.parts[0].text == system_input
    assert history[0].role == "user"
    assert history[0].parts[0].text == text_question1

    assert history[1].role == "model"
    assert history[1].parts[0].function_call == FunctionCall(
        name=tool_call_1["name"], args=tool_call_1["args"]
    )
    assert history[1].parts[1].function_call == FunctionCall(
        name=tool_call_2["name"], args=tool_call_2["args"]
    )

    assert history[2].role == "user"
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

    assert history[4].role == "user"
    assert history[4].parts[0].function_response == FunctionResponse(
        name=fn_name_3,
        response={"content": message6.content},
    )
    assert history[5].parts[0].text == text_answer1


@pytest.mark.parametrize(
    ("source_history", "expected_sys_message", "expected_history"),
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
    source_history, expected_sys_message, expected_history
) -> None:
    """Test parsing of chat history with multiple messages of same type.

    Needed to ensure that multiple system messages are combined into one, and that
    multiple AI messages are combined into one.
    """
    image_bytes_loader = ImageBytesLoader()
    sm, result_history = _parse_chat_history_gemini(
        history=source_history, imageBytesLoader=image_bytes_loader
    )

    for result, expected in zip(result_history, expected_history, strict=False):
        assert result == expected
    assert sm == expected_sys_message


def test_parse_chat_history_gemini_with_literal_eval() -> None:
    """Test that string content representing a list is parsed correctly."""
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
    """Test that string content is not parsed when literal eval is disabled."""
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


def test_python_literal_inputs() -> None:
    """In relation to literal eval, ensure that inputs are not misinterpreted."""
    llm = ChatVertexAI(model="gemini-2.5-flash", project="test-project")

    for input_string in ["None", "(1, 2)", "[1, 2, 3]", "{1, 2, 3}"]:
        _ = llm._prepare_request_gemini([HumanMessage(input_string)])


@pytest.mark.parametrize(
    ("raw_candidate", "expected"),
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
                content="Mike age is 30Arthur age is 30",
                additional_kwargs={},
            ),
        ),
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            inline_data=Blob(
                                data=base64.b64decode("Qk0eAAAAAABoAAMAQABAEAGAAP8A"),
                                mime_type="image/png",
                            )
                        )
                    ],
                )
            ),
            AIMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,Qk0eAAAAAABoAAMAQABAEAGAAP8A"
                        },
                    }
                ],
                additional_kwargs={},
            ),
        ),
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(text="Here is an image:"),
                        Part(
                            inline_data=Blob(
                                data=base64.b64decode("Qk0eAAAAAABoAAMAQABAEAGAAP8A"),
                                mime_type="image/jpeg",
                            )
                        ),
                    ],
                )
            ),
            AIMessage(
                content=[
                    "Here is an image:",
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,Qk0eAAAAAABoAAMAQABAEAGAAP8A"
                        },
                    },
                ],
                additional_kwargs={},
            ),
        ),
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            inline_data=Blob(
                                data=base64.b64decode("Qk0eAAAAAABoAAMAQABAEAGAAP8A"),
                                mime_type="image/png",
                            )
                        ),
                        Part(
                            inline_data=Blob(
                                data=base64.b64decode("Qk0eAAAAAABoAAMAQABAEAGAAP8A"),
                                mime_type="image/gif",
                            )
                        ),
                    ],
                )
            ),
            AIMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,Qk0eAAAAAABoAAMAQABAEAGAAP8A"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/gif;base64,Qk0eAAAAAABoAAMAQABAEAGAAP8A"
                        },
                    },
                ],
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
        (
            Candidate(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            function_call=FunctionCall(
                                name="my_tool",
                                args={"param1": "value1", "param2": "value2"},
                            ),
                            thought_signature=b"decafe42",
                        ),
                    ],
                )
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "type": "tool_call",
                        "id": "00000000-0000-0000-0000-00000000000",
                        "name": "my_tool",
                        "args": {"param1": "value1", "param2": "value2"},
                    }
                ],
                additional_kwargs={
                    "__gemini_function_call_thought_signatures__": {
                        "00000000-0000-0000-0000-00000000000": _bytes_to_base64(
                            b"decafe42"
                        )
                    }
                },
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


def test_parser_multiple_tools() -> None:
    """Test parsing of multiple function calls into Pydantic models.

    Uses two simple Pydantic models as tools: Add and Multiply. The model is expected
    to call both functions in a single response, and the parser should correctly parse
    the response into instances of the respective Pydantic models.
    """

    class Add(BaseModel):
        arg1: int
        arg2: int

    class Multiply(BaseModel):
        arg1: int
        arg2: int

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mc:
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

        model = ChatVertexAI(model_name="gemini-2.5-pro", project="test-project")
        message = HumanMessage(content="Hello")
        parser = PydanticToolsParser(tools=[Add, Multiply])
        llm = model | parser
        result = llm.invoke([message])
        mock_generate_content.assert_called_once()

        # Verify that the result is a list of Pydantic model instances
        assert isinstance(result, list)
        assert isinstance(result[0], Add)
        assert result[0] == Add(arg1=1, arg2=2)
        assert isinstance(result[1], Multiply)
        assert result[1] == Multiply(arg1=3, arg2=3)


def test_multiple_fc() -> None:
    """Test parsing of multiple function calls in a single response."""
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
            role="user",
        ),
    ]
    assert history == expected


def test_anthropic_format_output() -> None:
    """Test format output handles different content structures correctly."""

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int
        cache_creation_input_tokens: int | None
        cache_read_input_tokens: int | None

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
        "input_tokens": 4,  # 2 + 1 + 1 (original + cache_read + cache_creation)
        "output_tokens": 1,
        "total_tokens": 5,  # 4 + 1
        "input_token_details": {
            "cache_creation": 1,
            "cache_read": 1,
        },
    }


def test_anthropic_format_output_with_chain_of_thoughts() -> None:
    """Test format output handles chain of thoughts correctly."""

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int
        cache_creation_input_tokens: int | None
        cache_read_input_tokens: int | None

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
    assert isinstance(message, AIMessage)
    assert len(message.content) == 3
    assert message.content == test_msg.model_dump()["content"]
    assert message.usage_metadata == {
        "input_tokens": 4,  # 2 + 1 + 1 (original + cache_read + cache_creation)
        "output_tokens": 1,
        "total_tokens": 5,  # 4 + 1
        "input_token_details": {
            "cache_creation": 1,
            "cache_read": 1,
        },
    }


def test_thinking_configuration() -> None:
    """Test that thinking configuration is set correctly."""
    input_message = HumanMessage("Query requiring reasoning.")

    # Test init params
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        project="test-project",
        thinking_budget=100,
        include_thoughts=True,
    )
    request = llm._prepare_request_gemini([input_message])
    assert request.generation_config.thinking_config.thinking_budget == 100
    assert request.generation_config.thinking_config.include_thoughts is True

    # Test invocation params
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, project="test-project")
    request = llm._prepare_request_gemini(
        [input_message],
        thinking_budget=100,
        include_thoughts=True,
    )
    assert request.generation_config.thinking_config.thinking_budget == 100
    assert request.generation_config.thinking_config.include_thoughts is True


def test_thought_signature() -> None:
    """Test that thought signatures are correctly parsed and included in requests."""
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        project="test-project",
        include_thoughts=True,
    )
    thought_signature_bytes = b"decafe42"
    thought_signature_base64 = _bytes_to_base64(thought_signature_bytes)

    history = [
        HumanMessage("Query requiring reasoning."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "type": "tool_call",
                    "id": "abc123",
                    "name": "my_tool",
                    "args": {"param1": "value1", "param2": "value2"},
                },
            ],
            additional_kwargs={
                "__gemini_function_call_thought_signatures__": {
                    "abc123": thought_signature_base64
                }
            },
        ),
        ToolMessage("result", tool_call_id="abc123"),
    ]
    request = llm._prepare_request_gemini(history)
    assert request.contents == [
        Content(role="user", parts=[Part(text="Query requiring reasoning.")]),
        Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="my_tool",
                        args={"param1": "value1", "param2": "value2"},
                    ),
                    thought_signature=thought_signature_bytes,
                ),
            ],
        ),
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="my_tool",
                        response={"content": "result"},
                    )
                ),
            ],
        ),
    ]


def test_v1_function_parts() -> None:
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME, project="test-project", endpoint_version="v1"
    )

    messages = [
        HumanMessage(content="What is 2+2*2?"),
        AIMessage(
            content="I am calling a calculator to evaluate the expression",
            tool_calls=[
                {"name": "calculator", "args": {"expression": "2+2*2"}, "id": "123"}
            ],
        ),
        ToolMessage(content="6", tool_call_id="123"),
    ]

    assert llm._prepare_request_gemini(messages)


def test_thinking_budget_in_params() -> None:
    """Test that `thinking_budget` and `include_thoughts` are configured correctly."""
    # Init params
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        project="test-project",
        thinking_budget=1000,
        include_thoughts=True,
    )

    params = llm._prepare_params()
    # thinking_budget and include_thoughts should NOT be in top-level params
    # (to avoid conflicts with GenerationConfig)
    assert "thinking_budget" not in params
    assert "include_thoughts" not in params

    # But thinking_config should be set for the API
    assert "thinking_config" in params
    assert params["thinking_config"]["thinking_budget"] == 1000
    assert params["thinking_config"]["include_thoughts"] is True

    # Invocation params
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, project="test-project")

    params = llm._prepare_params(thinking_budget=500, include_thoughts=False)
    assert "thinking_budget" not in params
    assert "include_thoughts" not in params

    # Also check that thinking_config is set for the API
    assert "thinking_config" in params
    assert params["thinking_config"]["thinking_budget"] == 500
    assert params["thinking_config"]["include_thoughts"] is False


def test_thinking_budget_in_invocation_params() -> None:
    """Test that thinking parameters are available for LangSmith tracing."""
    # Init params
    llm = ChatVertexAI(
        model=_DEFAULT_MODEL_NAME,
        project="test-project",
        thinking_budget=1000,
        include_thoughts=True,
    )

    invocation_params = llm._get_invocation_params()

    # Verify thinking parameters are included for tracing
    assert "thinking_budget" in invocation_params
    assert "include_thoughts" in invocation_params
    assert invocation_params["thinking_budget"] == 1000
    assert invocation_params["include_thoughts"] is True

    # Invocation params
    llm = ChatVertexAI(model=_DEFAULT_MODEL_NAME, project="test-project")

    invocation_params = llm._get_invocation_params(
        thinking_budget=500, include_thoughts=False
    )

    # Verify thinking parameters are included for tracing
    assert "thinking_budget" in invocation_params
    assert "include_thoughts" in invocation_params
    assert invocation_params["thinking_budget"] == 500
    assert invocation_params["include_thoughts"] is False


def test_json_mode_with_pydantic_v2_fieldinfo_serialization() -> None:
    """Test that json_mode uses serialization mode for Pydantic v2 model_json_schema.

    This ensures that FieldInfo objects are properly serialized when generating
    the JSON schema for structured output in json_mode. Using mode='serialization'
    is semantically correct since the schema describes the model's output format,
    not its input validation rules.
    """
    from pydantic import Field

    class TestModel(BaseModel):
        """Test model with Pydantic v2 FieldInfo metadata."""

        name: str = Field(description="Person's name")
        age: int = Field(gt=0, le=150, description="Person's age")

    llm = ChatVertexAI(model_name="gemini-2.5-flash", project="test-project")

    # This should not raise any errors when creating structured output
    structured_llm = llm.with_structured_output(TestModel, method="json_mode")
    assert structured_llm is not None

    # Verify that model_json_schema works with mode='serialization'
    schema = TestModel.model_json_schema(mode="serialization")
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]
    # Verify field metadata is preserved
    assert "description" in schema["properties"]["name"]
    assert schema["properties"]["name"]["description"] == "Person's name"


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Pydantic v1 compatibility not supported on Python 3.14",
)
def test_json_mode_pydantic_v1_backward_compatibility() -> None:
    """Test that Pydantic v1 models continue to work with json_mode.

    This ensures backward compatibility - Pydantic v1 models use schema()
    method while v2 models use model_json_schema(mode='serialization').
    """
    from pydantic.v1 import BaseModel as BaseModelV1

    class V1Model(BaseModelV1):
        """Test model using Pydantic v1 API."""

        name: str
        age: int

    llm = ChatVertexAI(model_name="gemini-2.5-flash", project="test-project")

    # V1 models should work without issues
    structured_llm = llm.with_structured_output(V1Model, method="json_mode")
    assert structured_llm is not None

    # Verify V1 model uses schema() method, not model_json_schema()
    assert hasattr(V1Model, "schema")
    assert not hasattr(V1Model, "model_json_schema")

    # Verify schema generation works
    schema = V1Model.schema()
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "age" in schema["properties"]


def test_thought_signature_conversion() -> None:
    # Test ReasoningContentBlock with signature
    reasoning_block = {
        "type": "reasoning",
        "reasoning": "Let me think about this...",
        "extras": {"signature": "signature123"},
    }
    result = _convert_from_v1_to_vertex(
        [reasoning_block],  # type: ignore[list-item]
        "google_vertexai",
    )
    expected = [
        {
            "type": "thinking",
            "thinking": "Let me think about this...",
            "thought_signature": "signature123",
        }
    ]
    assert result == expected

    # Test ReasoningContentBlock without signature
    # (e.g. coming from a different integration)
    reasoning_without_sig = {
        "type": "reasoning",
        "reasoning": "Thinking without signature...",
        "extras": {},
    }
    result = _convert_from_v1_to_vertex(
        [reasoning_without_sig],  # type: ignore[list-item]
        "google_vertexai",
    )
    expected_no_sig = [
        {
            "type": "thinking",
            "thinking": "Thinking without signature...",
        }
    ]
    assert result == expected_no_sig

    # Test function_call_signature block is passed through unchanged
    sig_block = {
        "type": "function_call_signature",
        "signature": "sig123",
        "index": 0,
    }
    result = _convert_from_v1_to_vertex(
        [sig_block],  # type: ignore[list-item]
        "google_vertexai",
    )
    expected_sig = [sig_block]
    assert result == expected_sig

    # Test TextContentBlock with signature
    text_block = {
        "type": "text",
        "text": "Hello world",
        "extras": {"signature": "text_sig_123"},
    }
    result = _convert_from_v1_to_vertex(
        [text_block],  # type: ignore[list-item]
        "google_vertexai",
    )
    expected_text = [
        {
            "type": "text",
            "text": "Hello world",
            "thought_signature": "text_sig_123",
        }
    ]
    assert result == expected_text


def test_parse_chat_history_uses_index_for_signature() -> None:
    """Test `_parse_chat_history` uses index field to map signatures to tool calls.

    Also tests passing in a message from GenAI for compatibility.
    """
    sig_bytes = b"dummy_signature"
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

    # Content with reasoning block (index 0) and signature block (index 1)
    # The signature block points to tool call index 0
    content = [
        {"type": "reasoning", "reasoning": "I should use the tool."},
        {"type": "function_call_signature", "signature": sig_b64, "index": 0},
    ]

    tool_calls = [{"name": "my_tool", "args": {"param": "value"}, "id": "call_1"}]

    message = AIMessage(content=content, tool_calls=tool_calls)  # type: ignore[arg-type]
    message.response_metadata["output_version"] = "v1"
    message.response_metadata["model_provider"] = (
        "google_genai"  # Simulate genai message compat
    )

    mock_loader = MagicMock(spec=ImageBytesLoader)

    _, formatted_messages = _parse_chat_history_gemini([message], mock_loader)

    model_content = formatted_messages[0]
    assert model_content.role == "model"
    assert len(model_content.parts) == 2  # thinking + function_call

    # Part 0 is thinking
    assert model_content.parts[0].thought is True
    assert model_content.parts[0].text == "I should use the tool."

    # Part 1 is the function_call
    part = model_content.parts[1]
    assert part.function_call is not None
    assert part.function_call.name == "my_tool"
    assert part.thought_signature == sig_bytes


def test_parse_chat_history_with_text_signature() -> None:
    """Test `_parse_chat_history` handles signatures on text blocks.

    Also tests passing in a message from GenAI for compatibility.
    """
    sig_bytes = b"dummy_signature"
    sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

    content = [
        {"type": "reasoning", "reasoning": "Thinking..."},
        {"type": "text", "text": "Final answer", "extras": {"signature": sig_b64}},
    ]

    message = AIMessage(content=content)  # type: ignore[arg-type]
    message.response_metadata["output_version"] = "v1"
    message.response_metadata["model_provider"] = (
        "google_genai"  # Simulate genai message compat
    )

    mock_loader = MagicMock(spec=ImageBytesLoader)

    _, formatted_messages = _parse_chat_history_gemini([message], mock_loader)

    model_content = formatted_messages[0]
    assert model_content.role == "model"
    assert len(model_content.parts) == 2

    # Part 0 is thinking
    assert model_content.parts[0].thought is True
    assert model_content.parts[0].text == "Thinking..."

    # Part 1 is text with signature
    part = model_content.parts[1]
    assert part.text == "Final answer"
    assert part.thought_signature == sig_bytes


def test_timeout_parameter_override(clear_prediction_client_cache: Any) -> None:
    """Test that timeout can be set in constructor and overridden in invoke."""
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        project="test-project",
        timeout=30.0,  # Default timeout
    )
    assert llm.timeout == 30.0

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_generate = mock_prediction_service.return_value.generate_content
        mock_generate.return_value = response

        # Test 1: Using constructor timeout (no override)
        llm._generate_gemini(messages=[])
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs.get("timeout") == 30.0, (
            "Constructor timeout should be used when no override provided"
        )

        mock_generate.reset_mock()

        # Test 2: Override timeout via kwargs (simulates invoke(..., timeout=5))
        llm._generate_gemini(messages=[], timeout=5.0)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs.get("timeout") == 5.0, (
            "Invoke-time timeout should override constructor timeout"
        )


def test_timeout_parameter_none_override(clear_prediction_client_cache: Any) -> None:
    """Test that timeout=None in invoke uses constructor timeout."""
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        project="test-project",
        timeout=30.0,
    )

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        response = GenerateContentResponse(candidates=[])
        mock_generate = mock_prediction_service.return_value.generate_content
        mock_generate.return_value = response

        # Passing timeout=None explicitly overrides constructor value
        llm._generate_gemini(messages=[], timeout=None)
        call_kwargs = mock_generate.call_args.kwargs
        # When timeout=None is explicitly passed, it uses None (not constructor default)
        assert call_kwargs.get("timeout") is None


def test_get_num_tokens_from_messages(clear_prediction_client_cache: Any) -> None:
    """Test get_num_tokens_from_messages uses count_tokens API properly."""
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        project="test-project",
    )

    # Create a mock response for count_tokens
    mock_count_response = MagicMock()
    mock_count_response.total_tokens = 42

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        mock_count = mock_prediction_service.return_value.count_tokens
        mock_count.return_value = mock_count_response

        # Test with simple text message
        messages = [HumanMessage(content="Hello, world!")]
        token_count = llm.get_num_tokens_from_messages(messages)

        assert token_count == 42
        mock_count.assert_called_once()

        # Verify the call included contents
        call_args = mock_count.call_args
        assert "contents" in call_args[0][0]


def test_get_num_tokens_from_messages_multimodal(
    clear_prediction_client_cache: Any,
) -> None:
    """Test get_num_tokens_from_messages handles multi-modal messages.

    This test verifies that multi-modal content (like images) is properly
    passed to the count_tokens API rather than being converted to a string.
    """
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        project="test-project",
    )

    # Create a mock response for count_tokens
    mock_count_response = MagicMock()
    mock_count_response.total_tokens = 100

    with patch(
        "langchain_google_vertexai._client_utils.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        mock_count = mock_prediction_service.return_value.count_tokens
        mock_count.return_value = mock_count_response

        # Test with multi-modal message containing image
        # Using a small base64 encoded image (1x1 red pixel PNG)
        small_image_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
            "YGD4DwABBAEAcCBlCQAAAABJRU5ErkJggg=="
        )
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{small_image_b64}"
                        },
                    },
                ]
            )
        ]
        token_count = llm.get_num_tokens_from_messages(messages)

        assert token_count == 100
        mock_count.assert_called_once()

        # Verify the call was made with proper content structure
        call_args = mock_count.call_args
        request_dict = call_args[0][0]
        assert "contents" in request_dict
        assert len(request_dict["contents"]) > 0
