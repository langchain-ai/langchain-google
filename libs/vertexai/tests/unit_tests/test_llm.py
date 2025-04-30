from typing import Any, Dict
from unittest.mock import MagicMock, patch

from google.cloud.aiplatform_v1beta1.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    GenerationConfig,
    Part,
)
from pydantic import model_validator
from typing_extensions import Self

from langchain_google_vertexai._base import _BaseVertexAIModelGarden
from langchain_google_vertexai.llms import VertexAI


def test_model_name() -> None:
    for llm in [
        VertexAI(model_name="gemini-pro", project="test-project", max_output_tokens=10),
        VertexAI(model="gemini-pro", project="test-project", max_tokens=10),
    ]:
        assert llm.model_name == "gemini-pro"
        assert llm.max_output_tokens == 10

    # Test initialization with an invalid argument to check warning
    with patch("langchain_google_vertexai.llms.logger.warning") as mock_warning:
        llm = VertexAI(
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


def test_tuned_model_name() -> None:
    llm = VertexAI(
        model_name="gemini-pro",
        project="test-project",
        tuned_model_name="projects/123/locations/europe-west4/endpoints/456",
    )
    assert llm.model_name == "gemini-pro"
    assert llm.tuned_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert (
        llm.client.full_model_name
        == "projects/123/locations/europe-west4/endpoints/456"
    )


def test_vertexai_args_passed() -> None:
    response_text = "Goodbye"
    user_prompt = "Hello"
    prompt_params: Dict[str, Any] = {
        "max_output_tokens": 1,
        "temperature": 0,
        "top_k": 10,
        "top_p": 0.5,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
    }

    # Mock the library to ensure the args are passed correctly
    with patch(
        "langchain_google_vertexai._base.v1beta1PredictionServiceClient"
    ) as mock_prediction_service:
        mock_generate_content = MagicMock(
            return_value=GenerateContentResponse(
                candidates=[
                    Candidate(content=Content(parts=[Part(text=response_text)]))
                ]
            )
        )
        mock_prediction_service.return_value.generate_content = mock_generate_content

        llm = VertexAI(model_name="gemini-pro", **prompt_params)
        response = llm.invoke(
            user_prompt, temperature=0.5, frequency_penalty=0.5, presence_penalty=0.5
        )
        assert response == response_text
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
            temperature=0.5,
            top_p=0.5,
            top_k=10,
            max_output_tokens=1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        assert (
            mock_generate_content.call_args.kwargs["request"].generation_config
            == expected
        )
        assert mock_generate_content.call_args.kwargs["request"].tools == []
        assert not mock_generate_content.call_args.kwargs["request"].tool_config
        assert not mock_generate_content.call_args.kwargs["request"].safety_settings


def test_extract_response() -> None:
    class FakeModelGarden(_BaseVertexAIModelGarden):
        @model_validator(mode="after")
        def validate_environment(self) -> Self:
            return self

    prompts_results = [
        ("a prediction", "a prediction"),
        ("Prompt:\na prompt\nOutput:\na prediction", "a prediction"),
        (
            "Prompt:\na prompt\nOutput:\nFake output\nOutput:\na prediction",
            "a prediction",
        ),
        ("Prompt:\na prompt\nNo Output", "Prompt:\na prompt\nNo Output"),
    ]
    model = FakeModelGarden(endpoint_id="123", result_arg="result", credentials="Fake")
    for original_result, result in prompts_results:
        assert model._parse_prediction(original_result) == result
        assert model._parse_prediction({"result": original_result}) == result

    model = FakeModelGarden(endpoint_id="123", result_arg=None)

    class MyResult:
        def __init__(self, result):
            self.result = result

    for original_result, result in prompts_results:
        my_result = MyResult(original_result)
        assert model._parse_prediction(my_result) == my_result


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = VertexAI(model_name="gemini-pro")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_vertexai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_max_tokens": 128,
        "ls_temperature": 0.0,
    }

    llm = VertexAI(model_name="gemini-pro", temperature=0.1, max_output_tokens=10)
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_vertexai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }
