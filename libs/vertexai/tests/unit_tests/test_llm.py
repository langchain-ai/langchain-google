from typing import Any, Dict
from unittest import TestCase
from unittest.mock import MagicMock, patch

from langchain_google_vertexai.llms import VertexAI


def test_model_name() -> None:
    for llm in [
        VertexAI(model_name="gemini-pro", project="test-project", max_output_tokens=10),
        VertexAI(model="gemini-pro", project="test-project", max_tokens=10),
    ]:
        assert llm.model_name == "gemini-pro"
        assert llm.max_output_tokens == 10


def test_tuned_model_name() -> None:
    llm = VertexAI(
        model_name="gemini-pro",
        project="test-project",
        tuned_model_name="projects/123/locations/europe-west4/endpoints/456",
    )
    assert llm.model_name == "gemini-pro"
    assert llm.tuned_model_name == "projects/123/locations/europe-west4/endpoints/456"
    assert llm.client._model_name == "projects/123/locations/europe-west4/endpoints/456"


def test_vertexai_args_passed() -> None:
    response_text = "Goodbye"
    user_prompt = "Hello"
    prompt_params: Dict[str, Any] = {
        "max_output_tokens": 1,
        "temperature": 0,
        "top_k": 10,
        "top_p": 0.5,
    }

    # Mock the library to ensure the args are passed correctly
    with patch("langchain_google_vertexai.llms.GenerativeModel") as model:
        with patch("langchain_google_vertexai.llms.get_generation_info") as gen_info:
            gen_info.return_value = {}
            mock_response = MagicMock()
            candidate = MagicMock()
            candidate.text = response_text
            mock_response.candidates = [candidate]
            model_instance = MagicMock()
            model_instance.generate_content.return_value = mock_response
            model.return_value = model_instance

            llm = VertexAI(model_name="gemini-pro", **prompt_params)
            response = llm.invoke("Hello")
            assert response == response_text
            model_instance.generate_content.assert_called_once

            assert model_instance.generate_content.call_args.args[0] == [user_prompt]
            TestCase().assertCountEqual(
                model_instance.generate_content.call_args.kwargs,
                {
                    "stream": False,
                    "safety_settings": None,
                    "generation_config": {
                        "max_output_tokens": 1,
                        "temperature": 0,
                        "top_k": 10,
                        "top_p": 0.5,
                        "stop_sequences": None,
                    },
                },
            )
            assert (
                model_instance.generate_content.call_args.kwargs["generation_config"][
                    "temperature"
                ]
                == 0
            )


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = VertexAI(model_name="gemini-pro")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_vertexai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
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
