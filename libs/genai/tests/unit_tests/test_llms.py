from unittest.mock import ANY, Mock, patch

from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)
from pydantic import SecretStr

from langchain_google_genai.llms import GoogleGenerativeAI

MODEL_NAME = "gemini-2.5-flash"


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = GoogleGenerativeAI(model=MODEL_NAME, google_api_key="foo")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_temperature": 0.7,
    }

    llm = GoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0.1,
        max_output_tokens=10,
        google_api_key="foo",
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": MODEL_NAME,
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }

    # Test initialization with an invalid argument to check warning
    with patch("langchain_google_genai.llms.logger.warning") as mock_warning:
        llm = GoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key="foo",
            safety_setting={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
            },  # Invalid arg
        )
        assert llm.model == f"models/{MODEL_NAME}"
        ls_params = llm._get_ls_params()
        assert ls_params.get("ls_model_name") == MODEL_NAME
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]
        assert "Unexpected argument 'safety_setting'" in call_args
        assert "Did you mean: 'safety_settings'?" in call_args


def test_base_url_support() -> None:
    """Test that `base_url` is properly passed through to `ChatGoogleGenerativeAI`."""
    mock_client_instance = Mock()
    mock_models = Mock()
    mock_generate_content = Mock()

    # Create a proper mock response with the required attributes
    mock_response = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="test response")]))],
        prompt_feedback=None,  # This is optional and can be None
    )
    mock_generate_content.return_value = mock_response
    mock_models.generate_content = mock_generate_content
    mock_client_instance.models = mock_models

    mock_client_class = Mock()
    mock_client_class.return_value = mock_client_instance

    base_url = "https://example.com"
    param_api_key = "[secret]"
    param_secret_api_key = SecretStr(param_api_key)
    param_transport = "rest"

    with patch(
        "langchain_google_genai.chat_models.Client",
        mock_client_class,
    ):
        llm = GoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            base_url=base_url,
            transport=param_transport,
        )

        response = llm.invoke("test")
        assert response == "test response"

    mock_client_class.assert_called_once_with(
        api_key=param_api_key,
        http_options=ANY,
    )
    call_http_options = mock_client_class.call_args_list[0].kwargs["http_options"]
    assert call_http_options.base_url == base_url
