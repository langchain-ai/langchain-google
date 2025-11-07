from unittest.mock import ANY, Mock, patch

from google.ai.generativelanguage_v1beta.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    Part,
)
from pydantic import SecretStr

from langchain_google_genai.llms import GoogleGenerativeAI

MODEL_NAME = "gemini-flash-lite-latest"


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
    """Test that base_url is properly passed through to ChatGoogleGenerativeAI."""
    mock_client = Mock()
    mock_generate_content = Mock()
    mock_generate_content.return_value = GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text="test response")]))]
    )
    mock_client.return_value.generate_content = mock_generate_content
    base_url = "https://example.com"
    param_api_key = "[secret]"
    param_secret_api_key = SecretStr(param_api_key)
    param_transport = "rest"

    with patch(
        "langchain_google_genai._genai_extension.v1betaGenerativeServiceClient",
        mock_client,
    ):
        llm = GoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=param_secret_api_key,
            base_url=base_url,
            transport=param_transport,
        )

    response = llm.invoke("test")
    assert response == "test response"

    mock_client.assert_called_once_with(
        transport=param_transport,
        client_options=ANY,
        client_info=ANY,
    )
    call_client_options = mock_client.call_args_list[0].kwargs["client_options"]
    assert call_client_options.api_key == param_api_key
    assert call_client_options.api_endpoint == base_url
