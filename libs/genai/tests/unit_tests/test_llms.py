from unittest.mock import patch

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
