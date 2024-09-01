from langchain_google_genai.llms import GoogleGenerativeAI, GoogleModelFamily


def test_model_family() -> None:
    model = GoogleModelFamily("gemini-pro")
    assert model == GoogleModelFamily.GEMINI
    model = GoogleModelFamily("gemini-ultra")
    assert model == GoogleModelFamily.GEMINI


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="foo")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_temperature": 0.7,
    }

    llm = GoogleGenerativeAI(
        model="gemini-pro", temperature=0.1, max_output_tokens=10, google_api_key="foo"
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }
