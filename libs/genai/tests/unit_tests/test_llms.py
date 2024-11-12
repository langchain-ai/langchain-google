from langchain_google_genai.llms import (
    GoogleGenerativeAI,
    GoogleModelFamily,
    _strip_erroneous_characters,
)


def test_model_family() -> None:
    model = GoogleModelFamily("gemini-pro")
    assert model == GoogleModelFamily.GEMINI
    model = GoogleModelFamily("gemini-ultra")
    assert model == GoogleModelFamily.GEMINI


def test_tracing_params() -> None:
    # Test standard tracing params
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="foo")  # type: ignore[call-arg]
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_temperature": 0.7,
    }

    llm = GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        max_output_tokens=10,
        google_api_key="foo",  # type: ignore[call-arg]
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "google_genai",
        "ls_model_type": "llm",
        "ls_model_name": "gemini-pro",
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }


def test_strip_erroneous_characters_leading_spaces() -> None:
    """Test that leading spaces are stripped from lines > 1."""
    input_text = "First line\n Second line\n Third line"
    expected_output = "First line\nSecond line\nThird line"
    assert _strip_erroneous_characters(input_text) == expected_output


def test_strip_erroneous_characters_trailing_newlines() -> None:
    """Test that trailing newlines are stripped."""
    input_text = "First line\nSecond line\nThird line\n\n"
    expected_output = "First line\nSecond line\nThird line"
    assert _strip_erroneous_characters(input_text) == expected_output


def test_strip_erroneous_characters_both() -> None:
    """Test that both leading spaces and trailing newlines are stripped."""
    input_text = "First line\n Second line\n Third line\n\n"
    expected_output = "First line\nSecond line\nThird line"
    assert _strip_erroneous_characters(input_text) == expected_output


def test_strip_erroneous_characters_no_changes() -> None:
    """Test that text with no erroneous characters remains unchanged."""
    input_text = "First line\nSecond line\nThird line"
    expected_output = "First line\nSecond line\nThird line"
    assert _strip_erroneous_characters(input_text) == expected_output


def test_strip_erroneous_characters_empty_string() -> None:
    """Test that an empty string remains unchanged."""
    input_text = ""
    expected_output = ""
    assert _strip_erroneous_characters(input_text) == expected_output
