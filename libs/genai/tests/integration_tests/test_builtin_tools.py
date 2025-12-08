from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_genai import ChatGoogleGenerativeAI

_MODEL = "gemini-2.5-flash"
_COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"


def test_url_context_tool() -> None:
    model = ChatGoogleGenerativeAI(model=_MODEL)
    model_with_search = model.bind_tools([{"url_context": {}}])

    input = "What is this page's contents about? https://docs.langchain.com"
    response = model_with_search.invoke(input)
    assert isinstance(response, AIMessage)

    assert (
        response.response_metadata["grounding_metadata"]["grounding_chunks"][0]["web"][
            "uri"
        ]
        is not None
    )


def test_computer_use_tool() -> None:
    """Test computer_use built-in tool (Issue #1243)."""
    llm = ChatGoogleGenerativeAI(model=_COMPUTER_USE_MODEL)
    llm_with_cu = llm.bind_tools([{"computer_use": {"environment": "browser"}}])

    response = llm_with_cu.invoke(
        [HumanMessage(content="Open a web browser and navigate to google.com")]
    )

    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) > 0
    # Common actions: open_web_browser, navigate, click_at, type_text_at
    assert response.tool_calls[0]["name"] in [
        "open_web_browser",
        "navigate",
        "click_at",
        "type_text_at",
        "scroll_document",
    ]
