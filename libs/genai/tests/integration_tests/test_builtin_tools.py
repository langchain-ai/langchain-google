from langchain_core.messages import AIMessage

from langchain_google_genai import ChatGoogleGenerativeAI

_MODEL = "gemini-2.5-flash"


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
