from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI


class StreamingLLMCallbackHandler(BaseCallbackHandler):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokens: List[Any] = []
        self.generations: List[Any] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.generations.append(response.generations[0][0].text)


def test_streaming_callback() -> None:
    prompt_template = "Tell me details about the Company {name} with 2 bullet point?"
    cb = StreamingLLMCallbackHandler()
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-001", callbacks=[cb])
    llm_chain = PromptTemplate.from_template(prompt_template) | llm
    for t in llm_chain.stream({"name": "Google"}):
        pass
    assert len(cb.tokens) > 1
    assert len(cb.generations) == 1
