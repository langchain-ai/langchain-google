import pytest
from langchain_core.messages import HumanMessage

from langchain_google_vertexai.callbacks import VertexAICallbackHandler
from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.llms import VertexAI
from tests.integration_tests.conftest import _DEFAULT_MODEL_NAME


@pytest.mark.release
@pytest.mark.parametrize(
    "model_name",
    [_DEFAULT_MODEL_NAME],
)
def test_llm_invoke(model_name: str) -> None:
    vb = VertexAICallbackHandler()
    llm = VertexAI(model_name=model_name, temperature=0.0, callbacks=[vb])
    _ = llm.invoke("2+2")
    assert vb.successful_requests == 1
    assert vb.prompt_tokens > 0
    assert vb.completion_tokens > 0
    prompt_tokens = vb.prompt_tokens
    completion_tokens = vb.completion_tokens
    _ = llm.invoke("2+2")
    assert vb.successful_requests == 2
    assert vb.prompt_tokens > prompt_tokens
    assert vb.completion_tokens > completion_tokens


@pytest.mark.release
@pytest.mark.parametrize(
    "model_name",
    [_DEFAULT_MODEL_NAME],
)
def test_chat_call(model_name: str) -> None:
    vb = VertexAICallbackHandler()
    llm = ChatVertexAI(model_name=model_name, temperature=0.0, callbacks=[vb])
    message = HumanMessage(content="Hello")
    _ = llm.invoke([message])
    assert vb.successful_requests == 1
    assert vb.prompt_tokens > 0
    assert vb.completion_tokens > 0
    prompt_tokens = vb.prompt_tokens
    completion_tokens = vb.completion_tokens
    _ = llm.invoke([message])
    assert vb.successful_requests == 2
    assert vb.prompt_tokens > prompt_tokens
    assert vb.completion_tokens > completion_tokens


@pytest.mark.release
@pytest.mark.parametrize(
    "model_name",
    [_DEFAULT_MODEL_NAME],
)
def test_invoke_config(model_name: str) -> None:
    vb = VertexAICallbackHandler()
    llm = VertexAI(model_name=model_name, temperature=0.0)
    llm.invoke("2+2", config={"callbacks": [vb]})
    assert vb.successful_requests == 1
    assert vb.prompt_tokens > 0
    assert vb.completion_tokens > 0
    prompt_tokens = vb.prompt_tokens
    completion_tokens = vb.completion_tokens
    llm.invoke("2+2", config={"callbacks": [vb]})
    assert vb.successful_requests == 2
    assert vb.prompt_tokens > prompt_tokens
    assert vb.completion_tokens > completion_tokens


@pytest.mark.release
def test_llm_stream() -> None:
    vb = VertexAICallbackHandler()
    llm = VertexAI(model_name=_DEFAULT_MODEL_NAME, temperature=0.0, callbacks=[vb])
    for _ in llm.stream("2+2"):
        pass
    assert vb.successful_requests == 1
    assert vb.prompt_tokens > 0
    assert vb.completion_tokens > 0
