"""Test Vertex AI API wrapper.

Your end-user credentials would be used to make the calls (make sure you've run 
`gcloud auth login` first).
"""

import pytest
from langchain_core.outputs import LLMResult

from langchain_google_vertexai.llms import VertexAI

model_names_to_test = ["text-bison@001", "gemini-pro"]
model_names_to_test_with_default = [None] + model_names_to_test


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_initialization(model_name: str) -> None:
    llm = VertexAI(model_name=model_name) if model_name else VertexAI()
    assert llm._llm_type == "vertexai"
    try:
        assert llm.model_name == llm.client._model_id
    except AttributeError:
        assert llm.model_name == llm.client._model_name.split("/")[-1]


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_invoke(model_name: str) -> None:
    llm = (
        VertexAI(model_name=model_name, temperature=0)
        if model_name
        else VertexAI(temperature=0.0)
    )
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_vertex_generate(model_name: str) -> None:
    llm = (
        VertexAI(model_name=model_name, temperature=0)
        if model_name
        else VertexAI(temperature=0.0)
    )
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    usage_metadata = output.generations[0][0].generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) == 3
    assert int(usage_metadata["candidates_token_count"]) > 0


@pytest.mark.xfail(reason="VertexAI doesn't always respect number of candidates")
def test_vertex_generate_multiple_candidates() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="text-bison@001")
    output = llm.generate(["Say foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2


@pytest.mark.xfail(reason="VertexAI doesn't always respect number of candidates")
def test_vertex_generate_code() -> None:
    llm = VertexAI(temperature=0.3, n=2, model_name="code-bison@001")
    output = llm.generate(["generate a python method that says foo:"])
    assert isinstance(output, LLMResult)
    assert len(output.generations) == 1
    assert len(output.generations[0]) == 2
    usage_metadata = output.generations[0][0].generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) == 3
    assert int(usage_metadata["candidates_token_count"]) > 1


async def test_vertex_agenerate() -> None:
    llm = VertexAI(temperature=0)
    output = await llm.agenerate(["Please say foo:"])
    assert isinstance(output, LLMResult)
    usage_metadata = output.generations[0][0].generation_info["usage_metadata"]  # type: ignore
    assert int(usage_metadata["prompt_token_count"]) == 4
    assert int(usage_metadata["candidates_token_count"]) > 0


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test_with_default,
)
def test_stream(model_name: str) -> None:
    llm = (
        VertexAI(temperature=0, model_name=model_name)
        if model_name
        else VertexAI(temperature=0)
    )
    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token, str)


async def test_vertex_consistency() -> None:
    llm = VertexAI(temperature=0)
    output = llm.generate(["Please say foo:"])
    streaming_output = llm.generate(["Please say foo:"], stream=True)
    async_output = await llm.agenerate(["Please say foo:"])
    assert output.generations[0][0].text == streaming_output.generations[0][0].text
    assert output.generations[0][0].text == async_output.generations[0][0].text


async def test_astream() -> None:
    llm = VertexAI(temperature=0, model_name="gemini-pro")
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token, str)


@pytest.mark.parametrize(
    "model_name",
    model_names_to_test,
)
def test_vertex_call_count_tokens(model_name: str) -> None:
    llm = VertexAI(model_name=model_name)
    output = llm.get_num_tokens("How are you?")
    assert output == 4
