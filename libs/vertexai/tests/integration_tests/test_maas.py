import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
)

from langchain_google_vertexai.model_garden_maas.mistral import (
    VertexModelGardenMistral,
)

model_names = [
    "mistral-nemo@2407",
    "mistral-large@2407",
]


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_generate(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.invoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_agenerate(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = await llm.ainvoke("What is the meaning of life?")
    assert isinstance(output, AIMessage)
    print(output)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
def test_stream(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.stream("What is the meaning of life?")
    for chunk in output:
        assert isinstance(chunk, AIMessageChunk)


@pytest.mark.extended
@pytest.mark.parametrize("model_name", model_names)
async def test_astream(model_name: str) -> None:
    llm = VertexModelGardenMistral(model=model_name, location="us-central1")
    output = llm.astream("What is the meaning of life?")
    async for chunk in output:
        assert isinstance(chunk, AIMessageChunk)
