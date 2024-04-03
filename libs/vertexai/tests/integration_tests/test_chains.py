from typing import Optional

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, create_structured_runnable


class RecordPerson(BaseModel):
    """Record some identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")


class RecordDog(BaseModel):
    """Record some identifying information about a dog."""

    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: Optional[str] = Field(None, description="The dog's favorite food")


@pytest.mark.release
def test_create_structured_runnable() -> None:
    llm = ChatVertexAI(model_name="gemini-pro")
    prompt = ChatPromptTemplate.from_template(
        "You are a world class algorithm for recording entities.\nMake calls to the "
        "relevant function to record the entities in the following input:\n {input}\n"
        "Tip: Make sure to answer in the correct format"
    )
    chain = create_structured_runnable([RecordPerson, RecordDog], llm, prompt=prompt)
    res = chain.invoke({"input": "Harry was a chubby brown beagle who loved chicken"})
    assert isinstance(res, RecordDog)


@pytest.mark.release
def test_create_structured_runnable_with_prompt() -> None:
    llm = ChatVertexAI(model_name="gemini-pro")
    prompt = ChatPromptTemplate.from_template(
        "Describe a random {class} and mention their name, {attr} and favorite food"
    )
    chain = create_structured_runnable(
        [RecordPerson, RecordDog], llm, prompt=prompt, use_extra_step=True
    )
    res = chain.invoke({"class": "person", "attr": "age"})
    assert isinstance(res, RecordPerson)
