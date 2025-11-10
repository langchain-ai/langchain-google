import pytest
from langchain_core.messages import (
    AIMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_google_vertexai import ChatVertexAI, create_structured_runnable
from tests.integration_tests.conftest import _DEFAULT_MODEL_NAME


class RecordPerson(BaseModel):
    """Record some identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: str | None = Field(default=None, description="The person's favorite food")


class RecordDog(BaseModel):
    """Record some identifying information about a dog."""

    name: str = Field(..., description="The dog's name")
    color: str = Field(..., description="The dog's color")
    fav_food: str | None = Field(default=None, description="The dog's favorite food")


@pytest.mark.release
def test_create_structured_runnable() -> None:
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)
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
    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Describe a random {class} and mention their name, {attr} and favorite food"
    )
    chain = create_structured_runnable(
        [RecordPerson, RecordDog], llm, prompt=prompt, use_extra_step=True
    )
    res = chain.invoke({"class": "person", "attr": "age"})
    assert isinstance(res, RecordPerson)


@pytest.mark.release
def test_reflection() -> None:
    class Reflection(BaseModel):
        reflections: str = Field(
            description="The critique and reflections on the sufficiency, superfluency,"
            " and general quality of the response"
        )
        score: int = Field(
            description="Score from 0-10 on the quality of the candidate response.",
            # gte=0,
            # lte=10,
        )
        found_solution: bool = Field(
            description="Whether the response has fully solved the question or task."
        )

        def as_message(self):
            return AIMessage(
                content=f"Reasoning: {self.reflections}\nScore: {self.score}"
            )

        @property
        def normalized_score(self) -> float:
            return self.score / 10.0

    llm = ChatVertexAI(model_name=_DEFAULT_MODEL_NAME)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Reflect and grade the assistant response to the user question below.",
            ),
            (
                "user",
                "Which planet is the closest to the Earth?",
            ),
            ("ai", "{input}"),
        ]
    )

    reflection_llm_chain = prompt | llm.with_structured_output(Reflection)
    res = reflection_llm_chain.invoke({"input": "Mars"})
    assert isinstance(res, Reflection)
