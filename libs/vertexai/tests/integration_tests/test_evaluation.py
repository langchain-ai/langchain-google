import os

import pytest

from langchain_google_vertexai import (
    VertexPairWiseStringEvaluator,
    VertexStringEvaluator,
)


@pytest.mark.extended
def test_evaluate() -> None:
    evaluator = VertexStringEvaluator(
        metric="bleu", project_id=os.environ["PROJECT_ID"]
    )
    result = evaluator.evaluate(
        examples=[
            {"reference": "This is a test."},
            {"reference": "This is another test."},
        ],
        predictions=[
            {"prediction": "This is a test."},
            {"prediction": "This is another one."},
        ],
    )
    assert len(result) == 2
    assert result[0]["score"] == 1.0
    assert result[1]["score"] < 1.0


@pytest.mark.extended
@pytest.mark.flaky(retries=3)
def test_evaluate_strings() -> None:
    evaluator = VertexStringEvaluator(
        metric="safety", project_id=os.environ["PROJECT_ID"]
    )
    result = evaluator._evaluate_strings(prediction="This is a test")
    assert isinstance(result, dict)
    assert "score" in result
    assert "explanation" in result


@pytest.mark.extended
@pytest.mark.flaky(retries=3)
async def test_aevaluate_strings() -> None:
    evaluator = VertexStringEvaluator(
        metric="question_answering_quality", project_id=os.environ["PROJECT_ID"]
    )
    result = await evaluator._aevaluate_strings(
        prediction="London",
        input="What is the capital of Great Britain?",
        instruction="Be concise",
    )
    assert isinstance(result, dict)
    assert "score" in result
    assert "explanation" in result


@pytest.mark.extended
@pytest.mark.xfail(reason="TODO: investigate (started failing 2025-03-25).")
async def test_evaluate_pairwise() -> None:
    evaluator = VertexPairWiseStringEvaluator(
        metric="pairwise_question_answering_quality",
        project_id=os.environ["PROJECT_ID"],
    )
    result = evaluator.evaluate_string_pairs(
        prediction="London",
        prediction_b="Berlin",
        input="What is the capital of Great Britain?",
        instruction="Be concise",
    )
    assert isinstance(result, dict)
    assert "confidence" in result
    assert "explanation" in result
    assert result["pairwise_choice"] == "BASELINE"
