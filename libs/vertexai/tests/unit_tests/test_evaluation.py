from unittest.mock import MagicMock, patch

from google.cloud.aiplatform_v1beta1.types import (
    EvaluateInstancesRequest,
    EvaluateInstancesResponse,
)

from langchain_google_vertexai import VertexStringEvaluator
from langchain_google_vertexai.evaluators.evaluation import _prepare_request


def test_prepare_request_rouge() -> None:
    instances = [
        {"prediction": "test1", "reference": "test2"},
        {"prediction": "test3", "reference": "test4"},
    ]
    request = _prepare_request(
        instances, metric="rougeL", location="project/123/location/moon1"
    )
    expected = EvaluateInstancesRequest(
        rouge_input={"metric_spec": {"rouge_type": "rougeL"}, "instances": instances},
        location="project/123/location/moon1",
    )
    assert expected == request


def test_prepare_request_coherence() -> None:
    instance = {"prediction": "test1"}
    request = _prepare_request(
        [instance], metric="coherence", location="project/123/location/moon1"
    )
    expected = EvaluateInstancesRequest(
        coherence_input={"metric_spec": {}, "instance": instance},
        location="project/123/location/moon1",
    )
    assert expected == request


def test_prepare_request_question_answering_correctness() -> None:
    instance = {"prediction": "test1", "instruction": "test2", "context": "test3"}
    request = _prepare_request(
        [instance],
        metric="question_answering_correctness",
        location="project/123/location/moon1",
    )
    expected = EvaluateInstancesRequest(
        question_answering_correctness_input={"metric_spec": {}, "instance": instance},
        location="project/123/location/moon1",
    )
    assert expected == request


def test_evaluate():
    with patch(
        "langchain_google_vertexai.evaluators.evaluation.EvaluationServiceClient"
    ) as mc:
        with patch(
            "langchain_google_vertexai.evaluators.evaluation.EvaluationServiceAsyncClient"
        ) as amc:
            evaluator = VertexStringEvaluator(
                metric="bleu", project_id="test", location="moon1"
            )
            mc.assert_called_once()
            amc.assert_called_once()
            evaluator._location = "test"

            mock_evaluate = MagicMock()
            mock_evaluate.return_value = EvaluateInstancesResponse(
                bleu_results={"bleu_metric_values": [{"score": 1.0}, {"score": 0.5}]}
            )
            mc.return_value.evaluate_instances = mock_evaluate

            result = evaluator.evaluate(
                examples=[{"reference": "test1"}, {"reference": "test2"}],
                predictions=[{"prediction": "test3"}, {"prediction": "test4"}],
            )
            mock_evaluate.assert_called_once()
            assert result == [{"score": 1.0}, {"score": 0.5}]
