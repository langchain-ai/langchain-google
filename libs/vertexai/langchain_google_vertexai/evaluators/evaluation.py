from abc import ABC
from collections.abc import Sequence
from typing import Any

from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform.constants import base as constants
from google.cloud.aiplatform_v1beta1 import (
    EvaluationServiceAsyncClient,
    EvaluationServiceClient,
)
from google.cloud.aiplatform_v1beta1.types import (
    EvaluateInstancesRequest,
    EvaluateInstancesResponse,
)
from google.protobuf.json_format import MessageToDict

from langchain_google_vertexai._utils import (
    get_client_info,
    get_user_agent,
)
from langchain_google_vertexai.evaluators._core import (
    PairwiseStringEvaluator,
    StringEvaluator,
)

_METRICS = [
    "bleu",
    "exact_match",
    "rouge",
    "coherence",
    "fluency",
    "safety",
    "groundedness",
    "fulfillment",
    "summarization_quality",
    "summarization_helpfulness",
    "summarization_verbosity",
    "question_answering_quality",
    "question_answering_relevance",
    "question_answering_correctness",
]
_PAIRWISE_METRICS = [
    "pairwise_question_answering_quality",
    "pairwise_summarization_quality",
]
_METRICS_INPUTS = {
    "rouge1": {"rouge_type": "rouge1"},
    "rouge2": {"rouge_type": "rouge2"},
    "rougeL": {"rouge_type": "rougeL"},
    "rougeLsum": {"rouge_type": "rougeLsum"},
}
_METRICS_ATTRS = {
    "safety": ["prediction"],
    "coherence": ["prediction"],
    "fluency": ["prediction"],
    "groundedness": ["context", "prediction"],
    "fulfillment": ["prediction", "instruction"],
    "summarization_quality": ["prediction", "instruction", "context"],
    "summarization_helpfulness": ["prediction", "context"],
    "summarization_verbosity": ["prediction", "context"],
    "question_answering_quality": ["prediction", "context", "instruction"],
    "question_answering_relevance": ["prediction", "instruction"],
    "question_answering_correctness": ["prediction", "instruction"],
    "pairwise_question_answering_quality": [
        "prediction",
        "baseline_prediction",
        "context",
        "instruction",
    ],
    "pairwise_summarization_quality": [
        "prediction",
        "baseline_prediction",
        "context",
        "instruction",
    ],
}
_METRICS_OPTIONAL_ATTRS = {
    "summarization_quality": ["reference"],
    "summarization_helpfulness": ["reference", "instruction"],
    "summarization_verbosity": ["reference", "instruction"],
    "question_answering_quality": ["reference"],
    "question_answering_relevance": ["reference", "context"],
    "question_answering_correctness": ["reference", "context"],
    "pairwise_question_answering_quality": ["reference"],
    "pairwise_summarization_quality": ["reference"],
}
# a client supports multiple instances per request for these metrics
_METRICS_MULTIPLE_INSTANCES = ["bleu", "exact_match", "rouge"]


def _format_metric(metric: str) -> str:
    if metric.startswith("rouge"):
        return "rouge"
    return metric


def _format_instance(instance: dict[str, str], metric: str) -> dict[str, str]:
    attrs = _METRICS_ATTRS.get(metric, ["prediction", "reference"])
    result = {a: instance[a] for a in attrs}
    for attr in _METRICS_OPTIONAL_ATTRS.get(metric, []):
        if attr in instance:
            result[attr] = instance[attr]
    return result


def _prepare_request(
    instances: Sequence[dict[str, str]], metric: str, location: str
) -> EvaluateInstancesRequest:
    request = EvaluateInstancesRequest()
    metric_input: dict[str, Any] = {"metric_spec": _METRICS_INPUTS.get(metric, {})}
    if _format_metric(metric) not in _METRICS_MULTIPLE_INSTANCES:
        if len(instances) > 1:
            msg = (
                f"Metric {metric} supports only a single instance per request, "
                f"got {len(instances)}!"
            )
            raise ValueError(msg)
        metric_input["instance"] = _format_instance(instances[0], metric=metric)
    else:
        metric_input["instances"] = [
            _format_instance(i, metric=metric) for i in instances
        ]
    setattr(request, f"{_format_metric(metric)}_input", metric_input)
    request.location = location
    return request


def _parse_response(
    response: EvaluateInstancesResponse, metric: str
) -> list[dict[str, Any]]:
    metric = _format_metric(metric)
    result = MessageToDict(response._pb, preserving_proto_field_name=True)
    if metric in _METRICS_MULTIPLE_INSTANCES:
        return result[f"{metric}_results"][f"{metric}_metric_values"]
    return [result[f"{metric}_result"]]


class _EvaluatorBase(ABC):
    @property
    def _user_agent(self) -> str:
        """Gets the User Agent."""
        _, user_agent = get_user_agent(f"{type(self).__name__}_{self._metric}")
        return user_agent

    def __init__(
        self, metric: str, project_id: str, location: str = "us-central1"
    ) -> None:
        self._metric = metric
        client_options = ClientOptions(
            api_endpoint=f"{location}-{constants.PREDICTION_API_BASE_PATH}"
        )
        self._client = EvaluationServiceClient(
            client_options=client_options,
            client_info=get_client_info(module=self._user_agent),
        )
        self._async_client = EvaluationServiceAsyncClient(
            client_options=client_options,
            client_info=get_client_info(module=self._user_agent),
        )
        self._location = self._client.common_location_path(project_id, location)

    def _prepare_request(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> EvaluateInstancesRequest:
        instance = {"prediction": prediction}
        if reference:
            instance["reference"] = reference
        if input:
            instance["context"] = input
        instance = {**instance, **kwargs}
        return _prepare_request(
            [instance], metric=self._metric, location=self._location
        )


class VertexStringEvaluator(_EvaluatorBase, StringEvaluator):
    """Evaluate the perplexity of a predicted string."""

    def __init__(self, metric: str, **kwargs) -> None:
        super().__init__(metric, **kwargs)
        if _format_metric(metric) not in _METRICS:
            msg = f"Metric {metric} is not supported yet!"
            raise ValueError(msg)

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict:
        request = self._prepare_request(prediction, reference, input, **kwargs)
        response = self._client.evaluate_instances(request)
        return _parse_response(response, metric=self._metric)[0]

    def evaluate(
        self,
        examples: Sequence[dict[str, str]],
        predictions: Sequence[dict[str, str]],
        *,
        question_key: str = "context",
        answer_key: str = "reference",
        prediction_key: str = "prediction",
        instruction_key: str = "instruction",
        **kwargs: Any,
    ) -> list[dict]:
        instances: list[dict] = []
        for example, prediction in zip(examples, predictions, strict=False):
            row = {"prediction": prediction[prediction_key]}
            if answer_key in example:
                row["reference"] = example[answer_key]
            if question_key in example:
                row["context"] = example[question_key]
            if instruction_key in example:
                row["instruction"] = example[instruction_key]
            instances.append(row)

        if self._metric in _METRICS_MULTIPLE_INSTANCES:
            request = _prepare_request(
                instances, metric=self._metric, location=self._location
            )
            response = self._client.evaluate_instances(request)
            return _parse_response(response, metric=self._metric)
        return [self._evaluate_strings(**i) for i in instances]

    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict:
        request = self._prepare_request(prediction, reference, input, **kwargs)
        response = await self._async_client.evaluate_instances(request)
        return _parse_response(response, metric=self._metric)[0]


class VertexPairWiseStringEvaluator(_EvaluatorBase, PairwiseStringEvaluator):
    """Evaluate the perplexity of a predicted string."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return True

    def __init__(self, metric: str, **kwargs) -> None:
        super().__init__(metric, **kwargs)
        if _format_metric(metric) not in _PAIRWISE_METRICS:
            msg = f"Metric {metric} is not supported yet!"
            raise ValueError(msg)

    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict:
        request = self._prepare_request(
            prediction_b, reference, input, baseline_prediction=prediction, **kwargs
        )
        response = self._client.evaluate_instances(request)
        return _parse_response(response, metric=self._metric)[0]

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict:
        request = self._prepare_request(
            prediction_b, reference, input, baseline_prediction=prediction, **kwargs
        )
        response = await self._async_client.evaluate_instances(request)
        return _parse_response(response, metric=self._metric)[0]
