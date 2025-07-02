from __future__ import annotations

import re
from concurrent.futures import Executor
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import vertexai  # type: ignore[import-untyped]
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.constants import base as constants
from google.cloud.aiplatform.gapic import (
    PredictionServiceAsyncClient,
    PredictionServiceClient,
)
from google.cloud.aiplatform.models import Prediction
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceAsyncClient as v1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1.services.prediction_service import (
    PredictionServiceClient as v1PredictionServiceClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceAsyncClient as v1beta1PredictionServiceAsyncClient,
)
from google.cloud.aiplatform_v1beta1.services.prediction_service import (
    PredictionServiceClient as v1beta1PredictionServiceClient,
)
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from langchain_core.outputs import Generation, LLMResult
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Literal, Self
from vertexai.generative_models._generative_models import (  # type: ignore
    SafetySettingsType,
)

from langchain_google_vertexai._client_utils import (
    _get_async_prediction_client,
    _get_client_options,
    _get_prediction_client,
)
from langchain_google_vertexai._utils import (
    get_client_info,
    get_user_agent,
)

_DEFAULT_LOCATION = "us-central1"


class _VertexAIBase(BaseModel):
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    project: Optional[str] = None
    "The default GCP project to use when making Vertex API calls."
    location: str = Field(default=_DEFAULT_LOCATION)
    "The default location to use when making API calls."
    request_parallelism: int = 5
    "The amount of parallelism allowed for requests issued to VertexAI models. "
    "Default is 5."
    max_retries: int = 6
    """The maximum number of retries to make when generating."""
    task_executor: ClassVar[Optional[Executor]] = Field(default=None, exclude=True)
    stop: Optional[List[str]] = Field(default=None, alias="stop_sequences")
    "Optional list of stop words to use when generating."
    model_name: Optional[str] = Field(default=None, alias="model")
    "Underlying model name."
    full_model_name: Optional[str] = Field(
        default=None, exclude=True
    )  #: :meta private:
    "The full name of the model's endpoint."
    client_options: Optional["ClientOptions"] = Field(
        default=None, exclude=True
    )  #: :meta private:
    api_endpoint: Optional[str] = Field(default=None, alias="base_url")
    "Desired API endpoint, e.g., us-central1-aiplatform.googleapis.com"
    api_transport: Optional[str] = None
    """The desired API transport method, can be either 'grpc' or 'rest'. 
    Uses the default parameter in vertexai.init if defined.
    """
    default_metadata: Sequence[Tuple[str, str]] = Field(
        default_factory=list
    )  #: :meta private:
    additional_headers: Optional[Dict[str, str]] = Field(default=None)
    "A key-value dictionary representing additional headers for the model call"
    client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None
    "A callback which returns client certificate bytes and private key bytes both "
    "in PEM format."
    credentials: Any = Field(default=None, exclude=True)
    "The default custom credentials (google.auth.credentials.Credentials) to use "
    "when making API calls. If not provided, credentials will be ascertained from "
    "the environment."
    endpoint_version: Literal["v1", "v1beta1"] = "v1beta1"
    """Whether to use v1 or v1beta1 endpoint.
    
    v1 is more performant, but v1beta1 might have some new features.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_params_base(cls, values: dict) -> Any:
        if "model" in values and "model_name" not in values:
            values["model_name"] = values.pop("model")
        if "api_transport" not in values:
            values["api_transport"] = initializer.global_config._api_transport
        if "location" not in values:
            values["location"] = initializer.global_config.location
        if values.get("api_endpoint") or values.get("base_url"):
            api_endpoint = values.get("api_endpoint", values.get("base_url"))
        else:
            location = values.get("location", cls.model_fields["location"].default)
            api_endpoint = (
                f"{'' if location == 'global' else location + '-'}"
                f"{constants.PREDICTION_API_BASE_PATH}"
            )
        values["client_options"] = _get_client_options(
            api_endpoint=api_endpoint,
            cert_source=values.get("client_cert_source"),
        )
        additional_headers = values.get("additional_headers") or {}
        values["default_metadata"] = tuple(additional_headers.items())
        return values

    @model_validator(mode="after")
    def validate_project(self) -> Any:
        if self.project is None:
            if self.credentials and hasattr(self.credentials, "project_id"):
                self.project = self.credentials.project_id
            else:
                self.project = initializer.global_config.project
        return self

    @property
    def prediction_client(
        self,
    ) -> Union[v1beta1PredictionServiceClient, v1PredictionServiceClient]:
        """Returns PredictionServiceClient."""
        if self.client is None:
            self.client = _get_prediction_client(
                endpoint_version=self.endpoint_version,
                credentials=self.credentials,
                client_options=self.client_options,
                transport=self.api_transport,
                user_agent=self._user_agent,
            )
        return self.client

    @property
    def async_prediction_client(
        self,
    ) -> Union[v1PredictionServiceAsyncClient, v1beta1PredictionServiceAsyncClient]:
        """Returns PredictionServiceClient."""
        if self.async_client is None:
            self.async_client = _get_async_prediction_client(
                endpoint_version=self.endpoint_version,
                credentials=self.credentials,
                client_options=cast(ClientOptions, self.client_options),
                user_agent=self._user_agent,
            )
        return self.async_client

    @property
    def _user_agent(self) -> str:
        """Gets the User Agent."""
        _, user_agent = get_user_agent(f"{type(self).__name__}_{self.model_name}")
        return user_agent

    @property
    def _library_version(self) -> str:
        """Gets the library version for headers."""
        library_version, _ = get_user_agent(f"{type(self).__name__}_{self.model_name}")
        return library_version


class _VertexAICommon(_VertexAIBase):
    client_preview: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: Optional[str] = Field(default=None, alias="model")
    "Underlying model name."
    temperature: Optional[float] = None
    "Sampling temperature, it controls the degree of randomness in token selection."
    frequency_penalty: Optional[float] = None
    "Positive values penalize tokens that repeatedly appear in the generated text, "
    "decreasing the probability of repeating content."
    presence_penalty: Optional[float] = None
    "Positive values penalize tokens that already appear in the generated text, "
    "increasing the probability of generating more diverse content."
    max_output_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    "Token limit determines the maximum amount of text output from one prompt."
    top_p: Optional[float] = None
    "Tokens are selected from most probable to least until the sum of their "
    "probabilities equals the top-p value. Top-p is ignored for Codey models."
    top_k: Optional[int] = None
    "How the model selects tokens for output, the next token is selected from "
    "among the top-k most probable tokens. Top-k is ignored for Codey models."
    n: int = 1
    """How many completions to generate for each prompt."""
    seed: Optional[int] = None
    """Random seed for the generation."""
    streaming: bool = False
    """Whether to stream the results or not."""
    safety_settings: Optional["SafetySettingsType"] = None
    """The default safety settings to use for all generations. 
    
        For example: 

            from langchain_google_vertexai import HarmBlockThreshold, HarmCategory

            safety_settings = {
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            """  # noqa: E501

    tuned_model_name: Optional[str] = None
    """The name of a tuned model."""
    thinking_budget: Optional[int] = Field(
        default=None, description="Indicates the thinking budget in tokens."
    )
    include_thoughts: Optional[bool] = Field(
        default=None,
        description="Indicates whether to include thoughts in the response.",
    )
    audio_timestamp: Optional[bool] = Field(
        default=None,
        description="Enable timestamp understanding of audio-only files",
    )

    @property
    def _llm_type(self) -> str:
        return "vertexai"

    @property
    def max_tokens(self) -> int | None:
        return self.max_output_tokens

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Gets the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    @property
    def _default_params(self) -> Dict[str, Any]:
        default_params: Dict[str, Any] = {}
        params = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "candidate_count": self.n,
            "seed": self.seed,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        updated_params = {}
        for param_name, param_value in params.items():
            default_value = default_params.get(param_name)
            if param_value is not None or default_value is not None:
                updated_params[param_name] = (
                    param_value if param_value is not None else default_value
                )
        return updated_params

    @classmethod
    def _init_vertexai(cls, values: Dict) -> None:
        vertexai.init(
            project=values.get("project"),
            location=values.get("location"),
            credentials=values.get("credentials"),
            # if both project and api_transport are empty, vertexai.init() sets
            # the default transport to "rest"
            api_transport=values.get("api_transport", "grpc"),
            api_endpoint=values.get("api_endpoint"),
            request_metadata=values.get("default_metadata"),
        )
        return None

    def _prepare_params(
        self,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        stop_sequences = stop or self.stop
        params_mapping = {"n": "candidate_count"}
        params = {params_mapping.get(k, k): v for k, v in kwargs.items()}
        params = {**self._default_params, "stop_sequences": stop_sequences, **params}
        if stream or self.streaming:
            params.pop("candidate_count")
        return params


class _BaseVertexAIModelGarden(_VertexAIBase):
    """Large language models served from Vertex AI Model Garden."""

    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    endpoint_id: str
    "A name of an endpoint where the model has been deployed."
    allowed_model_args: Optional[List[str]] = None
    "Allowed optional args to be passed to the model."
    prompt_arg: str = "prompt"
    result_arg: Optional[str] = "generated_text"
    "Set result_arg to None if output of the model is expected to be a string."
    "Otherwise, if it's a dict, provided an argument that contains the result."
    single_example_per_request: bool = True
    "LLM endpoint currently serves only the first example in the request"

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that the python package exists in environment."""

        if not self.project:
            raise ValueError(
                "A GCP project should be provided to run inference on Model Garden!"
            )

        client_options = ClientOptions(
            api_endpoint=f"{self.location}-aiplatform.googleapis.com"
        )
        client_info = get_client_info(module="vertex-ai-model-garden")
        self.client = PredictionServiceClient(
            client_options=client_options, client_info=client_info
        )
        self.async_client = PredictionServiceAsyncClient(
            client_options=client_options, client_info=client_info
        )
        return self

    @property
    def endpoint_path(self) -> str:
        return self.client.endpoint_path(
            project=self.project, location=self.location, endpoint=self.endpoint_id
        )

    @property
    def _llm_type(self) -> str:
        return "vertexai_model_garden"

    def _prepare_request(self, prompts: List[str], **kwargs: Any) -> List["Value"]:
        instances = []
        for prompt in prompts:
            if self.allowed_model_args:
                instance = {
                    k: v for k, v in kwargs.items() if k in self.allowed_model_args
                }
            else:
                instance = {}
            instance[self.prompt_arg] = prompt
            instances.append(instance)

        predict_instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        return predict_instances

    def _parse_response(self, predictions: "Prediction") -> LLMResult:
        generations: List[List[Generation]] = []
        for result in predictions.predictions:
            if isinstance(result, str):
                generations.append([Generation(text=self._parse_prediction(result))])
            else:
                generations.append(
                    [
                        Generation(text=self._parse_prediction(prediction))
                        for prediction in result
                    ]
                )
        return LLMResult(generations=generations)

    def _parse_prediction(self, prediction: Any) -> str:
        def _clean_response(response: str) -> str:
            if response.startswith("Prompt:\n"):
                result = re.search(r"(?s:.*)\nOutput:\n((?s:.*))", response)
                if result:
                    return result[1]
            return response

        if isinstance(prediction, str):
            return _clean_response(prediction)

        if self.result_arg:
            try:
                return _clean_response(prediction[self.result_arg])
            except KeyError:
                if isinstance(prediction, str):
                    error_desc = (
                        "Provided non-None `result_arg` (result_arg="
                        f"{self.result_arg}). But got prediction of type "
                        f"{type(prediction)} instead of dict. Most probably, you"
                        "need to set `result_arg=None` during VertexAIModelGarden "
                        "initialization."
                    )
                    raise ValueError(error_desc)
                else:
                    raise ValueError(f"{self.result_arg} key not found in prediction!")

        return prediction
