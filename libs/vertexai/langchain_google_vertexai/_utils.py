"""Utilities to init Vertex AI."""

import dataclasses
import math
import re
from enum import Enum, auto
from importlib import metadata
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import google.api_core
import proto  # type: ignore[import-untyped]
from google.api_core.gapic_v1.client_info import ClientInfo
from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from vertexai.generative_models import (  # type: ignore[import-untyped]
    Candidate,
    Image,
)
from vertexai.language_models import (  # type: ignore[import-untyped]
    TextGenerationResponse,
)


def create_retry_decorator(
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Vertex / Palm LLMs."""

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def raise_vertex_import_error(minimum_expected_version: str = "1.44.0") -> None:
    """Raise ImportError related to Vertex SDK being not available.

    Args:
        minimum_expected_version: The lowest expected version of the SDK.
    Raises:
        ImportError: an ImportError that mentions a required version of the SDK.
    """
    raise ImportError(
        "Please, install or upgrade the google-cloud-aiplatform library: "
        f"pip install google-cloud-aiplatform>={minimum_expected_version}"
    )


def get_user_agent(module: Optional[str] = None) -> Tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        Tuple[str, str]
    """
    try:
        langchain_version = metadata.version("langchain-google-vertexai")
    except metadata.PackageNotFoundError:
        langchain_version = "0.0.0"
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    return client_library_version, f"langchain-google-vertexai/{client_library_version}"


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""Returns a client info object with a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    client_library_version, user_agent = get_user_agent(module)
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=user_agent,
    )


def _format_model_name(model: str, project: str, location: str) -> str:
    if "/" not in model:
        model = "publishers/google/models/" + model
    if model.startswith("models/"):
        model = "publishers/google/" + model
    if model.startswith("publishers/"):
        return f"projects/{project}/locations/{location}/{model}"
    return model


def load_image_from_gcs(path: str, project: Optional[str] = None) -> Image:
    """Loads an Image from GCS."""
    gcs_client = storage.Client(project=project)
    pieces = path.split("/")
    blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
    if len(blobs) > 1:
        raise ValueError(f"Found more than one candidate for {path}!")
    return Image.from_bytes(blobs[0].download_as_bytes())


class GoogleModelFamily(str, Enum):
    GEMINI = auto()
    GEMINI_ADVANCED = auto()
    CODEY = auto()
    PALM = auto()

    @classmethod
    def _missing_(cls, value: Any) -> "GoogleModelFamily":
        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning
        model_name = value.lower()
        gemini_advanced_models = {
            "medlm-large-1.5-preview",
            "medlm-large-1.5-001",
            "medlm-large-1.5@001",
        }
        if (
            "gemini-1.5" in model_name
            or model_name in gemini_advanced_models
            or "gemini-2" in model_name
        ):
            return GoogleModelFamily.GEMINI_ADVANCED
        if "gemini" in model_name:
            return GoogleModelFamily.GEMINI
        if "code" in model_name:
            return GoogleModelFamily.CODEY
        if "medlm-medium@latest" in model_name:
            return GoogleModelFamily.GEMINI
        if "bison" in model_name or "medlm" in model_name:
            return GoogleModelFamily.PALM
        return GoogleModelFamily.GEMINI


def is_gemini_model(model_family: GoogleModelFamily) -> bool:
    """Returns True if the model name is a Gemini model."""
    return model_family in [GoogleModelFamily.GEMINI, GoogleModelFamily.GEMINI_ADVANCED]


def get_generation_info(
    candidate: Union[TextGenerationResponse, Candidate],
    is_gemini: bool,
    *,
    stream: bool = False,
    usage_metadata: Optional[Dict] = None,
    logprobs: Union[bool, int] = False,
) -> Dict[str, Any]:
    if is_gemini:
        # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini#response_body
        info = {
            "is_blocked": any([rating.blocked for rating in candidate.safety_ratings]),
            "safety_ratings": [
                {
                    "category": rating.category.name,
                    "probability_label": rating.probability.name,
                    "probability_score": rating.probability_score,
                    "blocked": rating.blocked,
                    "severity": rating.severity.name,
                    "severity_score": rating.severity_score,
                }
                for rating in candidate.safety_ratings
            ],
            "citation_metadata": (
                proto.Message.to_dict(candidate.citation_metadata)
                if candidate.citation_metadata
                else None
            ),
            "usage_metadata": usage_metadata,
            "finish_reason": (
                candidate.finish_reason.name if candidate.finish_reason else None
            ),
            "finish_message": (
                candidate.finish_message if candidate.finish_message else None
            ),
        }
        if hasattr(candidate, "avg_logprobs") and candidate.avg_logprobs is not None:
            if (
                isinstance(candidate.avg_logprobs, float)
                and not math.isnan(candidate.avg_logprobs)
                and candidate.avg_logprobs < 0
            ):
                info["avg_logprobs"] = candidate.avg_logprobs

        if hasattr(candidate, "logprobs_result") and logprobs:

            def is_valid_logprob(prob):
                return isinstance(prob, float) and not math.isnan(prob) and prob < 0

            chosen_candidates = candidate.logprobs_result.chosen_candidates
            top_candidates_list = candidate.logprobs_result.top_candidates
            logprobs_int = 0 if logprobs is True else logprobs

            valid_log_probs = []
            for i, chosen in enumerate(chosen_candidates):
                if not is_valid_logprob(chosen.log_probability):
                    continue

                top_logprobs = []
                if logprobs_int > 0:
                    for top in top_candidates_list[i].candidates[:logprobs_int]:
                        if not is_valid_logprob(top.log_probability):
                            continue
                        top_logprobs.append(
                            {"token": top.token, "logprob": top.log_probability}
                        )

                valid_log_probs.append(
                    {
                        "token": chosen.token,
                        "logprob": chosen.log_probability,
                        "top_logprobs": top_logprobs,
                    }
                )

            if valid_log_probs:
                info["logprobs_result"] = valid_log_probs

        try:
            if candidate.grounding_metadata:
                info["grounding_metadata"] = proto.Message.to_dict(
                    candidate.grounding_metadata
                )
        except AttributeError:
            pass
        info = {k: v for k, v in info.items() if v is not None}
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat#response_body
    else:
        info = dataclasses.asdict(candidate)
        info.pop("text")
        info = {k: v for k, v in info.items() if not k.startswith("_")}
        if usage_metadata:
            info_usage_metadata = {}
            output_usage = usage_metadata.get("tokenMetadata", {}).get(
                "outputTokenCount", {}
            )
            info_usage_metadata["candidates_billable_characters"] = output_usage.get(
                "totalBillableCharacters"
            )
            info_usage_metadata["candidates_token_count"] = output_usage.get(
                "totalTokens"
            )
            input_usage = usage_metadata.get("tokenMetadata", {}).get(
                "inputTokenCount", {}
            )
            info_usage_metadata["prompt_billable_characters"] = input_usage.get(
                "totalBillableCharacters"
            )
            info_usage_metadata["prompt_token_count"] = input_usage.get("totalTokens")
            info["usage_metadata"] = {k: v for k, v in info_usage_metadata.items() if v}

            # NOTE:
            # "safety_attributes" can contain different values for the same keys
            # for each generation. Put it in a list, so it can be merged later by
            # merge_dicts().
            #
            safety_attributes = info.get("safety_attributes") or {}
            info["safety_attributes"] = [safety_attributes]

    if stream:
        # Remove non-streamable types, like bools.
        info.pop("is_blocked")

    return info


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def replace_defs_in_schema(original_schema: dict, defs: Optional[dict] = None) -> dict:
    """Given an OpenAPI schema with a property '$defs' replaces all occurrences of
    referenced items in the dictionary.

    Args:
        original_schema: Schema generated by `BaseModel.model_schema_json`
        defs: Definitions for recursive calls.

    Returns:
        Schema with refs replaced.
    """

    new_defs = defs or original_schema.get("$defs")

    if new_defs is None or not isinstance(new_defs, dict):
        return original_schema.copy()

    resulting_schema = {}

    for key, value in original_schema.items():
        if key == "$defs":
            continue

        if not isinstance(value, dict):
            resulting_schema[key] = value
        else:
            if "$ref" in value:
                new_value = value.copy()

                path = new_value.pop("$ref")
                def_key = _get_def_key_from_schema_path(path)
                new_item = new_defs.get(def_key)

                assert isinstance(new_item, dict)
                new_value.update(new_item)

                resulting_schema[key] = replace_defs_in_schema(new_value, defs=new_defs)
            else:
                resulting_schema[key] = replace_defs_in_schema(value, defs=new_defs)

    return resulting_schema


def _get_def_key_from_schema_path(schema_path: str) -> str:
    error_message = f"Malformed schema reference path {schema_path}"

    if not isinstance(schema_path, str) or not schema_path.startswith("#/$defs/"):
        raise ValueError(error_message)

    # Schema has to have only one extra level.
    parts = schema_path.split("/")
    if len(parts) != 3:
        raise ValueError(error_message)

    return parts[-1]
