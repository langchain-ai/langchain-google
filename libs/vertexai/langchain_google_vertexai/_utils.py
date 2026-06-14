"""Utilities to init Vertex AI."""

import math
import os
import re
from collections.abc import Callable
from importlib import metadata
from typing import Any

import google.api_core
import proto  # type: ignore[import-untyped]
from google.api_core.gapic_v1.client_info import ClientInfo
from google.cloud import storage  # type: ignore[attr-defined, unused-ignore]
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from vertexai.generative_models import (
    Candidate,  # TODO: migrate to google-genai since this is deprecated
    Image,
)
from vertexai.language_models import (
    TextGenerationResponse,  # TODO: migrate to google-genai since this is deprecated
)

from langchain_google_vertexai._retry import create_base_retry_decorator

_TELEMETRY_TAG = "remote_reasoning_engine"
_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

# Cache package version at module import time to avoid blocking I/O in async contexts
try:
    _LANGCHAIN_VERTEXAI_VERSION = metadata.version("langchain-google-vertexai")
except metadata.PackageNotFoundError:
    _LANGCHAIN_VERTEXAI_VERSION = "0.0.0"


def create_retry_decorator(
    *,
    max_retries: int = 1,
    run_manager: AsyncCallbackManagerForLLMRun | CallbackManagerForLLMRun | None = None,
    wait_exponential_kwargs: dict[str, float] | None = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Vertex / Palm LLMs.

    Args:
        max_retries: Number of retries.
        run_manager: Callback manager for the run.
        wait_exponential_kwargs: Optional dictionary with parameters:

            - multiplier: Initial wait time multiplier (Default: `1.0`)
            - min: Minimum wait time in seconds (Default: `4.0`)
            - max: Maximum wait time in seconds (Default: `10.0`)
            - exp_base: Exponent base to use (Default: `2.0`)

    Returns:
        A retry decorator.
    """
    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    return create_base_retry_decorator(
        error_types=errors,
        max_retries=max_retries,
        run_manager=run_manager,
        wait_exponential_kwargs=wait_exponential_kwargs,
    )


def raise_vertex_import_error(minimum_expected_version: str = "1.44.0") -> None:
    """Raise `ImportError` related to Vertex SDK being not available.

    Args:
        minimum_expected_version: The lowest expected version of the SDK.

    Raises:
        ImportError: An `ImportError` that mentions a required version of the SDK.
    """
    msg = (
        "Please, install or upgrade the google-cloud-aiplatform library: "
        f"pip install google-cloud-aiplatform>={minimum_expected_version}"
    )
    raise ImportError(msg)


def get_user_agent(module: str | None = None) -> tuple[str, str]:
    r"""Returns a custom user agent header.

    Args:
        module: The module for a custom user agent header.
    """
    # Use cached version to avoid blocking I/O in async contexts
    client_library_version = (
        f"{_LANGCHAIN_VERTEXAI_VERSION}-{module}"
        if module
        else _LANGCHAIN_VERTEXAI_VERSION
    )
    if os.environ.get(_TELEMETRY_ENV_VARIABLE_NAME):
        client_library_version += f"+{_TELEMETRY_TAG}"
    return client_library_version, f"langchain-google-vertexai/{client_library_version}"


def get_client_info(module: str | None = None) -> "ClientInfo":
    r"""Returns a `ClientInfo` object with a custom user agent header.

    Args:
        module: The module for a custom user agent header.

    Returns:
        `google.api_core.gapic_v1.client_info.ClientInfo`
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


def load_image_from_gcs(path: str, project: str | None = None) -> Image:
    """Loads an `Image` from GCS."""
    gcs_client = storage.Client(project=project)
    pieces = path.split("/")
    blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
    if len(blobs) > 1:
        msg = f"Found more than one candidate for {path}!"
        raise ValueError(msg)
    return Image.from_bytes(blobs[0].download_as_bytes())


def _get_finish_reason_string(finish_reason: Any) -> str | None:
    """Convert finish_reason to string, handling both `enum` and raw `int` values.

    Args:
        finish_reason: The finish reason value from the candidate.

    Returns:
        String representation of the finish reason, or `None` if not present.
    """
    if finish_reason is None:
        return None
    if hasattr(finish_reason, "name"):
        return finish_reason.name
    if isinstance(finish_reason, int):
        return f"UNKNOWN_{finish_reason}"
    return None


def get_generation_info(
    candidate: TextGenerationResponse | Candidate,
    *,
    stream: bool = False,
    usage_metadata: dict | None = None,
    logprobs: bool | int = False,
) -> dict[str, Any]:
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini#response_body

    # Handle TextGenerationResponse vs Candidate differences
    # These types have different attributes, so we need type guards
    if isinstance(candidate, TextGenerationResponse):
        # TextGenerationResponse has limited attributes compared to Candidate
        info = {
            "is_blocked": False,  # TextGenerationResponse doesn't have safety_ratings
            "safety_ratings": [],
            "citation_metadata": None,
            "usage_metadata": usage_metadata,
            "finish_reason": None,  # Doesn't have finish_reason
            "finish_message": None,
        }
    else:
        # Handle Candidate type - has full set of attributes
        info = {
            "is_blocked": any(rating.blocked for rating in candidate.safety_ratings),
            "safety_ratings": [
                {
                    "category": rating.category.name,
                    "probability_label": rating.probability.name,
                    "probability_score": rating.probability_score,
                    "blocked": rating.blocked,
                    "severity": rating.severity.name,
                    "severity_score": rating.severity_score,
                }
                # Image generation models sometime return ratings that are not
                # included in the proto.
                for rating in candidate.safety_ratings
                if hasattr(rating.category, "name")
            ],
            "citation_metadata": (
                proto.Message.to_dict(candidate.citation_metadata)
                if candidate.citation_metadata
                else None
            ),
            "usage_metadata": usage_metadata,
            "finish_reason": _get_finish_reason_string(candidate.finish_reason),
            "finish_message": (
                candidate.finish_message if candidate.finish_message else None
            ),
        }

    # Check for avg_logprobs attribute - only available on Candidate
    if (
        not isinstance(candidate, TextGenerationResponse)
        and hasattr(candidate, "avg_logprobs")
        and candidate.avg_logprobs is not None
    ):
        if (
            isinstance(candidate.avg_logprobs, float)
            and not math.isnan(candidate.avg_logprobs)
            and candidate.avg_logprobs < 0
        ):
            info["avg_logprobs"] = candidate.avg_logprobs

    # Check for logprobs_result attribute - only available on Candidate
    if (
        not isinstance(candidate, TextGenerationResponse)
        and hasattr(candidate, "logprobs_result")
        and logprobs
    ):

        def is_valid_logprob(prob):
            # Logprobs can be 0.0 (probability=1.0, fully certain) or negative
            # (probability < 1.0). We should include all valid logprobs, not just
            # strictly negative ones.
            return isinstance(prob, (float, int)) and not math.isnan(prob) and prob <= 0

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

    # Check for grounding_metadata attribute - only available on Candidate
    if not isinstance(candidate, TextGenerationResponse):
        try:
            if candidate.grounding_metadata:
                info["grounding_metadata"] = proto.Message.to_dict(
                    candidate.grounding_metadata
                )
        except AttributeError:
            pass
    info = {k: v for k, v in info.items() if v is not None}
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-chat#response_body

    if stream:
        # Remove non-streamable types, like bools.
        info.pop("is_blocked")

    return info


def enforce_stop_tokens(text: str, stop: list[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def replace_defs_in_schema(original_schema: dict, defs: dict | None = None) -> dict:
    """Given an OpenAPI schema with a property `$defs` replaces all occurrences of
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
        elif "$ref" in value:
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


def _strip_nullable_anyof(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse `anyOf([{...}, {"type": "null"}])` into the non-null schema,
    leave the rest of the keywords alone, and make the property optional.

    Works in place.
    """

    def walk(node) -> None:
        if not isinstance(node, dict):
            return

        props = node.get("properties", {})
        for prop_name, prop_schema in list(props.items()):
            any_of = prop_schema.get("anyOf")
            if any_of and len(any_of) == 2:
                null_branch = next((b for b in any_of if b.get("type") == "null"), None)
                other_branch = next((b for b in any_of if b is not null_branch), None)

                if null_branch and other_branch:
                    # remove the anyOf *only*
                    prop_schema.pop("anyOf")
                    # and overlay the surviving branch
                    prop_schema.update(other_branch)

                    # make the property optional
                    req = node.get("required", [])
                    if prop_name in req:
                        req.remove(prop_name)
                        if not req:
                            node.pop("required")

            walk(prop_schema)

        if "items" in node:
            walk(node["items"])

        for combiner in ("allOf", "anyOf", "oneOf"):
            for sub in node.get(combiner, []):
                walk(sub)

    walk(schema)
    return schema
