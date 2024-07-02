"""Utilities to init Vertex AI."""

import dataclasses
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
            "gemini-1.5-flash-preview-0514",
            "gemini-1.5-pro-preview-0514",
            "gemini-1.5-pro-preview-0409",
            "gemini-1.5-flash-001",
            "gemini-1.5-pro-001",
            "medlm-large-1.5-preview",
            "medlm-large-1.5-001",
        }
        if model_name in gemini_advanced_models:
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
) -> Dict[str, Any]:
    if is_gemini:
        # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini#response_body
        info = {
            "is_blocked": any([rating.blocked for rating in candidate.safety_ratings]),
            "safety_ratings": [
                {
                    "category": rating.category.name,
                    "probability_label": rating.probability.name,
                    "blocked": rating.blocked,
                    "severity": rating.severity.name,
                }
                for rating in candidate.safety_ratings
            ],
            "citation_metadata": (
                proto.Message.to_dict(candidate.citation_metadata)
                if candidate.citation_metadata
                else None
            ),
            "usage_metadata": usage_metadata,
        }
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
