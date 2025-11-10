"""
This module provides a base class for creating runnables that
sanitize user prompts and model responses using Google Cloud Model Armor.
Ref: https://cloud.google.com/security-command-center/docs/model-armor-overview
"""

import logging
import os
from typing import Any, List, Optional, TypeVar, Union

import google.auth
from google.api_core.client_options import ClientOptions
from google.auth import credentials as google_auth_credentials
from google.cloud.modelarmor_v1 import (
    FilterMatchState,
    SanitizationResult,
)
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from langchain_google_community.model_armor import _client_utils

_DEFAULT_LOCATION = "us-central1"

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ModelArmorParams(BaseModel):
    """Model Armor parameters."""

    model_config = {"arbitrary_types_allowed": True}

    # Client Parameters
    project: Optional[str] = Field(
        default=None,
        description="The default GCP project to use when making Model Armor API calls.",
    )

    location: Optional[str] = Field(
        default=_DEFAULT_LOCATION,
        description="The default location to use when making API calls.",
    )

    credentials: Optional[google_auth_credentials.Credentials] = Field(
        default=None,
        description="The default custom credentials to use when making API calls.",
    )

    transport: Optional[str] = Field(
        default="grpc",
        alias="api_transport",
        description="The desired API transport method, can be either 'grpc' or 'rest'.",
    )

    client_options: Optional["ClientOptions"] = Field(
        default=None, exclude=True, description="Client options for the API client."
    )

    client_info: Optional[Any] = Field(
        default=None, exclude=True, description="Client info for the API client."
    )

    # Model Armor Parameters
    template_id: Optional[str] = Field(
        default=None, description="Model Armor template ID for sanitization."
    )

    fail_open: bool = Field(
        default=False,
        description="If True, allows unsafe prompts/responses to pass through, "
        "otherwise, raises an error.",
    )

    @model_validator(mode="after")
    def validate_project(self) -> "ModelArmorParams":
        """
        Validate the project ID.

        If not provided, will attempt to infer via credentials file, env or ADC.
        """
        if not self.project:
            # Get Project ID from credentials.
            if self.credentials and hasattr(self.credentials, "project_id"):
                self.project = self.credentials.project_id
            # Get Project ID from env variable.
            elif os.getenv("GOOGLE_CLOUD_PROJECT"):
                self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
            # Get Project ID using ADC.
            else:
                _, self.project = google.auth.default()

            if not self.project:
                raise ValueError(
                    "Unable to get GCP Project ID. Please set it explicitly while "
                    "creating runnables or use the supported auth methods."
                )
        return self


class ModelArmorSanitizeBaseRunnable(ModelArmorParams, RunnableSerializable):
    """
    Base runnable for user prompt or model response sanitization using
    Model Armor.

    Setup:
        You must either:
            - Have credentials configured for your environment (gcloud, workload
                identity , etc...)
            - Store the path to a service account JSON file as the
                `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

        For more information, see:

        - https://cloud.google.com/docs/authentication/application-default-credentials#GAC
        - https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth

    Attributes:
        client: A Model Armor client instance for making sanitization requests.
    """

    client: Any = Field(default=None, exclude=True)

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = _DEFAULT_LOCATION,
        credentials: Optional[google_auth_credentials.Credentials] = None,
        transport: Optional[str] = None,
        client_options: Optional[Any] = None,
        client_info: Optional[Any] = None,
        template_id: Optional[str] = None,
        fail_open: bool = True,
        **kwargs: Any,
    ) -> None:
        # Initialize the ModelArmorParams base class.
        super().__init__(
            project=project,
            location=location,
            credentials=credentials,
            api_transport=transport,
            client_options=client_options,
            client_info=client_info,
            template_id=template_id,
            fail_open=fail_open,
            **kwargs,
        )

        self.client = _client_utils._get_model_armor_client(
            location=self.location,
            credentials=self.credentials,
            transport=self.transport,
            client_options=self.client_options,
            client_info=self.client_info,
        )

        logger.debug(
            (
                "Initialized %s with model armor "
                "client, %s template id and fail open flag as %s"
            ),
            self.__class__.__name__,
            self.template_id,
            self.fail_open,
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_google_community", "model_armor"]

    def _extract_input(
        self,
        value: Union[str, BaseMessage, BasePromptTemplate, List[BaseMessage], object],
    ) -> str:
        """
        Extract text content compatible with Model Armor API sanitization requests.

        Model Armor API only accepts string content, so this method converts
        various LangChain input types to their string representation.

        Args:
            value: Input content that can be:
                - `str`: Direct string content
                - `BaseMessage`: LangChain message (`HumanMessage`, `AIMessage`, etc.)
                - `BasePromptTemplate`: LangChain prompt template
                - `list[BaseMessage]`: List of LangChain messages
                - Any other object with `__str__`, `to_string()`, or `format()` methods

        Returns:
            str: Extracted string content for Model Armor API sanitization requests

        Raises:
            TypeError: If the input type cannot be converted to string
        """
        # Handle basic string type.
        if isinstance(value, str):
            return value

        # Handle LangChain message types.
        if isinstance(value, BaseMessage):
            return getattr(value, "content", str(value))

        # Handle LangChain prompt templates.
        if isinstance(value, BasePromptTemplate):
            try:
                return str(value.format())
            except (AttributeError, KeyError, ValueError) as e:
                logging.debug(f"Failed to format BasePromptTemplate: {e}")
                return str(value)

        # Handle lists (typically List[BaseMessage]).
        if isinstance(value, list):
            return "\n".join(self._extract_input(item) for item in value)

        # Handle objects with to_string method.
        if hasattr(value, "to_string") and callable(getattr(value, "to_string")):
            try:
                return str(value.to_string())
            except Exception as e:
                logging.debug(f"Failed to call to_string() method: {e}")
                return str(value)

        # Handle objects with format method (excluding BasePromptTemplate).
        if hasattr(value, "format") and callable(getattr(value, "format")):
            try:
                return str(value.format())
            except Exception as e:
                logging.debug(f"Failed to call format() method: {e}")
                return str(value)

        # Final fallback: try to convert to string.
        try:
            return str(value)
        except Exception as e:
            raise TypeError(
                f"Unsupported input type: {type(value).__name__}. "
                f"Cannot convert to string. "
                f"Supported types are: str, BaseMessage, BasePromptTemplate, "
                f"List[BaseMessage], or objects with to_string() or format() methods."
            ) from e

    def evaluate(
        self,
        content: str,
        findings: SanitizationResult,
        config: Optional[RunnableConfig] = None,
    ) -> bool:
        """
        Evaluate findings from Model Armor.

        Args:
            content: User prompt or model response.
            findings: `SanitizationResult` object from Model Armor sanitization request.
            config: Config to use when invoking the `Runnable`.

                Please refer to `RunnableConfig` for more details.

        Returns:
            bool: `True` if all findings are safe, `False` if any are unsafe
                (`MATCH_FOUND`).
        """
        is_safe = True
        if not findings:
            logger.info("No findings found. Marking as safe.")
            return is_safe

        if findings.filter_match_state == FilterMatchState.MATCH_FOUND:
            is_safe = False
            if config:
                dispatch_custom_event(
                    "on_model_armor_finding",
                    {
                        "text_content": content,
                        "findings": findings,
                        "template_id": self.template_id,
                    },
                    config=config,
                )

        logger.info(
            "Evaluated content based on Model Armor sanitization response as %s",
            "Safe" if is_safe else "Unsafe",
        )
        return is_safe
