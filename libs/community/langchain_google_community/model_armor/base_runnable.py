"""
This module provides a base class for creating runnables that
sanitize user prompts and model responses using Google Cloud Model Armor.
"""

import logging
import sys
from typing import Any, Optional

from google.cloud.modelarmor_v1 import (
    FilterMatchState,
    ModelArmorClient,
    SanitizationResult,
)
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


def setup_default_logging() -> logging.Logger:
    """
    Set up a logger for the Model Armor runnables.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Configure logger for module
    local_logger = logging.getLogger(__name__)
    local_logger.setLevel(logging.INFO)

    # Clear existing handlers
    if local_logger.hasHandlers():
        local_logger.handlers.clear()

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    local_logger.addHandler(console_handler)

    return local_logger


class ModelArmorSanitizeBaseRunnable(Runnable):
    """
    Base runnable for user prompt or model response sanitization using
    Model Armor.

    Attributes:
        client: A Model Armor client instance for making sanitization requests.
        template_id: Model Armor template ID for sanitization.
        fail_open: If True, allows unsafe prompts/responses to pass through.
        return_findings: If True, returns dict with prompt/response with
                    the model armor sanitization findings.
        logger: Logger instance for logging events. if not provided,
            a default logger is created with INFO level.
    """

    def __init__(
        self,
        client: ModelArmorClient,
        template_id: str,
        fail_open: bool = True,
        return_findings: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.template_id = template_id
        self.fail_open = fail_open
        self.return_findings = return_findings

        self.logger = logger or setup_default_logging()

        self.logger.debug(
            (
                "Initialized %s with model armor "
                "client, %s template id and fail open flag as %s"
            ),
            self.__class__.__name__,
            self.template_id,
            self.fail_open,
        )

    def _extract_input(self, value: Any) -> str:
        """
        Extract text from various LangChain prompt/message types,
        including BaseMessage,BasePromptTemplate and string.

        Args:
            value (Any): Prompt/Response content to be extracted
                    from langchain formats

        Returns:
            str: Extracted string content of prompt/response
        """
        if isinstance(value, str):
            return value
        if isinstance(value, BaseMessage):
            return getattr(value, "content", str(value))
        if isinstance(value, BasePromptTemplate):
            # Render with empty dict if possible, else fallback to string
            try:
                return value.format()
            except Exception:
                return str(value)
        if isinstance(value, list):
            return "\n".join(self._extract_input(msg) for msg in value)
        if hasattr(value, "to_string") and callable(value.to_string):
            return value.to_string()
        if hasattr(value, "format") and callable(value.format):
            try:
                return value.format()
            except Exception:
                return str(value)
        return str(value)

    def evaluate(
        self,
        findings: SanitizationResult,
        config: Optional[RunnableConfig] = None,
    ) -> bool:
        """
        Evaluate findings from Model Armor.

        Args:
            findings (SanitizationResult): SanitizationResult object from
                Model Armor sanitization request.
            config (Optional[RunnableConfig]): A config to use when invoking
               the Runnable. The config supports standard keys like 'tags',
               'metadata' for tracing purposes, 'max_concurrency' for
               controlling how much work to do in parallel, and other keys.
               Please refer to the RunnableConfig for more details.

        Returns:
            bool: True if all findings are safe, False if any are unsafe (MATCH_FOUND).
        """
        is_safe = True
        if not findings:
            self.logger.warning("No findings found. Marking as safe.")
            return is_safe
        if findings.filter_match_state == FilterMatchState.MATCH_FOUND:
            is_safe = False
            if config:
                dispatch_custom_event(
                    "on_model_armor_finding",
                    {
                        "findings": findings,
                        "template_id": self.template_id,
                    },
                    config=config,
                )

        self.logger.info(
            "Evaluated content based on Model Armor sanitization response as %s",
            "Safe" if is_safe else "Unsafe",
        )
        return is_safe
