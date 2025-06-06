"""
Model Armor Runnables for LangChain
for prompt/response sanitization using Google Cloud Model Armor.
"""

import sys
import json
import logging
from typing import Any, List, Optional, Union

from google.cloud.modelarmor_v1 import (
    DataItem,
    FilterMatchState,
    ModelArmorClient,
    SanitizationResult,
    SanitizeModelResponseRequest,
    SanitizeUserPromptRequest,
)
from langchain_core.callbacks.manager import dispatch_custom_event
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable


class ModelArmorSanitizeBaseRunnable(Runnable):
    """
    Base runnable for Model Armor prompt/response sanitization.
    """

    def __init__(
        self,
        client: ModelArmorClient,
        template_id: str,
        fail_open: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.client = client
        self.template_id = template_id
        self.fail_open = fail_open

        # Configure logger
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

        self.logger = logger or local_logger

        self.logger.debug(
            (
                "Initialized %s with model armor "
                "client, %s template id and fail open flag as %s"
            ),
            self.__class__.__name__,
            self.template_id,
            self.fail_open,
        )

    def _extract_input(
        self,
        value: Union[
            str,
            BaseMessage,
            BasePromptTemplate,
            List[BasePromptTemplate],
            List[BasePromptTemplate],
        ],
    ) -> str:
        """
        Extract text from various LangChain prompt/message types,
        including BaseMessage,BasePromptTemplate and string.

        Args:
            value (Any): The input value to extract text from. Can be a string, BaseMessage, BasePromptTemplate, or a list of these types.

        Returns:
            str: The extracted text representation.
        """
        if isinstance(value, str):
            return value
        if isinstance(value, BaseMessage):
            return getattr(value, "content", str(value))
        if isinstance(value, BasePromptTemplate):
            # Render with empty dict if possible, else fallback to string
            try:
                return value.format({})
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

    def evaluate(self, findings: SanitizationResult) -> bool:
        """
        Evaluate findings from Model Armor.

        Args:
            findings (SanitizationResult): SanitizationResult object from
                Model Armor sanitization request.

        Returns:
            bool: True if all findings are safe, False if any are unsafe (MATCH_FOUND).
        """
        is_safe = True
        if not findings:
            self.logger.warning("No findings found. Marking as safe.")
            return is_safe
        if findings.filter_match_state == FilterMatchState.MATCH_FOUND:
            is_safe = False
            dispatch_custom_event(
                "on_model_armor_finding",
                {
                    "findings": findings,
                    "template_id": self.template_id,
                },
            )

        self.logger.info(
            (
                "Evaluated content, based on Model Armor "
                "sanitization response as %s"
            ),
            "Safe" if is_safe else "Unsafe",
        )
        return is_safe


class ModelArmorSanitizePromptRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize user prompts using Model Armor.
    """

    def invoke(
        self,
        prompt: Union[
            str,
            BaseMessage,
            BasePromptTemplate,
            List[BasePromptTemplate],
            List[BasePromptTemplate],
        ],
        config: Optional[dict] = None,
    ) -> Any:
        """
        Sanitize a user prompt using Model Armor.

        Args:
            prompt (Any): The user prompt to sanitize.
            config (Optional[Any]): Optional config for invocation (unused).

        Returns:
            Any: The original prompt if safe or fail_open,
                otherwise raises ValueError.

        Raises:
            ValueError: If the prompt is flagged as unsafe by Model Armor and
                fail_open is False.
        """
        content = self._extract_input(prompt)
        self.logger.info(
            "Starting prompt sanitization request with template id %s",
            self.template_id,
        )
        result = self.client.sanitize_user_prompt(
            request=SanitizeUserPromptRequest(
                name=self.template_id,
                user_prompt_data=DataItem(text=content),
            )
        )
        sanitization_findings = result.sanitization_result
        if not self.evaluate(sanitization_findings):
            self.logger.info(
                "Found following unsafe prompt findings from Model Armor: %s",
                sanitization_findings,
            )
            if self.fail_open:
                self.logger.warning(
                    "Continuing for unsafe prompt as fail open flag is true"
                )
            else:
                raise ValueError("Prompt flagged as unsafe by Model Armor.")
        return prompt


class ModelArmorSanitizeResponseRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize LLM responses using Model Armor.
    """

    def invoke(self, response: Any, config: Optional[Any] = None) -> Any:
        """
        Sanitize an LLM response using Model Armor.

        Args:
            response (Any): The LLM response to sanitize.
            config (Optional[Any]): Optional config for invocation (unused).

        Returns:
            Any: The original response if safe or fail_open,
                otherwise raises ValueError.

        Raises:
            ValueError: If the response is flagged as unsafe by Model Armor
                and fail_open is False.
        """
        content = self._extract_input(response)
        self.logger.info(
            "Starting response sanitization request with template id %s",
            self.template_id,
        )
        result = self.client.sanitize_model_response(
            request=SanitizeModelResponseRequest(
                name=self.template_id,
                model_response_data=DataItem(text=content),
            )
        )

        sanitization_findings = result.sanitization_result
        if not self.evaluate(sanitization_findings):
            self.logger.info(
                "Found following unsafe response findings from Model Armor: %s",
                sanitization_findings,
            )
            if self.fail_open:
                self.logger.warning(
                    "Continuing for unsafe response as fail open flag is true"
                )
            else:
                raise ValueError("Response flagged as unsafe by Model Armor.")
        return response
