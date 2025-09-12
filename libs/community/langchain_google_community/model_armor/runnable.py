"""
Langchain Runnables to screen user prompt and/or model response using Google
Cloud Model Armor.
"""

import logging
from typing import Any, Optional, TypeVar, cast

from google.cloud.modelarmor_v1 import (
    DataItem,
    ModelArmorClient,
    SanitizeModelResponseRequest,
    SanitizeUserPromptRequest,
)
from langchain_core.runnables.config import RunnableConfig

from langchain_google_community.model_armor.base_runnable import (
    ModelArmorSanitizeBaseRunnable,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ModelArmorSanitizePromptRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize user prompts using Model Armor.
    """

    def invoke(
        self,
        input: T,
        config: Optional[RunnableConfig] = None,
        fail_open: Optional[bool] = None,
        **kwargs: Any,
    ) -> T:
        """
        Sanitize a user prompt using Model Armor.

        Args:
            input: The user prompt to sanitize. Can be string, BaseMessage,
                BasePromptTemplate, list of messages, or any object with
                string conversion.
            config: A config to use when invoking the Runnable.
            fail_open: If True, allows unsafe prompts to pass through.

        Returns:
            Same type as input (original input is always returned).

        Raises:
            ValueError: If the prompt is flagged as unsafe by Model Armor and
                fail_open is False.
        """
        content = self._extract_input(input)

        logger.info(
            "Starting prompt sanitization request with template id %s",
            self.template_id,
        )

        result = cast(ModelArmorClient, self.client).sanitize_user_prompt(
            request=SanitizeUserPromptRequest(
                name=ModelArmorClient.template_path(
                    cast(str, self.project),
                    cast(str, self.location),
                    cast(str, self.template_id),
                ),
                user_prompt_data=DataItem(text=content),
            )
        )
        sanitization_findings = result.sanitization_result

        # Determine effective flags.
        effective_fail_open = fail_open if fail_open is not None else self.fail_open

        if not self.evaluate(content, sanitization_findings, config=config):
            if effective_fail_open:
                logger.info(
                    "Found following unsafe prompt findings from Model Armor: %s",
                    sanitization_findings,
                )
                logger.warning("Continuing for unsafe prompt as fail_open flag is true")
            else:
                raise ValueError("Prompt flagged as unsafe by Model Armor.")

        return input


class ModelArmorSanitizeResponseRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize LLM responses using Model Armor.
    """

    def invoke(
        self,
        input: T,
        config: Optional[RunnableConfig] = None,
        fail_open: Optional[bool] = None,
        **kwargs: Any,
    ) -> T:
        """
        Sanitize an LLM response using Model Armor.

        Args:
            input: The model response to sanitize. Can be string, BaseMessage,
                list of messages, or any object with string conversion.
            config: A config to use when invoking the Runnable.
            fail_open: If True, allows unsafe responses to pass through.

        Returns:
            Same type as input (original input is always returned).

        Raises:
            ValueError: If the response is flagged as unsafe by Model Armor
                and fail_open is False.
        """
        content = self._extract_input(input)

        logger.info(
            "Starting model response sanitization request with template id %s",
            self.template_id,
        )

        result = cast(ModelArmorClient, self.client).sanitize_model_response(
            request=SanitizeModelResponseRequest(
                name=ModelArmorClient.template_path(
                    cast(str, self.project),
                    cast(str, self.location),
                    cast(str, self.template_id),
                ),
                model_response_data=DataItem(text=content),
            )
        )

        sanitization_findings = result.sanitization_result
        if not self.evaluate(content, sanitization_findings, config=config):
            if self.fail_open:
                logger.info(
                    "Found following unsafe response findings from Model Armor: %s",
                    sanitization_findings,
                )
                logger.warning(
                    "Continuing for unsafe response as fail open flag is true"
                )
            else:
                raise ValueError("Response flagged as unsafe by Model Armor.")

        return input
