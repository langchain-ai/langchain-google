"""
Langchain Runnables to screen user prompt and/or model response using Google
Cloud Model Armor.
"""

from typing import Any, Optional

from google.cloud.modelarmor_v1 import (
    DataItem,
    SanitizeModelResponseRequest,
    SanitizeUserPromptRequest,
)
from langchain_core.runnables.config import RunnableConfig

from langchain_google_community.model_armor.base_runnable import (
    ModelArmorSanitizeBaseRunnable,
)


class ModelArmorSanitizePromptRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize user prompts using Model Armor.
    """

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Sanitize a user prompt using Model Armor.

        Args:
            input (Any): The user prompt to sanitize.
            config (Optional[RunnableConfig]): A config to use when invoking
               the Runnable. The config supports standard keys like 'tags',
               'metadata' for tracing purposes, 'max_concurrency' for
               controlling how much work to do in parallel, and other keys.
               Please refer to the RunnableConfig for more details.

        Returns:
            Any: The original prompt if safe or fail_open,
                otherwise raises ValueError.
                If return_findings is True in case of safe or fail_open,
                returns a dict with the prompt and the sanitization findings.

        Raises:
            ValueError: If the prompt is flagged as unsafe by Model Armor and
                fail_open is False.
        """
        content = self._extract_input(input)
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
        if not self.evaluate(sanitization_findings, config=config):
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

        if self.return_findings:
            return {
                "prompt": input,
                "findings": sanitization_findings,
            }

        return input


class ModelArmorSanitizeResponseRunnable(ModelArmorSanitizeBaseRunnable):
    """
    Runnable to sanitize LLM responses using Model Armor.
    """

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Sanitize an LLM response using Model Armor.

        Args:
            input (Any): The LLM response to sanitize.
            config (Optional[RunnableConfig]): A config to use when invoking
               the Runnable. The config supports standard keys like 'tags',
               'metadata' for tracing purposes, 'max_concurrency' for
               controlling how much work to do in parallel, and other keys.
               Please refer to the RunnableConfig for more details.

        Returns:
            Any: The original response if safe or fail_open,
                otherwise raises ValueError.
                If return_findings is True in case of safe or fail_open,
                returns a dict with the response and the sanitization findings.

        Raises:
            ValueError: If the response is flagged as unsafe by Model Armor
                and fail_open is False.
        """
        content = self._extract_input(input)
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
        if not self.evaluate(sanitization_findings, config=config):
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

        if self.return_findings:
            return {
                "response": input,
                "findings": sanitization_findings,
            }

        return input
