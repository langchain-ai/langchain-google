"""Interfaces to be implemented by general evaluators.

Remove after interfaces will be moved to lc-core.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from warnings import warn

from langchain_core.runnables.config import run_in_executor

logger = logging.getLogger(__name__)


class _EvalArgsMixin:
    """Mixin for checking evaluation arguments."""

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @property
    def requires_input(self) -> bool:
        """Whether this evaluator requires an input string."""
        return False

    @property
    def _skip_input_warning(self) -> str:
        """Warning to show when input is ignored."""
        return f"Ignoring input in {self.__class__.__name__}, as it is not expected."

    @property
    def _skip_reference_warning(self) -> str:
        """Warning to show when reference is ignored."""
        return (
            f"Ignoring reference in {self.__class__.__name__}, as it is not expected."
        )

    def _check_evaluation_args(
        self,
        reference: Optional[str] = None,
        input: Optional[str] = None,
    ) -> None:
        """Check if the evaluation arguments are valid.

        Args:
            reference (Optional[str], optional): The reference label.
            input (Optional[str], optional): The input string.
        Raises:
            ValueError: If the evaluator requires an input string but none is provided,
                or if the evaluator requires a reference label but none is provided.
        """
        if self.requires_input and input is None:
            raise ValueError(f"{self.__class__.__name__} requires an input string.")
        elif input is not None and not self.requires_input:
            warn(self._skip_input_warning)
        if self.requires_reference and reference is None:
            raise ValueError(f"{self.__class__.__name__} requires a reference string.")
        elif reference is not None and not self.requires_reference:
            warn(self._skip_reference_warning)


class StringEvaluator(_EvalArgsMixin, ABC):
    """Grade, tag, or otherwise evaluate predictions relative to their inputs
    and/or reference labels."""

    @property
    def evaluation_name(self) -> str:
        """The name of the evaluation."""
        return self.__class__.__name__

    @property
    def requires_reference(self) -> bool:
        """Whether this evaluator requires a reference label."""
        return False

    @abstractmethod
    def _evaluate_strings(
        self,
        *,
        prediction: Union[str, Any],
        reference: Optional[Union[str, Any]] = None,
        input: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
                It is recommended that the dictionary contain the following keys:
                     - score: the score of the evaluation, if applicable.
                     - value: the string value of the evaluation, if applicable.
                     - reasoning: the reasoning for the evaluation, if applicable.
        """  # noqa: E501

    async def _aevaluate_strings(
        self,
        *,
        prediction: Union[str, Any],
        reference: Optional[Union[str, Any]] = None,
        input: Optional[Union[str, Any]] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
                It is recommended that the dictionary contain the following keys:
                     - score: the score of the evaluation, if applicable.
                     - value: the string value of the evaluation, if applicable.
                     - reasoning: the reasoning for the evaluation, if applicable.
        """  # noqa: E501
        return await run_in_executor(
            None,
            self._evaluate_strings,
            prediction=prediction,
            reference=reference,
            input=input,
            **kwargs,
        )

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): The LLM or chain prediction to evaluate.
            reference (Optional[str], optional): The reference label to evaluate against.
            input (Optional[str], optional): The input to consider during evaluation.
            **kwargs: Additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_strings(
            prediction=prediction, reference=reference, input=input, **kwargs
        )


class PairwiseStringEvaluator(_EvalArgsMixin, ABC):
    """Compare the output of two models (or two outputs of the same model)."""

    @abstractmethod
    def _evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (Optional[str], optional): The expected output / reference string.
            input (Optional[str], optional): The input string.
            **kwargs: Additional keyword arguments, such as callbacks and optional reference strings.
        Returns:
            dict: A dictionary containing the preference, scores, and/or other information.
        """  # noqa: E501

    async def _aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (Optional[str], optional): The expected output / reference string.
            input (Optional[str], optional): The input string.
            **kwargs: Additional keyword arguments, such as callbacks and optional reference strings.
        Returns:
            dict: A dictionary containing the preference, scores, and/or other information.
        """  # noqa: E501
        return await run_in_executor(
            None,
            self._evaluate_string_pairs,
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )

    def evaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (Optional[str], optional): The expected output / reference string.
            input (Optional[str], optional): The input string.
            **kwargs: Additional keyword arguments, such as callbacks and optional reference strings.
        Returns:
            dict: A dictionary containing the preference, scores, and/or other information.
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input=input)
        return self._evaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )

    async def aevaluate_string_pairs(
        self,
        *,
        prediction: str,
        prediction_b: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Asynchronously evaluate the output string pairs.

        Args:
            prediction (str): The output string from the first model.
            prediction_b (str): The output string from the second model.
            reference (Optional[str], optional): The expected output / reference string.
            input (Optional[str], optional): The input string.
            **kwargs: Additional keyword arguments, such as callbacks and optional reference strings.
        Returns:
            dict: A dictionary containing the preference, scores, and/or other information.
        """  # noqa: E501
        self._check_evaluation_args(reference=reference, input=input)
        return await self._aevaluate_string_pairs(
            prediction=prediction,
            prediction_b=prediction_b,
            reference=reference,
            input=input,
            **kwargs,
        )
