"""Decision model for the Gemini Decisions API."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

from google import genai
from google.genai import types
from langchain_core._api import beta
from langchain_core.decisions import BaseDecisionModel
from langchain_core.exceptions import ModelRefusalError
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import override

from langchain_google_genai.classifier._input import (
    to_decision_input,
    to_decision_questions,
)
from langchain_google_genai.classifier.types import ClassifierResponse

if TYPE_CHECKING:
    from langchain_google_genai.classifier.types import Question, State


@beta()
class GoogleGenAIClassifier(BaseDecisionModel):
    """Ask typed questions about text, JSON, or media with the Gemini Decisions API.

    A single request can combine `Noul` probabilities, `Choice` selections, and
    `Score` ratings over shared state. The response preserves probabilities,
    confidence, and token usage so application code can decide whether to act,
    route, or request human review.

    `ChoiceAnswer.confidence` and `ScoreAnswer.confidence` are the probability of the
    selected label or level. The API's `NoulAnswer` confidence, `max(noul, 1 - noul)`,
    is not kept because it can be computed from `noul`.

    Requests always use the Gemini Developer API.

    Args:
        model: Decisions model used to answer questions.
        api_key: Google API key. If omitted, reads `GOOGLE_API_KEY`, then
            `GEMINI_API_KEY`.
        base_url: Base URL for API requests. If omitted, the `google-genai` SDK
            default is used.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of request attempts. Uses the `google-genai` SDK
            default when `None`.
        client: Optional `google.genai.Client` used for requests. Other connection
            settings are ignored when it is supplied.

    Raises:
        ValueError: If no API key is available and no `client` is supplied. Also
            raised by `invoke` for a `Box` question.
        ModelRefusalError: From `invoke` when the API blocked the input and
            returned no answers.

    ??? example "Ask several questions about the same state"

        ```python
        from langchain_google_genai.classifier import (
            Choice,
            GoogleGenAIClassifier,
            Noul,
            Score,
        )

        classifier = GoogleGenAIClassifier(model="gemini-decision-1")
        response = classifier.invoke(
            {
                "state": "Stripe has failed to connect for three days. Help ASAP.",
                "questions": {
                    "department": Choice(
                        instructions="Which team should handle this request?",
                        criteria={
                            "billing": "Payment or subscription issues.",
                            "technical": "Product bugs or integration failures.",
                        },
                    ),
                    "urgent": Noul(instructions="Is this request urgent?"),
                    "frustration": Score(
                        instructions="How frustrated does the customer appear?",
                        criteria=["Calm.", "Concerned but civil.", "Very angry."],
                    ),
                },
            }
        )
        print(response.choices["department"].choice)
        print(response.nouls["urgent"].noul)
        print(response.scores["frustration"].score)
        ```

    ??? example "Classify an image"

        Standard LangChain media blocks are sent as media wherever they appear in the
        state.

        ```python
        from langchain_core.messages import HumanMessage

        response = classifier.invoke(
            {
                "state": [
                    HumanMessage(
                        content_blocks=[
                            {"type": "text", "text": "Photo of the delivered item."},
                            {
                                "type": "image",
                                "base64": image_data,
                                "mime_type": "image/jpeg",
                            },
                        ]
                    )
                ],
                "questions": {
                    "damaged": Noul(instructions="Did the item arrive damaged?"),
                },
            }
        )
        ```
    """

    model: str = Field(min_length=1)
    """Decisions model name, such as `gemini-decision-1`."""

    google_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["GOOGLE_API_KEY", "GEMINI_API_KEY"], default=None
        ),
    )
    """Google API key.

    If not specified, reads the env vars `GOOGLE_API_KEY` and `GEMINI_API_KEY`, with
    precedence given to `GOOGLE_API_KEY`.
    """

    base_url: str | None = None
    """Base URL for API requests. Uses the `google-genai` SDK default when `None`."""

    timeout: float | None = Field(default=None, gt=0)
    """Request timeout in seconds. Uses the `google-genai` SDK default when `None`."""

    max_retries: int | None = Field(default=None, ge=1)
    """Maximum number of request attempts, including the first.

    Uses the `google-genai` SDK default when `None`. Set to `1` to disable retries.
    """

    client: genai.Client | None = Field(default=None, exclude=True, repr=False)
    """Optional `google.genai.Client` used for requests.

    When supplied, `api_key`, `base_url`, `timeout`, and `max_retries` are ignored.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def _validate_environment(self) -> Self:
        if self.client is None and (
            self.google_api_key is None
            or not self.google_api_key.get_secret_value().strip()
        ):
            msg = (
                "Google API key is required. Pass `api_key` or set `GOOGLE_API_KEY` "
                "or `GEMINI_API_KEY`."
            )
            raise ValueError(msg)
        return self

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_google_genai", "classifier"]

    @property
    @override
    def lc_secrets(self) -> dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @cached_property
    def _client(self) -> genai.Client:
        if self.client is not None:
            client = self.client
        else:
            api_key = self.google_api_key.get_secret_value()  # type: ignore[union-attr]
            client = genai.Client(
                vertexai=False,
                api_key=api_key,
                http_options=self._http_options,
            )
        return client

    @property
    def _http_options(self) -> types.HttpOptions:
        return types.HttpOptions(
            base_url=self.base_url,
            timeout=int(self.timeout * 1000) if self.timeout is not None else None,
            retry_options=(
                types.HttpRetryOptions(attempts=self.max_retries)
                if self.max_retries is not None
                else None
            ),
        )

    @property
    @override
    def _provider(self) -> str:
        return "google_genai"

    @override
    def _classify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        payload = self._payload(state, questions)
        response = self._client.decisions.create(**payload)
        return self._parse_response(response)

    @override
    async def _aclassify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        payload = self._payload(state, questions)
        response = await self._client.aio.decisions.create(**payload)
        return self._parse_response(response)

    def _payload(self, state: State, questions: dict[str, Question]) -> dict[str, Any]:
        return {
            "model": self.model,
            "input": to_decision_input(state),
            "questions": to_decision_questions(questions),
        }

    def _parse_response(self, response: Any) -> ClassifierResponse:
        body = response.model_dump() if hasattr(response, "model_dump") else response
        _raise_if_blocked(body.get("prompt_feedback"))
        usage = body.get("usage")
        return ClassifierResponse.model_validate(
            {
                "model": self.model,
                "answers": body.get("answers") or {},
                "usage": {
                    "input_tokens": usage.get("total_input_tokens"),
                    "output_tokens": usage.get("total_output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
                if usage
                else None,
            }
        )


def _raise_if_blocked(prompt_feedback: dict[str, Any] | None) -> None:
    """Raise `ModelRefusalError` when the API blocked the input."""
    block_reason = (prompt_feedback or {}).get("block_reason")
    if not block_reason:
        return
    msg = f"The Decisions API blocked the input (block_reason={block_reason!r})."
    if details := prompt_feedback.get("block_reason_message"):  # type: ignore[union-attr]
        msg = f"{msg} {details}"
    raise ModelRefusalError(msg)


__all__ = ["GoogleGenAIClassifier"]
