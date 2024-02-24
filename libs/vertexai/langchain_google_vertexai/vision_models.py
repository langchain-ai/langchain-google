from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import BaseModel, Field
from vertexai.vision_models import Image, ImageTextModel  # type: ignore[import-untyped]

from langchain_google_vertexai._image_utils import ImageBytesLoader


class _BaseImageTextModel(BaseModel):
    """Base class for all integrations that use ImageTextModel"""

    model_name: str = Field(default="imagetext@001")
    """ Name of the model to use"""
    number_of_results: int = Field(default=1)
    """Number of results to return from one query"""
    language: str = Field(default="en")
    """Language of the query"""
    project: str = Field(default=None)
    """Google cloud project"""

    def _create_model(self) -> ImageTextModel:
        """Builds the model object from the class attributes."""
        return ImageTextModel.from_pretrained(model_name=self.model_name)

    def _get_image_from_message_part(self, message_part: str | Dict) -> Image | None:
        """Given a message part obtain a image if the part represents it.

        Args:
            message_part: Item of a message content.

        Returns:
            Image is successful otherwise None.
        """

        if isinstance(message_part, str):
            return None

        if message_part.get("type") != "image_url":
            return None

        image_str = message_part.get("image_url", {}).get("url")

        if not isinstance(image_str, str):
            return None

        loader = ImageBytesLoader(project=self.project)
        image_bytes = loader.load_bytes(image_str)
        return Image(image_bytes=image_bytes)

    def _get_text_from_message_part(self, message_part: str | Dict) -> str | None:
        """Given a message part obtain a text if the part represents it.

        Args:
            message_part: Item of a message content.

        Returns:
            str is successful otherwise None.
        """

        if isinstance(message_part, str):
            return message_part

        if message_part.get("type") != "text":
            return None

        message_text = message_part.get("text")

        if not isinstance(message_text, str):
            return None

        return message_text

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM"""
        return "vertexai-vision"


class _BaseVertexAIImageCaptioning(_BaseImageTextModel):
    """Base class for Image Captioning models."""

    def _get_captions(self, image: Image) -> List[str]:
        """Uses the sdk methods to generate a list of captions.

        Args:
            image: Image to get the captions for.

        Returns:
            List of captions obtained from the image.
        """
        model = self._create_model()
        captions = model.get_captions(
            image=image,
            number_of_results=self.number_of_results,
            language=self.language,
        )
        return captions


class VertexAIImageCaptioning(_BaseVertexAIImageCaptioning, BaseLLM):
    """Implementation of the Image Captioning model as an LLM."""

    def _generate(
        self,
        prompts: List[str],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generates the captions.

        Args:
            prompts: List of prompts to use. Each prompt must be a string
                that represents an image. Currently supported are:
                - Google Cloud Storage URI
                - B64 encoded string
                - Local file path
                - Remote url

        Returns:
            Captions generated from every prompt.
        """

        generations = [self._generate_one(prompt=prompt) for prompt in prompts]

        return LLMResult(generations=generations)

    def _generate_one(self, prompt: str) -> List[Generation]:
        """Generates the captions for a single prompt.

        Args:
            prompt: Image url for the generation.

        Returns:
            List of generations
        """
        image_loader = ImageBytesLoader(project=self.project)
        image_bytes = image_loader.load_bytes(prompt)
        image = Image(image_bytes=image_bytes)
        caption_list = self._get_captions(image=image)
        return [Generation(text=caption) for caption in caption_list]


class VertexAIImageCaptioningChat(_BaseVertexAIImageCaptioning, BaseChatModel):
    """Implementation of the Image Captioning model as a chat."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates the results.

        Args:
            messages: List of messages. Currently only one message is supported.
                The message content must be a list with only one element with
                a dict with format:
                {
                    'type': 'image_url',
                    'image_url': {
                        'url' <image_string>
                    }
                }
                Currently supported image strings are:
                - Google Cloud Storage URI
                - B64 encoded string
                - Local file path
                - Remote url
        """

        image = None

        is_valid = (
            len(messages) == 1
            and isinstance(messages[0].content, List)
            and len(messages[0].content) == 1
        )

        if is_valid:
            content = messages[0].content[0]
            image = self._get_image_from_message_part(content)

        if image is None:
            raise ValueError(
                f"{self.__class__.__name__} messages should be a list with "
                "only one message. This message content must be a list with "
                "one dictionary with the format: "
                "{'type': 'image_url', 'image_url': {'image': <image_str>}}"
            )

        captions = self._get_captions(image)

        generations = [
            ChatGeneration(message=AIMessage(content=caption)) for caption in captions
        ]

        return ChatResult(generations=generations)


class VertexAIVisualQnAChat(_BaseImageTextModel, BaseChatModel):
    """Chat implementation of a visual QnA model"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generates the results.

        Args:
            messages: List of messages. The first message should contain a
            string representation of the image.
                Currently supported are:
                    - Google Cloud Storage URI
                    - B64 encoded string
                    - Local file path
                    - Remote url
                There has to be at least other message with the first question.
        """

        image = None
        user_question = None

        is_valid = (
            len(messages) == 1
            and isinstance(messages[0].content, List)
            and len(messages[0].content) == 2
        )

        if is_valid:
            image_part = messages[0].content[0]
            user_question_part = messages[0].content[1]
            image = self._get_image_from_message_part(image_part)
            user_question = self._get_text_from_message_part(user_question_part)

        if (image is None) or (user_question is None):
            raise ValueError(
                f"{self.__class__.__name__} messages should be a list with "
                "only one message. The message content should be a list with "
                "two elements. The first element should be the image, a dictionary "
                "with format"
                "{'type': 'image_url', 'image_url': {'image': <image_str>}}."
                "The second one should be the user question. Either a simple string"
                "or a dictionary with format {'type': 'text', 'text': <message>}"
            )

        answers = self._ask_questions(image=image, query=user_question)

        generations = [
            ChatGeneration(message=AIMessage(content=answer)) for answer in answers
        ]

        return ChatResult(generations=generations)

    def _ask_questions(self, image: Image, query: str) -> List[str]:
        """Interfaces with the sdk to get the question.

        Args:
            image: Image to question about.
            query: User query.

        Returns:
            List of responses to the query.
        """
        model = self._create_model()
        answers = model.ask_question(
            image=image, question=query, number_of_results=self.number_of_results
        )
        return answers
