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

    @staticmethod
    def _get_image_from_message(message: BaseMessage) -> Image:
        """Extracts an image from a message.

        Args:
            message: Message to extract the image from.

        Returns:
            Image extracted from the message.
        """

        loader = ImageBytesLoader()

        if isinstance(message.content, str):
            return Image(loader.load_bytes(image_string=message.content))

        if isinstance(message.content, List):
            if len(message.content) > 1:
                raise ValueError(
                    "Expected message content to have only one part"
                    f"but found {len(message.content)}."
                )

            content = message.content[0]

            if isinstance(content, str):
                return Image(loader.load_bytes(content))

            if isinstance(content, Dict):
                image_url = content.get("image_url", {}).get("url")

                if image_url is not None:
                    return Image(loader.load_bytes(image_url))

                raise ValueError(f"Message content: {content} is not an image.")

            raise ValueError(
                "Expected message content part to be either a str or a "
                f"list, but found a {content.__class__} instance"
            )

        raise ValueError(
            "Message content must be either a str or a List, but found"
            f"an instance of {message.content.__class__}."
        )

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
                The message must contain a string representation of the image.
                Currently supported are:
                - Google Cloud Storage URI
                - B64 encoded string
                - Local file path
                - Remote url
        """

        if len(messages) != 1:
            raise ValueError(
                "Image captioning only works with one message: the image. "
                f"instead got {len(messages)}"
            )

        message = messages[0]
        image = self._get_image_from_message(message)
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

        if len(messages) < 2:
            raise ValueError(
                "Image QnA must have at least two messages: First the"
                "image and then the question and answers. Instead got "
                f"{len(messages)} messages."
            )

        image = self._get_image_from_message(messages[0])

        query = self._build_query(messages=messages[1:])

        answers = self._ask_questions(image=image, query=query)

        generations = [
            ChatGeneration(message=AIMessage(content=answer)) for answer in answers
        ]

        return ChatResult(generations=generations)

    def _build_query(self, messages: List[BaseMessage]) -> str:
        """Builds the query from the messages.

        Args:
            messages: List of text messages.

        Returns:
            Composed query.
        """

        query = ""

        for message in messages:
            content = message.content

            if isinstance(content, str):
                content = [
                    content,
                ]

            full_message_content = ""
            for content_part in content:
                if isinstance(content_part, str):
                    full_message_content += content_part
                else:
                    raise ValueError("All query message content parts must be str.")

            query += f"{message.type}: {full_message_content}\n"

        return query

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
