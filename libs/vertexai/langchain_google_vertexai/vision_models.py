from __future__ import annotations

from typing import Any, Dict, List, Union

from google.cloud.aiplatform import telemetry
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, LLMResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.generation import Generation
from langchain_core.pydantic_v1 import BaseModel, Field
from vertexai.preview.vision_models import (  # type: ignore[import-untyped]
    GeneratedImage,
    ImageGenerationModel,
)
from vertexai.vision_models import Image, ImageTextModel  # type: ignore[import-untyped]

from langchain_google_vertexai._image_utils import (
    ImageBytesLoader,
    create_image_content_part,
    get_image_str_from_content_part,
    get_text_str_from_content_part,
    image_bytes_to_b64_string,
)
from langchain_google_vertexai._utils import get_user_agent


class _BaseImageTextModel(BaseModel):
    """Base class for all integrations that use ImageTextModel"""

    model_name: str = Field(default="imagetext@001")
    """ Name of the model to use"""
    number_of_results: int = Field(default=1)
    """Number of results to return from one query"""
    language: str = Field(default="en")
    """Language of the query"""
    project: Union[str, None] = Field(default=None)
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

        image_str = get_image_str_from_content_part(message_part)

        if isinstance(image_str, str):
            loader = ImageBytesLoader(project=self.project)
            image_bytes = loader.load_bytes(image_str)
            return Image(image_bytes=image_bytes)
        else:
            return None

    def _get_text_from_message_part(self, message_part: str | Dict) -> str | None:
        """Given a message part obtain a text if the part represents it.

        Args:
            message_part: Item of a message content.

        Returns:
            str is successful otherwise None.
        """
        return get_text_str_from_content_part(message_part)

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM"""
        return "vertexai-vision"

    @property
    def _user_agent(self) -> str:
        """Gets the User Agent."""
        _, user_agent = get_user_agent(f"{type(self).__name__}_{self.model_name}")
        return user_agent


class _BaseVertexAIImageCaptioning(_BaseImageTextModel):
    """Base class for Image Captioning models."""

    def _get_captions(self, image: Image) -> List[str]:
        """Uses the sdk methods to generate a list of captions.

        Args:
            image: Image to get the captions for.

        Returns:
            List of captions obtained from the image.
        """
        with telemetry.tool_context_manager(self._user_agent):
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
        with telemetry.tool_context_manager(self._user_agent):
            model = self._create_model()
            answers = model.ask_question(
                image=image, question=query, number_of_results=self.number_of_results
            )
            return answers


class _BaseVertexAIImageGenerator(BaseModel):
    """Base class form generation and edition of images."""

    model_name: str = Field(default="imagegeneration@002")
    """Name of the base model"""
    negative_prompt: Union[str, None] = Field(default=None)
    """A description of what you want to omit in
        the generated images"""
    number_of_images: int = Field(default=1)
    """Number of images to generate"""
    guidance_scale: Union[float, None] = Field(default=None)
    """Controls the strength of the prompt"""
    language: Union[str, None] = Field(default=None)
    """Language of the text prompt for the image Supported values are "en" for English, 
    "hi" for Hindi, "ja" for Japanese, "ko" for Korean, and "auto" for automatic 
    language detection"""
    seed: Union[int, None] = Field(default=None)
    """Random seed for the image generation"""
    project: Union[str, None] = Field(default=None)
    """Google cloud project id"""

    def _generate_images(self, prompt: str) -> List[str]:
        """Generates images given a prompt.

        Args:
            prompt: Description of what the image should look like.

        Returns:
            List of b64 encoded strings.
        """
        with telemetry.tool_context_manager(self._user_agent):
            model = ImageGenerationModel.from_pretrained(self.model_name)

            generation_result = model.generate_images(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                number_of_images=self.number_of_images,
                language=self.language,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )

        image_str_list = [
            self._to_b64_string(image) for image in generation_result.images
        ]

        return image_str_list

    def _edit_images(self, image_str: str, prompt: str) -> List[str]:
        """Edit an image given a image and a prompt.

        Args:
            image_str: String representation of the image.
            prompt: Description of what the image should look like.

        Returns:
            List of b64 encoded strings.
        """
        with telemetry.tool_context_manager(self._user_agent):
            model = ImageGenerationModel.from_pretrained(self.model_name)

            image_loader = ImageBytesLoader(project=self.project)
            image_bytes = image_loader.load_bytes(image_str)
            image = Image(image_bytes=image_bytes)

            generation_result = model.edit_image(
                prompt=prompt,
                base_image=image,
                negative_prompt=self.negative_prompt,
                number_of_images=self.number_of_images,
                language=self.language,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            )

        image_str_list = [
            self._to_b64_string(image) for image in generation_result.images
        ]

        return image_str_list

    def _to_b64_string(self, image: GeneratedImage) -> str:
        """Transforms a generated image into a b64 encoded string.

        Args:
            image: Image to convert.

        Returns:
            b64 encoded string of the image.
        """

        # This is a hack because at the moment, GeneratedImage doesn't provide
        # a way to get the bytes of the image (or anything else). There is
        # only private methods that are not reliable.

        from tempfile import NamedTemporaryFile

        temp_file = NamedTemporaryFile()
        image.save(temp_file.name, include_generation_parameters=False)
        temp_file.seek(0)
        image_bytes = temp_file.read()
        temp_file.close()

        return image_bytes_to_b64_string(image_bytes=image_bytes)

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM"""
        return "vertexai-vision"

    @property
    def _user_agent(self) -> str:
        """Gets the User Agent."""
        _, user_agent = get_user_agent(f"{type(self).__name__}_{self.model_name}")
        return user_agent


class VertexAIImageGeneratorChat(_BaseVertexAIImageGenerator, BaseChatModel):
    """Generates an image from a prompt."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Args:
            messages: The message must be a list of only one element with one part:
                The user prompt.
        """

        # Only one message allowed with one text part.
        user_query = None
        is_valid = len(messages) == 1 and len(messages[0].content) == 1
        if is_valid:
            user_query = get_text_str_from_content_part(messages[0].content[0])
        if user_query is None:
            raise ValueError(
                "Only one message with one text part allowed for image generation"
                " Must The prompt of the image"
            )

        image_str_list = self._generate_images(prompt=user_query)
        image_content_part_list = [
            create_image_content_part(image_str=image_str)
            for image_str in image_str_list
        ]

        generations = [
            ChatGeneration(message=AIMessage(content=[content_part]))
            for content_part in image_content_part_list
        ]

        return ChatResult(generations=generations)


class VertexAIImageEditorChat(_BaseVertexAIImageGenerator, BaseChatModel):
    """Given an image and a prompt, edits the image.
    Currently only supports mask free editing.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Args:
            messages: The message must be a list of only one element with two part:
                - The image as a dict {
                    'type': 'image_url', 'image_url': {'url': <message_str>}
                    }
                - The user prompt.
        """

        # Only one message allowed with two parts: the image and the text.
        user_query = None
        is_valid = len(messages) == 1 and len(messages[0].content) == 2
        if is_valid:
            image_str = get_image_str_from_content_part(messages[0].content[0])
            user_query = get_text_str_from_content_part(messages[0].content[1])
        if (user_query is None) or (image_str is None):
            raise ValueError(
                "Only one message allowed for image edition. The message must have"
                "two parts: First the image and then the user prompt."
            )

        image_str_list = self._edit_images(image_str=image_str, prompt=user_query)
        image_content_part_list = [
            create_image_content_part(image_str=image_str)
            for image_str in image_str_list
        ]

        generations = [
            ChatGeneration(message=AIMessage(content=[content_part]))
            for content_part in image_content_part_list
        ]

        return ChatResult(generations=generations)
