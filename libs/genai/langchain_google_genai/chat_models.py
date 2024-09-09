from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
import warnings
from io import BytesIO
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from urllib.parse import urlparse

import google.api_core

# TODO: remove ignore once the google package is published with types
import proto  # type: ignore[import]
import requests
from google.ai.generativelanguage_v1beta.types import (
    Blob,
    Candidate,
    Content,
    FileData,
    FunctionCall,
    FunctionResponse,
    GenerateContentRequest,
    GenerateContentResponse,
    GenerationConfig,
    Part,
    SafetySetting,
    ToolConfig,
    VideoMetadata,
)
from google.generativeai.types import Tool as GoogleTool  # type: ignore[import]
from google.generativeai.types.content_types import (  # type: ignore[import]
    FunctionDeclarationType,
    ToolDict,
)
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import invalid_tool_call, tool_call, tool_call_chunk
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
    parse_tool_calls,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.utils import secret_from_env
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import Self

from langchain_google_genai._common import (
    GoogleGenerativeAIError,
    SafetySettingDict,
    get_client_info,
)
from langchain_google_genai._function_utils import (
    _tool_choice_to_tool_config,
    _ToolChoiceType,
    _ToolConfigDict,
    convert_to_genai_function_declarations,
    tool_to_dict,
)
from langchain_google_genai._image_utils import ImageBytesLoader
from langchain_google_genai.llms import _BaseGoogleGenerativeAI

from . import _genai_extension as genaix

IMAGE_TYPES: Tuple = ()
try:
    import PIL
    from PIL.Image import Image

    IMAGE_TYPES = IMAGE_TYPES + (Image,)
except ImportError:
    PIL = None  # type: ignore
    Image = None  # type: ignore

logger = logging.getLogger(__name__)


class ChatGoogleGenerativeAIError(GoogleGenerativeAIError):
    """
    Custom exception class for errors associated with the `Google GenAI` API.

    This exception is raised when there are specific issues related to the
    Google genai API usage in the ChatGoogleGenerativeAI class, such as unsupported
    message types or roles.
    """


def _create_retry_decorator() -> Callable[[Any], Any]:
    """
    Creates and returns a preconfigured tenacity retry decorator.

    The retry decorator is configured to handle specific Google API exceptions
    such as ResourceExhausted and ServiceUnavailable. It uses an exponential
    backoff strategy for retries.

    Returns:
        Callable[[Any], Any]: A retry decorator configured for handling specific
        Google API exceptions.
    """
    multiplier = 2
    min_seconds = 1
    max_seconds = 60
    max_retries = 2

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=multiplier, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
            | retry_if_exception_type(google.api_core.exceptions.ServiceUnavailable)
            | retry_if_exception_type(google.api_core.exceptions.GoogleAPIError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _chat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator()

    @retry_decorator
    def _chat_with_retry(**kwargs: Any) -> Any:
        try:
            return generation_method(**kwargs)
        # Do not retry for these errors.
        except google.api_core.exceptions.FailedPrecondition as exc:
            if "location is not supported" in exc.message:
                error_msg = (
                    "Your location is not supported by google-generativeai "
                    "at the moment. Try to use ChatVertexAI LLM from "
                    "langchain_google_vertexai."
                )
                raise ValueError(error_msg)

        except google.api_core.exceptions.InvalidArgument as e:
            raise ChatGoogleGenerativeAIError(
                f"Invalid argument provided to Gemini: {e}"
            ) from e
        except Exception as e:
            raise e

    return _chat_with_retry(**kwargs)


async def _achat_with_retry(generation_method: Callable, **kwargs: Any) -> Any:
    """
    Executes a chat generation method with retry logic using tenacity.

    This function is a wrapper that applies a retry mechanism to a provided
    chat generation function. It is useful for handling intermittent issues
    like network errors or temporary service unavailability.

    Args:
        generation_method (Callable): The chat generation method to be executed.
        **kwargs (Any): Additional keyword arguments to pass to the generation method.

    Returns:
        Any: The result from the chat generation method.
    """
    retry_decorator = _create_retry_decorator()
    from google.api_core.exceptions import InvalidArgument  # type: ignore

    @retry_decorator
    async def _achat_with_retry(**kwargs: Any) -> Any:
        try:
            return await generation_method(**kwargs)
        except InvalidArgument as e:
            # Do not retry for these errors.
            raise ChatGoogleGenerativeAIError(
                f"Invalid argument provided to Gemini: {e}"
            ) from e
        except Exception as e:
            raise e

    return await _achat_with_retry(**kwargs)


def _is_openai_parts_format(part: dict) -> bool:
    return "type" in part


def _is_vision_model(model: str) -> bool:
    return "vision" in model


def _is_url(s: str) -> bool:
    try:
        result = urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _is_b64(s: str) -> bool:
    return s.startswith("data:image")


def _load_image_from_gcs(path: str, project: Optional[str] = None) -> Image:
    try:
        from google.cloud import storage  # type: ignore[attr-defined]
    except ImportError:
        raise ImportError(
            "google-cloud-storage is required to load images from GCS."
            " Install it with `pip install google-cloud-storage`"
        )
    if PIL is None:
        raise ImportError(
            "PIL is required to load images. Please install it "
            "with `pip install pillow`"
        )

    gcs_client = storage.Client(project=project)
    pieces = path.split("/")
    blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))
    if len(blobs) > 1:
        raise ValueError(f"Found more than one candidate for {path}!")
    img_bytes = blobs[0].download_as_bytes()
    return PIL.Image.open(BytesIO(img_bytes))


def _url_to_pil(image_source: str) -> Image:
    if PIL is None:
        raise ImportError(
            "PIL is required to load images. Please install it "
            "with `pip install pillow`"
        )
    try:
        if isinstance(image_source, IMAGE_TYPES):
            return image_source  # type: ignore[return-value]
        elif _is_url(image_source):
            if image_source.startswith("gs://"):
                return _load_image_from_gcs(image_source)
            response = requests.get(image_source)
            response.raise_for_status()
            return PIL.Image.open(BytesIO(response.content))
        elif _is_b64(image_source):
            _, encoded = image_source.split(",", 1)
            data = base64.b64decode(encoded)
            return PIL.Image.open(BytesIO(data))
        elif os.path.exists(image_source):
            return PIL.Image.open(image_source)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _convert_to_parts(
    raw_content: Union[str, Sequence[Union[str, dict]]],
) -> List[Part]:
    """Converts a list of LangChain messages into a google parts."""
    parts = []
    content = [raw_content] if isinstance(raw_content, str) else raw_content
    image_loader = ImageBytesLoader()
    for part in content:
        if isinstance(part, str):
            parts.append(Part(text=part))
        elif isinstance(part, Mapping):
            # OpenAI Format
            if _is_openai_parts_format(part):
                if part["type"] == "text":
                    parts.append(Part(text=part["text"]))
                elif part["type"] == "image_url":
                    img_url = part["image_url"]
                    if isinstance(img_url, dict):
                        if "url" not in img_url:
                            raise ValueError(
                                f"Unrecognized message image format: {img_url}"
                            )
                        img_url = img_url["url"]
                    parts.append(image_loader.load_part(img_url))
                # Handle media type like LangChain.js
                # https://github.com/langchain-ai/langchainjs/blob/e536593e2585f1dd7b0afc187de4d07cb40689ba/libs/langchain-google-common/src/utils/gemini.ts#L93-L106
                elif part["type"] == "media":
                    if "mime_type" not in part:
                        raise ValueError(f"Missing mime_type in media part: {part}")
                    mime_type = part["mime_type"]
                    media_part = Part()

                    if "data" in part:
                        media_part.inline_data = Blob(
                            data=part["data"], mime_type=mime_type
                        )
                    elif "file_uri" in part:
                        media_part.file_data = FileData(
                            file_uri=part["file_uri"], mime_type=mime_type
                        )
                    else:
                        raise ValueError(
                            f"Media part must have either data or file_uri: {part}"
                        )
                    if "video_metadata" in part:
                        metadata = VideoMetadata(part["video_metadata"])
                        media_part.video_metadata = metadata
                    parts.append(media_part)
                else:
                    raise ValueError(
                        f"Unrecognized message part type: {part['type']}. Only text, "
                        f"image_url, and media types are supported."
                    )
            else:
                # Yolo
                logger.warning(
                    "Unrecognized message part format. Assuming it's a text part."
                )
                parts.append(Part(text=str(part)))
        else:
            # TODO: Maybe some of Google's native stuff
            # would hit this branch.
            raise ChatGoogleGenerativeAIError(
                "Gemini only supports text and inline_data parts."
            )
    return parts


def _parse_chat_history(
    input_messages: Sequence[BaseMessage], convert_system_message_to_human: bool = False
) -> Tuple[Optional[Content], List[Content]]:
    messages: List[Content] = []

    if convert_system_message_to_human:
        warnings.warn("Convert_system_message_to_human will be deprecated!")

    system_instruction: Optional[Content] = None
    for i, message in enumerate(input_messages):
        if i == 0 and isinstance(message, SystemMessage):
            system_instruction = Content(parts=_convert_to_parts(message.content))
            continue
        elif isinstance(message, AIMessage):
            role = "model"
            raw_function_call = message.additional_kwargs.get("function_call")
            if raw_function_call:
                function_call = FunctionCall(
                    {
                        "name": raw_function_call["name"],
                        "args": json.loads(raw_function_call["arguments"]),
                    }
                )
                parts = [Part(function_call=function_call)]
            else:
                parts = _convert_to_parts(message.content)
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message.content)
            if i == 1 and convert_system_message_to_human and system_instruction:
                parts = [p for p in system_instruction.parts] + parts
                system_instruction = None
        elif isinstance(message, FunctionMessage):
            role = "user"
            response: Any
            if not isinstance(message.content, str):
                response = message.content
            else:
                try:
                    response = json.loads(message.content)
                except json.JSONDecodeError:
                    response = message.content  # leave as str representation
            parts = [
                Part(
                    function_response=FunctionResponse(
                        name=message.name,
                        response=(
                            {"output": response}
                            if not isinstance(response, dict)
                            else response
                        ),
                    )
                )
            ]
        elif isinstance(message, ToolMessage):
            role = "user"
            prev_message: Optional[BaseMessage] = (
                input_messages[i - 1] if i > 0 else None
            )
            if (
                prev_message
                and isinstance(prev_message, AIMessage)
                and prev_message.tool_calls
            ):
                # message.name can be null for ToolMessage
                name: str = prev_message.tool_calls[0]["name"]
            else:
                name = message.name  # type: ignore
            tool_response: Any
            if not isinstance(message.content, str):
                tool_response = message.content
            else:
                try:
                    tool_response = json.loads(message.content)
                except json.JSONDecodeError:
                    tool_response = message.content  # leave as str representation
            parts = [
                Part(
                    function_response=FunctionResponse(
                        name=name,
                        response=(
                            {"output": tool_response}
                            if not isinstance(tool_response, dict)
                            else tool_response
                        ),
                    )
                )
            ]
        else:
            raise ValueError(
                f"Unexpected message with type {type(message)} at the position {i}."
            )

        messages.append(Content(role=role, parts=parts))
    return system_instruction, messages


def _parse_response_candidate(
    response_candidate: Candidate, streaming: bool = False
) -> AIMessage:
    content: Union[None, str, List[str]] = None
    additional_kwargs = {}
    tool_calls = []
    invalid_tool_calls = []
    tool_call_chunks = []

    for part in response_candidate.content.parts:
        try:
            text: Optional[str] = part.text
        except AttributeError:
            text = None

        if text is not None:
            if not content:
                content = text
            elif isinstance(content, str) and text:
                content = [content, text]
            elif isinstance(content, list) and text:
                content.append(text)
            elif text:
                raise Exception("Unexpected content type")

        if part.function_call:
            function_call = {"name": part.function_call.name}
            # dump to match other function calling llm for now
            function_call_args_dict = proto.Message.to_dict(part.function_call)["args"]
            function_call["arguments"] = json.dumps(
                {k: function_call_args_dict[k] for k in function_call_args_dict}
            )
            additional_kwargs["function_call"] = function_call

            if streaming:
                tool_call_chunks.append(
                    tool_call_chunk(
                        name=function_call.get("name"),
                        args=function_call.get("arguments"),
                        id=function_call.get("id", str(uuid.uuid4())),
                        index=function_call.get("index"),  # type: ignore
                    )
                )
            else:
                try:
                    tool_call_dict = parse_tool_calls(
                        [{"function": function_call}],
                        return_id=False,
                    )[0]
                except Exception as e:
                    invalid_tool_calls.append(
                        invalid_tool_call(
                            name=function_call.get("name"),
                            args=function_call.get("arguments"),
                            id=function_call.get("id", str(uuid.uuid4())),
                            error=str(e),
                        )
                    )
                else:
                    tool_calls.append(
                        tool_call(
                            name=tool_call_dict["name"],
                            args=tool_call_dict["args"],
                            id=tool_call_dict.get("id", str(uuid.uuid4())),
                        )
                    )
    if content is None:
        content = ""

    if streaming:
        return AIMessageChunk(
            content=cast(Union[str, List[Union[str, Dict[Any, Any]]]], content),
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )

    return AIMessage(
        content=cast(Union[str, List[Union[str, Dict[Any, Any]]]], content),
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid_tool_calls,
    )


def _response_to_result(
    response: GenerateContentResponse,
    stream: bool = False,
) -> ChatResult:
    """Converts a PaLM API response into a LangChain ChatResult."""
    llm_output = {"prompt_feedback": proto.Message.to_dict(response.prompt_feedback)}

    # Get usage metadata
    try:
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        if input_tokens + output_tokens + total_tokens > 0:
            lc_usage = UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        else:
            lc_usage = None
    except AttributeError:
        lc_usage = None

    generations: List[ChatGeneration] = []

    for candidate in response.candidates:
        generation_info = {}
        if candidate.finish_reason:
            generation_info["finish_reason"] = candidate.finish_reason.name
        generation_info["safety_ratings"] = [
            proto.Message.to_dict(safety_rating, use_integers_for_enums=False)
            for safety_rating in candidate.safety_ratings
        ]
        message = _parse_response_candidate(candidate, streaming=stream)
        message.usage_metadata = lc_usage
        generations.append(
            (ChatGenerationChunk if stream else ChatGeneration)(
                message=message,
                generation_info=generation_info,
            )
        )
    if not response.candidates:
        # Likely a "prompt feedback" violation (e.g., toxic input)
        # Raising an error would be different than how OpenAI handles it,
        # so we'll just log a warning and continue with an empty message.
        logger.warning(
            "Gemini produced an empty response. Continuing with empty message\n"
            f"Feedback: {response.prompt_feedback}"
        )
        generations = [
            (ChatGenerationChunk if stream else ChatGeneration)(
                message=(AIMessageChunk if stream else AIMessage)(content=""),
                generation_info={},
            )
        ]
    return ChatResult(generations=generations, llm_output=llm_output)


def _is_event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class ChatGoogleGenerativeAI(_BaseGoogleGenerativeAI, BaseChatModel):
    """`Google AI` chat models integration.

    Instantiation:
        To use, you must have either:

            1. The ``GOOGLE_API_KEY``` environment variable set with your API key, or
            2. Pass your API key using the google_api_key kwarg to the ChatGoogle
               constructor.

        .. code-block:: python

            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
            llm.invoke("Write me a ballad about LangChain")

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore programmer. \\n",
                response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'safety_ratings': [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'blocked': False}]},
                id='run-56cecc34-2e54-4b52-a974-337e47008ad2-0',
                usage_metadata={'input_tokens': 18, 'output_tokens': 5, 'total_tokens': 23}
            )

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            AIMessageChunk(content='J', response_metadata={'finish_reason': 'STOP', 'safety_ratings': []}, id='run-e905f4f4-58cb-4a10-a960-448a2bb649e3', usage_metadata={'input_tokens': 18, 'output_tokens': 1, 'total_tokens': 19})
            AIMessageChunk(content="'adore programmer. \n", response_metadata={'finish_reason': 'STOP', 'safety_ratings': [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'blocked': False}]}, id='run-e905f4f4-58cb-4a10-a960-448a2bb649e3', usage_metadata={'input_tokens': 18, 'output_tokens': 5, 'total_tokens': 23})

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(
                content="J'adore programmer. \\n",
                response_metadata={'finish_reason': 'STOPSTOP', 'safety_ratings': [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'blocked': False}]},
                id='run-3ce13a42-cd30-4ad7-a684-f1f0b37cdeec',
                usage_metadata={'input_tokens': 36, 'output_tokens': 6, 'total_tokens': 42}
            )

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field


            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is hotter today and which is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [{'name': 'GetWeather',
              'args': {'location': 'Los Angeles, CA'},
              'id': 'c186c99f-f137-4d52-947f-9e3deabba6f6'},
             {'name': 'GetWeather',
              'args': {'location': 'New York City, NY'},
              'id': 'cebd4a5d-e800-4fa5-babd-4aa286af4f31'},
             {'name': 'GetPopulation',
              'args': {'location': 'Los Angeles, CA'},
              'id': '4f92d897-f5e4-4d34-a3bc-93062c92591e'},
             {'name': 'GetPopulation',
              'args': {'location': 'New York City, NY'},
              'id': '634582de-5186-4e4b-968b-f192f0a93678'}]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup='Why are cats so good at video games?',
                punchline='They have nine lives on the internet',
                rating=None
            )

    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            'The weather in this image appears to be sunny and pleasant. The sky is a bright blue with scattered white clouds, suggesting fair weather. The lush green grass and trees indicate a warm and possibly slightly breezy day. There are no signs of rain or storms. \n'

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 18, 'output_tokens': 5, 'total_tokens': 23}


        Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'prompt_feedback': {'block_reason': 0, 'safety_ratings': []},
                'finish_reason': 'STOP',
                'safety_ratings': [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'blocked': False}]
            }

    """  # noqa: E501

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    google_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("GOOGLE_API_KEY", default=None)
    )
    """Google AI API key.         
    If not specified will be read from env var ``GOOGLE_API_KEY``."""
    default_metadata: Sequence[Tuple[str, str]] = Field(
        default_factory=list
    )  #: :meta private:

    convert_system_message_to_human: bool = False
    """Whether to merge any leading SystemMessage into the following HumanMessage.

    Gemini does not support system messages; any unsupported messages will
    raise an error."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"google_api_key": "GOOGLE_API_KEY"}

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai"

    @classmethod
    def is_lc_serializable(self) -> bool:
        return True

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and passes them to google-generativeai package."""
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")

        if not self.model.startswith("models/"):
            self.model = f"models/{self.model}"

        additional_headers = self.additional_headers or {}
        self.default_metadata = tuple(additional_headers.items())
        client_info = get_client_info("ChatGoogleGenerativeAI")
        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key
        transport: Optional[str] = self.transport
        self.client = genaix.build_generative_service(
            credentials=self.credentials,
            api_key=google_api_key,
            client_info=client_info,
            client_options=self.client_options,
            transport=transport,
        )

        # NOTE: genaix.build_generative_async_service requires
        # a running event loop, which causes an error
        # when initialized inside a ThreadPoolExecutor.
        # this check ensures that async client is only initialized
        # within an asyncio event loop to avoid the error
        if _is_event_loop_running():
            self.async_client = genaix.build_generative_async_service(
                credentials=self.credentials,
                api_key=google_api_key,
                client_info=client_info,
                client_options=self.client_options,
                transport=transport,
            )
        else:
            self.async_client = None

        return self

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "n": self.n,
            "safety_settings": self.safety_settings,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="google_genai",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_output_tokens", self.max_output_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _prepare_params(
        self,
        stop: Optional[List[str]],
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> GenerationConfig:
        gen_config = {
            k: v
            for k, v in {
                "candidate_count": self.n,
                "temperature": self.temperature,
                "stop_sequences": stop,
                "max_output_tokens": self.max_output_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }.items()
            if v is not None
        }
        if generation_config:
            gen_config = {**gen_config, **generation_config}
        return GenerationConfig(**gen_config)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
        )
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            **kwargs,
            generation_method=self.client.generate_content,
            metadata=self.default_metadata,
        )
        return _response_to_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.async_client:
            updated_kwargs = {
                **kwargs,
                **{
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                },
            }
            return await super()._agenerate(
                messages, stop, run_manager, **updated_kwargs
            )

        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
        )
        response: GenerateContentResponse = await _achat_with_retry(
            request=request,
            **kwargs,
            generation_method=self.async_client.generate_content,
            metadata=self.default_metadata,
        )
        return _response_to_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request(
            messages,
            stop=stop,
            tools=tools,
            functions=functions,
            safety_settings=safety_settings,
            tool_config=tool_config,
            generation_config=generation_config,
        )
        response: GenerateContentResponse = _chat_with_retry(
            request=request,
            generation_method=self.client.stream_generate_content,
            **kwargs,
            metadata=self.default_metadata,
        )
        for chunk in response:
            _chat_result = _response_to_result(chunk, stream=True)
            gen = cast(ChatGenerationChunk, _chat_result.generations[0])

            if run_manager:
                run_manager.on_llm_new_token(gen.text)
            yield gen

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if not self.async_client:
            updated_kwargs = {
                **kwargs,
                **{
                    "tools": tools,
                    "functions": functions,
                    "safety_settings": safety_settings,
                    "tool_config": tool_config,
                    "generation_config": generation_config,
                },
            }
            async for value in super()._astream(
                messages, stop, run_manager, **updated_kwargs
            ):
                yield value
        else:
            request = self._prepare_request(
                messages,
                stop=stop,
                tools=tools,
                functions=functions,
                safety_settings=safety_settings,
                tool_config=tool_config,
                generation_config=generation_config,
            )
            async for chunk in await _achat_with_retry(
                request=request,
                generation_method=self.async_client.stream_generate_content,
                **kwargs,
                metadata=self.default_metadata,
            ):
                _chat_result = _response_to_result(chunk, stream=True)
                gen = cast(ChatGenerationChunk, _chat_result.generations[0])

                if run_manager:
                    await run_manager.on_llm_new_token(gen.text)
                yield gen

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        tools: Optional[Sequence[Union[ToolDict, GoogleTool]]] = None,
        functions: Optional[Sequence[FunctionDeclarationType]] = None,
        safety_settings: Optional[SafetySettingDict] = None,
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GenerateContentRequest, Dict[str, Any]]:
        formatted_tools = None
        if tools:
            formatted_tools = [convert_to_genai_function_declarations(tools)]
        elif functions:
            formatted_tools = [convert_to_genai_function_declarations(functions)]

        system_instruction, history = _parse_chat_history(
            messages,
            convert_system_message_to_human=self.convert_system_message_to_human,
        )
        formatted_tool_config = None
        if tool_config:
            formatted_tool_config = ToolConfig(
                function_calling_config=tool_config["function_calling_config"]
            )
        formatted_safety_settings = []
        if safety_settings:
            formatted_safety_settings = [
                SafetySetting(category=c, threshold=t)
                for c, t in safety_settings.items()
            ]
        request = GenerateContentRequest(
            model=self.model,
            contents=history,
            tools=formatted_tools,
            tool_config=formatted_tool_config,
            safety_settings=formatted_safety_settings,
            generation_config=self._prepare_params(
                stop, generation_config=generation_config
            ),
        )
        if system_instruction:
            request.system_instruction = system_instruction

        return request

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input will fit in a model's context window.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.
        """
        result = self.client.count_tokens(
            model=self.model, contents=[Content(parts=[Part(text=text)])]
        )
        return result.total_tokens

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            parser = JsonOutputToolsParser()
        tool_choice = _get_tool_name(schema) if self._supports_tool_choice else None
        llm = self.bind_tools([schema], tool_choice=tool_choice)
        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback
        else:
            return llm | parser

    def bind_tools(
        self,
        tools: Sequence[Union[ToolDict, GoogleTool]],
        tool_config: Optional[Union[Dict, _ToolConfigDict]] = None,
        *,
        tool_choice: Optional[Union[_ToolChoiceType, bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with google-generativeAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be a pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        if tool_choice and tool_config:
            raise ValueError(
                "Must specify at most one of tool_choice and tool_config, received "
                f"both:\n\n{tool_choice=}\n\n{tool_config=}"
            )
        # Bind dicts for easier serialization/deserialization.
        genai_tools = [tool_to_dict(convert_to_genai_function_declarations(tools))]
        if tool_choice:
            all_names = [
                f["name"]  # type: ignore[index]
                for t in genai_tools
                for f in t["function_declarations"]
            ]
            tool_config = _tool_choice_to_tool_config(tool_choice, all_names)
        return self.bind(tools=genai_tools, tool_config=tool_config, **kwargs)

    @property
    def _supports_tool_choice(self) -> bool:
        return "gemini-1.5-pro" in self.model


def _get_tool_name(
    tool: Union[ToolDict, GoogleTool],
) -> str:
    genai_tool = tool_to_dict(convert_to_genai_function_declarations([tool]))
    return [f["name"] for f in genai_tool["function_declarations"]][0]  # type: ignore[index]
