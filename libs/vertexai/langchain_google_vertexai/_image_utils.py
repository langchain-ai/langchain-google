from __future__ import annotations

import base64
import os
import re
from enum import Enum
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import requests
from google.cloud import storage
from google.cloud.aiplatform_v1beta1.types.content import Part as GapicPart
from vertexai.generative_models import Image, Part  # type: ignore


class Route(Enum):
    """Image Loading Route"""

    GOOGLE_CLOUD_STORAGE = 1
    BASE64 = 2
    LOCAL_FILE = 3
    URL = 4


class ImageBytesLoader:
    """Loads image bytes from multiple sources given a string.

    Currently supported:
        - Google cloud storage URI
        - B64 Encoded image string
        - Local file path
        - URL
    """

    def __init__(
        self,
        project: Union[str, None] = None,
    ) -> None:
        """Constructor

        Args:
            project: Google Cloud project id. Defaults to none.
        """
        self._project = project

    def load_bytes(self, image_string: str) -> bytes:
        """Routes to the correct loader based on the image_string.

        Args:
            image_string: Can be either:
                    - Google cloud storage URI
                    - B64 Encoded image string
                    - URL

        Returns:
            Image bytes.
        """

        route = self._route(image_string)

        if route == Route.GOOGLE_CLOUD_STORAGE:
            blob = self._blob_from_gcs(image_string)
            return blob.download_as_bytes()

        if route == Route.BASE64:
            return self._bytes_from_b64(image_string)

        if route == Route.URL:
            return self._bytes_from_url(image_string)

        if route == Route.LOCAL_FILE:
            msg = (
                "Support for loading local files has been removed for security "
                "reasons. Please pass in images as one of: "
                "Google Cloud Storage URI, b64 encoded image string (data:image/...), "
                "or valid image url. "
            )
            raise ValueError(msg)

        raise ValueError(
            "Image string must be one of: Google Cloud Storage URI, "
            "b64 encoded image string (data:image/...), valid image url. "
            f"Instead got '{image_string}'."
        )

    def load_part(self, image_string: str) -> Part:
        """Gets Part for loading from Gemini.

        Args:
            image_string: Can be either:
                    - Google cloud storage URI
                    - B64 Encoded image string
                    - Local file path
                    - URL

        Returns:
            generative_models.Part
        """
        route = self._route(image_string)

        if route == Route.GOOGLE_CLOUD_STORAGE:
            blob = self._blob_from_gcs(image_string)
            return Part.from_uri(uri=image_string, mime_type=blob.content_type)

        if route == Route.BASE64:
            bytes_ = self._bytes_from_b64(image_string)

        if route == Route.URL:
            bytes_ = self._bytes_from_url(image_string)

        if route == Route.LOCAL_FILE:
            msg = (
                "Support for loading local files has been removed for security "
                "reasons. Please pass in images as one of: "
                "Google Cloud Storage URI, b64 encoded image string (data:image/...), "
                "or valid image url. "
            )
            raise ValueError(msg)

        mime_type = self._has_known_mimetype(image_string)
        if mime_type:
            return Part.from_data(bytes_, mime_type=mime_type)

        return Part.from_image(Image.from_bytes(bytes_))

    def load_gapic_part(self, image_string: str) -> GapicPart:
        part = self.load_part(image_string)
        return part._raw_part

    def _route(self, image_string: str) -> Route:
        if image_string.startswith("gs://"):
            return Route.GOOGLE_CLOUD_STORAGE

        if image_string.startswith("data:"):
            return Route.BASE64

        if self._is_url(image_string):
            return Route.URL

        if os.path.exists(image_string):
            return Route.LOCAL_FILE

        raise ValueError(
            "Image string must be one of: Google Cloud Storage URI, "
            "b64 encoded image string (data:image/...), or valid image url. "
            f"Instead got '{image_string}'."
        )

    def _bytes_from_b64(self, base64_image: str) -> bytes:
        """Gets image bytes from a base64 encoded string.

        Args:
            base64_image: Encoded image in b64 format.

        Returns:
            Image bytes
        """

        pattern = r"data:\w+/\w{2,4};base64,(.*)"
        match = re.search(pattern, base64_image)

        if match is not None:
            encoded_string = match.group(1)
            return base64.b64decode(encoded_string)

        raise ValueError(f"Error in b64 encoded image. Must follow pattern: {pattern}")

    def _bytes_from_url(self, url: str) -> bytes:
        """Gets image bytes from a public url.

        Args:
            url: Valid url.

        Raises:
            HTTP Error if there is one.

        Returns:
            Image bytes
        """

        response = requests.get(url)

        if not response.ok:
            response.raise_for_status()

        return response.content

    def _blob_from_gcs(self, gcs_uri: str) -> storage.Blob:
        """Gets image Blob from a Google Cloud Storage uri.

        Args:
            gcs_uri: Valid gcs uri.

        Raises:
            ValueError if there are more than one blob matching the uri.

        Returns:
            storage.Blob
        """

        gcs_client = storage.Client(project=self._project)
        blob = storage.Blob.from_string(gcs_uri, gcs_client)
        blob.reload(client=gcs_client)
        return blob

    def _is_url(self, url_string: str) -> bool:
        """Checks if a url is valid.

        Args:
            url_string: Url to check.

        Returns:
            Whether the url is valid.
        """
        try:
            result = urlparse(url_string)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _has_known_mimetype(self, image_url: str) -> Optional[str]:
        """Checks weather the image needs other mimetype. Currently only identifies
        pdfs, otherwise it will return None and it will be treated as an image.
        """

        # For local files or urls
        if image_url.endswith(".pdf"):
            return "application/pdf"

        # for b64 encoded data
        if image_url.startswith("data:application/pdf;base64"):
            return "application/pdf"

        return None


def image_bytes_to_b64_string(
    image_bytes: bytes, encoding: str = "ascii", image_format: str = "png"
) -> str:
    """Encodes image bytes into a b64 encoded string.

    Args:
        image_bytes: Bytes of the image.
        encoding: Type of encoding in the string. 'ascii' by default.
        image_format: Format of the image. 'png' by default.

    Returns:
        B64 image encoded string.
    """
    encoded_bytes = base64.b64encode(image_bytes).decode(encoding)
    return f"data:image/{image_format};base64,{encoded_bytes}"


def create_text_content_part(message_str: str) -> Dict:
    """Create a dictionary that can be part of a message content list.

    Args:
        message_str: Message as an string.

    Returns:
        Dictionary that can be part of a message content list.
    """
    return {"type": "text", "text": message_str}


def create_image_content_part(image_str: str) -> Dict:
    """Create a dictionary that can be part of a message content list.

    Args:
        image_str: Can be either:
            - b64 encoded image data
            - GCS uri
            - Url
            - Path to an image.

    Returns:
        Dictionary that can be part of a message content list.
    """
    return {"type": "image_url", "image_url": {"url": image_str}}


def get_image_str_from_content_part(content_part: str | Dict) -> str | None:
    """Parses an image string from a dictionary with the correct format.

    Args:
        content_part: String or dictionary.

    Returns:
        Image string if the dictionary has the correct format otherwise None.
    """

    if isinstance(content_part, str):
        return None

    if content_part.get("type") != "image_url":
        return None

    image_str = content_part.get("image_url", {}).get("url")

    if isinstance(image_str, str):
        return image_str
    else:
        return None


def get_text_str_from_content_part(content_part: str | Dict) -> str | None:
    """Parses an string from a dictionary or string with the correct format.

    Args:
        content_part:  String or dictionary.

    Returns:
        String if the dictionary has the correct format or the input is an string,
        otherwise None.
    """

    if isinstance(content_part, str):
        return content_part

    if content_part.get("type") != "text":
        return None

    text = content_part.get("text")

    if isinstance(text, str):
        return text
    else:
        return None
