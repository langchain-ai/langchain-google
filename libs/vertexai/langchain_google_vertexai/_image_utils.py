import base64
import os
import re
from typing import Union
from urllib.parse import urlparse

import requests
from google.cloud import storage  # type: ignore[attr-defined]


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
                    - Local file path
                    - URL

        Returns:
            Image bytes.
        """

        if image_string.startswith("gs://"):
            return self._bytes_from_gsc(image_string)

        if image_string.startswith("data:image/"):
            return self._bytes_from_b64(image_string)

        if self._is_url(image_string):
            return self._bytes_from_url(image_string)

        if os.path.exists(image_string):
            return self._bytes_from_file(image_string)

        raise ValueError(
            "Image string must be one of: Google Cloud Storage URI, "
            "b64 encoded image string (data:image/...), valid image url, "
            f"or existing local image file. Instead got '{image_string}'."
        )

    def _bytes_from_b64(self, base64_image: str) -> bytes:
        """Gets image bytes from a base64 encoded string.

        Args:
            base64_image: Encoded image in b64 format.

        Returns:
            Image bytes
        """

        pattern = r"data:image/\w{2,4};base64,(.*)"
        match = re.search(pattern, base64_image)

        if match is not None:
            encoded_string = match.group(1)
            return base64.b64decode(encoded_string)

        raise ValueError(f"Error in b64 encoded image. Must follow pattern: {pattern}")

    def _bytes_from_file(self, file_path: str) -> bytes:
        """Gets image bytes from a local file path.

        Args:
            file_path: Existing file path.

        Returns:
            Image bytes
        """
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
        return image_bytes

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

    def _bytes_from_gsc(self, gcs_uri: str) -> bytes:
        """Gets image bytes from a google cloud storage uri.

        Args:
            gcs_uri: Valid gcs uri.

        Raises:
            ValueError if there are more than one blob matching the uri.

        Returns:
            Image bytes
        """

        gcs_client = storage.Client(project=self._project)

        pieces = gcs_uri.split("/")

        blobs = list(gcs_client.list_blobs(pieces[2], prefix="/".join(pieces[3:])))

        if len(blobs) > 1:
            raise ValueError(f"Found more than one candidate for {gcs_uri}!")

        return blobs[0].download_as_bytes()

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
