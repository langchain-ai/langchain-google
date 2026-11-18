"""Tests for the _image_utils module."""

import base64
from pathlib import Path
from threading import get_ident
from unittest.mock import Mock, patch

import pytest
import requests
from google.genai.types import Blob, Part

from langchain_google_genai._image_utils import ImageBytesLoader, Route

BASE64_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
    "FcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestImageBytesLoader:
    """Tests for ImageBytesLoader class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.loader = ImageBytesLoader()

    def test_route_gcs_uri(self) -> None:
        """Test that GCS URIs are routed correctly."""
        assert self.loader._route("gs://bucket/blob") == Route.GCS_URI
        assert self.loader._route("gs://my-bucket/path/to/image.png") == Route.GCS_URI
        gcs_uri_with_spaces = "gs://bucket/path/with spaces/file.jpg"
        assert self.loader._route(gcs_uri_with_spaces) == Route.GCS_URI

    def test_route_http_url(self) -> None:
        """Test that HTTP URLs are routed correctly."""
        assert self.loader._route("https://example.com/image.png") == Route.URL
        assert self.loader._route("http://example.com/image.png") == Route.URL

    def test_route_base64(self) -> None:
        """Test that base64 data URIs are routed correctly."""
        assert self.loader._route("data:image/png;base64,abc123") == Route.BASE64
        assert self.loader._route("data:application/pdf;base64,xyz") == Route.BASE64

    def test_load_part_gcs_uri(self) -> None:
        """Test that load_part returns Part with file_data for GCS URIs."""
        part = self.loader.load_part("gs://bucket/image.png")

        assert part.file_data is not None
        assert part.file_data.file_uri == "gs://bucket/image.png"
        assert part.file_data.mime_type == "image/png"
        assert part.inline_data is None

    def test_load_part_gcs_uri_with_jpeg(self) -> None:
        """Test MIME type detection for JPEG files."""
        part = self.loader.load_part("gs://bucket/photo.jpg")

        assert part.file_data is not None
        assert part.file_data.file_uri == "gs://bucket/photo.jpg"
        assert part.file_data.mime_type == "image/jpeg"

    def test_load_part_gcs_uri_with_pdf(self) -> None:
        """Test MIME type detection for PDF files."""
        part = self.loader.load_part("gs://bucket/document.pdf")

        assert part.file_data is not None
        assert part.file_data.file_uri == "gs://bucket/document.pdf"
        assert part.file_data.mime_type == "application/pdf"

    def test_load_part_gcs_uri_unknown_mime_type(self) -> None:
        """Test that unknown MIME types result in None."""
        part = self.loader.load_part("gs://bucket/file")

        assert part.file_data is not None
        assert part.file_data.file_uri == "gs://bucket/file"
        assert part.file_data.mime_type is None

    def test_load_bytes_gcs_uri_raises_error(self) -> None:
        """Test that load_bytes raises an error for GCS URIs."""
        with pytest.raises(ValueError) as exc_info:
            self.loader.load_bytes("gs://bucket/image.png")

        assert "Cannot load raw bytes from GCS URIs" in str(exc_info.value)
        assert "load_part()" in str(exc_info.value)

    def test_load_part_base64(self) -> None:
        """Test that load_part handles base64 data URIs."""
        # A minimal valid base64 PNG (1x1 transparent pixel)
        part = self.loader.load_part(BASE64_PNG)

        assert part.inline_data is not None
        assert part.inline_data.mime_type == "image/png"
        assert part.file_data is None

    @pytest.mark.parametrize(
        "image_string",
        [
            "gs://bucket/image.png",
            "gs://bucket/document.pdf",
            "gs://bucket/file",
            BASE64_PNG,
            "data:application/pdf;base64,JVBERi0xLjQK",
        ],
        ids=["gcs-image", "gcs-pdf", "gcs-unknown-mime", "base64-image", "base64-pdf"],
    )
    async def test_aload_part_matches_load_part(self, image_string: str) -> None:
        """Async loading preserves routing, MIME detection, and media content."""
        with patch("langchain_google_genai._image_utils.requests.get") as mock_get:
            expected = self.loader.load_part(image_string)
            assert await self.loader.aload_part(image_string) == expected
        mock_get.assert_not_called()

    @pytest.mark.parametrize(
        "image_url", ["https://example.com/image.png", "http://example.com/image"]
    )
    async def test_aload_part_url_runs_off_event_loop(self, image_url: str) -> None:
        """Downloads run in a worker and preserve URL/content-based MIME detection."""
        event_loop_thread = get_ident()
        image_bytes = base64.b64decode(BASE64_PNG.split(",", 1)[1])

        def download_image(url: str) -> Mock:
            assert url == image_url
            assert get_ident() != event_loop_thread
            return Mock(ok=True, content=image_bytes)

        with patch(
            "langchain_google_genai._image_utils.requests.get",
            side_effect=download_image,
        ) as mock_get:
            part = await self.loader.aload_part(image_url)

        mock_get.assert_called_once_with(image_url)
        assert part == Part(inline_data=Blob(data=image_bytes, mime_type="image/png"))

    @pytest.mark.parametrize(
        "image_string",
        ["invalid-media-input", "data:image/png,invalid", "data:image/png;base64,a"],
        ids=["invalid-route", "invalid-data-uri", "invalid-base64"],
    )
    async def test_aload_part_invalid_input_matches_load_part(
        self, image_string: str
    ) -> None:
        with pytest.raises(ValueError) as sync_error:
            self.loader.load_part(image_string)
        with pytest.raises(type(sync_error.value)) as async_error:
            await self.loader.aload_part(image_string)
        assert str(async_error.value) == str(sync_error.value)

    async def test_aload_part_rejects_local_file(self, tmp_path: Path) -> None:
        """Async loading must retain the local-file security restriction."""
        image_path = tmp_path / "image.png"
        image_path.write_bytes(base64.b64decode(BASE64_PNG.split(",", 1)[1]))
        with pytest.raises(
            ValueError, match="no longer supported for security reasons"
        ):
            await self.loader.aload_part(str(image_path))

    async def test_aload_part_url_http_error_propagates(self) -> None:
        error = requests.HTTPError("Image download failed")
        response = Mock(ok=False)
        response.raise_for_status.side_effect = error
        with patch(
            "langchain_google_genai._image_utils.requests.get", return_value=response
        ):
            with pytest.raises(requests.HTTPError) as exc_info:
                await self.loader.aload_part("https://example.com/missing.png")
        assert exc_info.value is error
        response.raise_for_status.assert_called_once_with()
