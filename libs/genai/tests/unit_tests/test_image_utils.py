"""Tests for the _image_utils module."""

import pytest

from langchain_google_genai._image_utils import ImageBytesLoader, Route


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
        base64_png = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAf"
            "FcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        part = self.loader.load_part(base64_png)

        assert part.inline_data is not None
        assert part.inline_data.mime_type == "image/png"
        assert part.file_data is None

    def test_load_part_url_as_file_uri(self) -> None:
        """Test that load_part returns Part with file_data for URLs (default behavior)."""
        url = "https://example.com/image.png"
        part = self.loader.load_part(url)

        assert part.file_data is not None
        assert part.file_data.file_uri == url
        assert part.file_data.mime_type == "image/png"
        assert part.inline_data is None

    def test_load_part_url_as_file_uri_with_signed_url(self) -> None:
        """Test that load_part handles signed URLs as file_uri."""
        signed_url = (
            "https://storage.googleapis.com/bucket/file.pdf?"
            "X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=..."
        )
        part = self.loader.load_part(signed_url)

        assert part.file_data is not None
        assert part.file_data.file_uri == signed_url
        assert part.file_data.mime_type == "application/pdf"
        assert part.inline_data is None

    def test_load_part_url_backward_compatibility(self) -> None:
        """Test that load_part can download URLs for backward compatibility."""
        url = "https://example.com/image.png"
        # Mock the _bytes_from_url method to avoid actual HTTP request
        original_method = self.loader._bytes_from_url
        self.loader._bytes_from_url = lambda u: b"fake_image_data"

        try:
            part = self.loader.load_part(url, use_file_uri_for_urls=False)
            assert part.inline_data is not None
            assert part.inline_data.data == b"fake_image_data"
            assert part.file_data is None
        finally:
            self.loader._bytes_from_url = original_method