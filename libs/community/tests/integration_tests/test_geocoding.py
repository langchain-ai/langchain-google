"""Integration tests for Google Maps Geocoding API wrapper and tool."""

import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

from langchain_google_community.geocoding import (
    GoogleGeocodingAPIWrapper,
    GoogleGeocodingTool,
)


def load_api_key():
    """Load API key from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_MAPS_API_KEY environment variable not set")
    return api_key


@pytest.mark.extended
def test_geocoding_wrapper_live():
    """Test that the wrapper can make actual API calls."""
    api_key = load_api_key()

    wrapper = GoogleGeocodingAPIWrapper(api_key=api_key)
    result = wrapper.geocode("1600 Amphitheatre Parkway, Mountain View, CA")

    assert isinstance(result, str)
    assert "1600 Amphitheatre" in result
    assert "Mountain View" in result
    assert "37." in result  # Latitude should start with 37
    assert "-122." in result  # Longitude should start with -122


@pytest.mark.extended
def test_geocoding_tool_live():
    """Test that the tool can make actual API calls."""
    api_key = load_api_key()

    tool = GoogleGeocodingTool(api_key=api_key)
    result = tool._run("1600 Amphitheatre Parkway, Mountain View, CA")

    assert isinstance(result, str)
    assert "1600 Amphitheatre" in result
    assert "Mountain View" in result
    assert "37." in result  # Latitude should start with 37
    assert "-122." in result  # Longitude should start with -122


@pytest.mark.extended
def test_geocoding_with_components():
    """Test geocoding with component filters."""
    api_key = load_api_key()

    tool = GoogleGeocodingTool(api_key=api_key)
    result = tool._run(
        address="Mountain View",
        components={"country": "US", "administrative_area": "CA"},
    )

    assert isinstance(result, str)
    assert "Mountain View" in result
    assert "CA" in result
    assert "37." in result  # Latitude should start with 37
    assert "-122." in result  # Longitude should start with -122


@pytest.mark.extended
def test_geocoding_with_language():
    """Test geocoding with different languages."""
    api_key = load_api_key()

    tool = GoogleGeocodingTool(api_key=api_key)
    result = tool._run(address="Eiffel Tower", language="fr")

    assert isinstance(result, str)
    assert "Gustave Eiffel" in result  # Street name in French
    assert "Paris" in result
    assert "48." in result  # Latitude should start with 48
    assert "2." in result  # Longitude should start with 2
