"""Test Google Maps Geocoding API wrapper."""
from unittest.mock import MagicMock, patch

from langchain_google_community.geocoding import (
    GoogleGeocodingAPIWrapper,
    GoogleGeocodingTool,
)


@patch("googlemaps.Client")
def test_geocoding_wrapper_initialization(mock_client):
    """Test that the wrapper can be initialized with an API key."""
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance

    wrapper = GoogleGeocodingAPIWrapper(api_key="test_key")
    assert wrapper.api_key == "test_key"
    assert wrapper.client == mock_instance


@patch("googlemaps.Client")
def test_geocoding_wrapper_geocode(mock_client):
    """Test that the wrapper can geocode an address."""
    # Mock response data
    mock_response = [
        {
            "formatted_address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
            "geometry": {"location": {"lat": 37.4224764, "lng": -122.0842499}},
        }
    ]

    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.geocode.return_value = mock_response
    mock_client.return_value = mock_instance

    # Create wrapper and test geocoding
    wrapper = GoogleGeocodingAPIWrapper(api_key="test_key")
    result = wrapper.geocode("Google HQ")

    # Verify the result
    assert isinstance(result, str)
    assert "1600 Amphitheatre Pkwy" in result
    assert "37.4224764" in result
    assert "-122.0842499" in result


@patch("googlemaps.Client")
def test_geocoding_tool_initialization(mock_client):
    """Test that the tool can be initialized."""
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance

    tool = GoogleGeocodingTool(api_key="test_key")
    assert tool.name == "google_geocoding"
    assert "geocoding" in tool.description.lower()
    assert tool.api_wrapper.api_key == "test_key"
    assert tool.api_wrapper.client == mock_instance


@patch("googlemaps.Client")
def test_geocoding_tool_run(mock_client):
    """Test that the tool can run a geocoding query."""
    # Mock response data
    mock_response = [
        {
            "formatted_address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA",
            "geometry": {"location": {"lat": 37.4224764, "lng": -122.0842499}},
        }
    ]

    # Set up the mock
    mock_instance = MagicMock()
    mock_instance.geocode.return_value = mock_response
    mock_client.return_value = mock_instance

    # Create tool and test geocoding
    tool = GoogleGeocodingTool(api_key="test_key")
    result = tool.run("Google HQ")

    # Verify the result
    assert isinstance(result, str)
    assert "1600 Amphitheatre Pkwy" in result
    assert "37.4224764" in result
    assert "-122.0842499" in result
