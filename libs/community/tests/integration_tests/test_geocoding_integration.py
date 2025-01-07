"""Integration tests for Google Geocoding API."""

import os

import pytest
from pydantic import SecretStr

from langchain_google_community import GoogleGeocodingAPIWrapper, GoogleGeocodingTool


@pytest.fixture
def api_wrapper() -> GoogleGeocodingAPIWrapper:
    """Create API wrapper with credentials from environment."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_MAPS_API_KEY environment variable not set")
    return GoogleGeocodingAPIWrapper(google_api_key=SecretStr(api_key))


@pytest.fixture
def geocoding_tool(api_wrapper: GoogleGeocodingAPIWrapper) -> GoogleGeocodingTool:
    """Create geocoding tool with the API wrapper."""
    return GoogleGeocodingTool(
        api_wrapper=api_wrapper,
        max_results=5,
        include_bounds=True,
        include_metadata=True,
    )


@pytest.mark.asyncio
async def test_geocode_async(api_wrapper: GoogleGeocodingAPIWrapper) -> None:
    """Test async geocoding functionality."""
    result = await api_wrapper.geocode_async("Statue of Liberty, New York")

    assert result["status"] == "OK"
    assert len(result["results"]) > 0

    location = result["results"][0]
    assert "address" in location
    assert "New York" in location["address"]["full"]

    # Check coordinates are roughly correct for Statue of Liberty
    lat = location["geometry"]["location"]["lat"]
    lng = location["geometry"]["location"]["lng"]
    assert 40.68 < lat < 40.69  # Statue of Liberty latitude
    assert -74.05 < lng < -74.04  # Statue of Liberty longitude


def test_geocode_batch(api_wrapper: GoogleGeocodingAPIWrapper) -> None:
    """Test batch geocoding with multiple locations."""
    locations = ["Times Square, New York", "Big Ben, London", "Sydney Opera House"]

    result = api_wrapper.batch_geocode(locations)
    assert result["status"] == "OK"
    assert len(result["results"]) == len(locations)

    # Verify each location has valid data
    for location in result["results"]:
        assert "address" in location
        assert "full" in location["address"]
        assert "geometry" in location
        assert "location" in location["geometry"]
        assert "lat" in location["geometry"]["location"]
        assert "lng" in location["geometry"]["location"]


def test_geocode_error_handling(api_wrapper: GoogleGeocodingAPIWrapper) -> None:
    """Test error handling with invalid queries."""
    result = api_wrapper.raw_results("NonexistentPlace12345!@#$%")

    assert result["status"] == "ZERO_RESULTS"
    assert len(result["results"]) == 0


def test_tool_batch_query(geocoding_tool: GoogleGeocodingTool) -> None:
    """Test geocoding tool with batch query."""
    query = "Times Square, Central Park, Empire State Building"
    result, metadata = geocoding_tool._run(query)

    assert len(result) == 3
    assert metadata["status"] == "OK"
    assert all("geometry" in loc for loc in result)
