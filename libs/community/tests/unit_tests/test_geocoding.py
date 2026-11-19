from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_google_community.geocoding import (
    GoogleGeocodingAPIWrapper,
    GoogleGeocodingTool,
)


def _api_response(address: str) -> Dict[str, Any]:
    return {
        "status": "OK",
        "results": [
            {
                "formatted_address": f"<{address}>",
                "geometry": {"location": {"lat": 1.0, "lng": 2.0}},
                "place_id": address,
                "types": ["point_of_interest"],
            }
        ],
    }


class _FakeAiohttpResponse:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    async def json(self) -> Dict[str, Any]:
        return self._data

    async def __aenter__(self) -> "_FakeAiohttpResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeAiohttpSession:
    def __init__(self, calls: List[str]) -> None:
        self._calls = calls

    def get(self, url: str, params: Dict[str, str], **kwargs: Any) -> Any:
        self._calls.append(params["address"])
        return _FakeAiohttpResponse(_api_response(params["address"]))

    async def __aenter__(self) -> "_FakeAiohttpSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


@pytest.fixture
def api_calls() -> Iterator[Dict[str, List[str]]]:
    """Fake the Geocoding API and record each request by HTTP client."""
    calls: Dict[str, List[str]] = {"requests": [], "aiohttp": []}

    def fake_get(url: str, params: Dict[str, str], **kwargs: Any) -> MagicMock:
        calls["requests"].append(params["address"])
        response = MagicMock()
        response.json.return_value = _api_response(params["address"])
        return response

    with (
        patch(
            "langchain_google_community.geocoding.requests.get", side_effect=fake_get
        ),
        patch(
            "langchain_google_community.geocoding.aiohttp.ClientSession",
            side_effect=lambda *a, **k: _FakeAiohttpSession(calls["aiohttp"]),
        ),
    ):
        yield calls


@pytest.fixture
def wrapper() -> GoogleGeocodingAPIWrapper:
    return GoogleGeocodingAPIWrapper(google_api_key=SecretStr("fake"))


async def test_geocode_async_makes_a_single_async_request(
    wrapper: GoogleGeocodingAPIWrapper, api_calls: Dict[str, List[str]]
) -> None:
    result = await wrapper.geocode_async("Eiffel Tower")

    assert api_calls == {"requests": [], "aiohttp": ["Eiffel Tower"]}
    assert result["status"] == "OK"
    assert result["results"] == wrapper.clean_results(
        _api_response("Eiffel Tower")["results"]
    )


@pytest.mark.parametrize(
    "query", ["Eiffel Tower", "Eiffel Tower, Times Square, Central Park"]
)
async def test_arun_geocodes_each_location_like_run(
    wrapper: GoogleGeocodingAPIWrapper,
    api_calls: Dict[str, List[str]],
    query: str,
) -> None:
    tool = GoogleGeocodingTool(api_wrapper=wrapper, max_results=2)
    expected_locations = [loc.strip() for loc in query.split(",")][:2]

    sync_results, _ = tool._run(query)
    assert api_calls["requests"] == expected_locations
    api_calls["requests"].clear()

    async_results, async_response = await tool._arun(query)

    assert api_calls["requests"] == []
    assert sorted(api_calls["aiohttp"]) == sorted(expected_locations)
    assert async_results == sync_results
    assert async_response["status"] == "OK"
