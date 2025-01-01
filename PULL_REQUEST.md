## PR Description

Add Google Maps Geocoding API integration to the community package. This integration allows users to convert addresses into geographic coordinates using the Google Maps Geocoding API. The tool is designed to be composable with other Google services and LangChain tools, making it ideal for building location-aware applications.

Key Features:
- Basic address to coordinates conversion
- Support for multiple languages
- Region and viewport biasing
- Component filtering for precise results
- Seamless integration with LangChain chains and agents

Integration Opportunities:
1. **Routes Integration**: Can be combined with Google Maps Routes API for end-to-end navigation solutions
2. **Places Integration**: Works well with Google Places API for location discovery and details
3. **Solar API**: Can provide coordinates for solar potential calculations
4. **Sequential Tool Usage**: Perfect for multi-step location-based workflows, such as:
   - Address → Coordinates → Route Planning
   - Location Search → Geocoding → Places Details
   - Address Validation → Geocoding → Solar Analysis

## Relevant issues

N/A

## Type

🆕 New Feature

## Changes

1. Added new files:
   - `langchain_google_community/geocoding.py`: Main implementation with complete type hints
   - `tests/unit_tests/test_geocoding.py`: Unit tests
   - `tests/integration_tests/test_geocoding.py`: Integration tests
   - `docs/docs/integrations/geocoding.ipynb`: Documentation and examples

2. Modified existing files:
   - Updated `langchain_google_community/__init__.py` to expose the new geocoding tools
   - Added `googlemaps` dependency to the places group in `pyproject.toml`

3. Features implemented:
   - `GoogleGeocodingAPIWrapper`: Direct API interaction
   - `GoogleGeocodingTool`: LangChain tool integration
   - Support for all major Geocoding API features:
     - Basic address geocoding
     - Component filtering
     - Viewport biasing
     - Region biasing
     - Language support

4. Type System:
   - Full type hints provided for all wrapper and tool implementations
   - Note: The `googlemaps` package doesn't provide type stubs (common for many Google API clients)
   - Added appropriate type ignore comments and documentation for third-party dependencies

## Testing

1. Unit Tests:
```bash
poetry run pytest tests/unit_tests/test_geocoding.py -v
```
All tests pass successfully.

2. Integration Tests:
```bash
poetry run pytest tests/integration_tests/test_geocoding.py -v --extended
```
All tests pass successfully with a valid Google Maps API key.

## Note

Requirements:
1. Users need a Google Maps API key
2. The Geocoding API must be enabled in their Google Cloud project
3. Billing must be enabled for the project

Example usage:
```python
from langchain_google_community.geocoding import GoogleGeocodingTool

# Basic usage
tool = GoogleGeocodingTool(api_key="your_api_key")
result = tool.run("1600 Amphitheatre Parkway, Mountain View, CA")
print(result)
# Output: Address: 1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA
#         Coordinates: 37.4224764, -122.0842499

# Integration example with other Google services
from langchain.agents import Tool
from langchain_google_community.places import GooglePlacesTool

tools = [
    Tool(
        name="geocoding",
        func=geocoding_tool.run,
        description="Convert addresses to coordinates"
    ),
    Tool(
        name="places",
        func=places_tool.run,
        description="Search for places and get details"
    )
]

# Now you can use these tools together in sequences like:
# 1. Get coordinates for an address
# 2. Search for nearby places
# 3. Get details for each place
```

## Checklist

- [x] PR Title follows convention: "community: add Google Maps Geocoding integration"
- [x] Code follows project patterns and style
- [x] Added comprehensive tests
- [x] All tests pass successfully
- [x] Documentation added
- [x] Dependencies properly managed
- [x] Error handling implemented
- [x] Type hints and docstrings added
- [x] Third-party dependency limitations documented 