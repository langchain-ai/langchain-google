from unittest.mock import MagicMock

from langchain_google_community.calendar.create_event import CalendarCreateEvent


def test_create_simple_event() -> None:
    """Test google calendar create event."""
    mock_api_resource = MagicMock()
    # bypass pydantic validation as google-api-python-client is not a package dependency
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_description_and_location() -> None:
    """Test google calendar create event with location."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
        "description": "Event description",
        "location": "Sante Fe, Mexico City",
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_attendees() -> None:
    """Test google calendar create event with attendees."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
        "attendees": ["fake123@email.com", "123fake@email.com"],
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_reminders() -> None:
    """Test google calendar create event with reminders."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
        "reminders": [
            {"method": "email", "minutes": 10},
            {"method": "popup", "minutes": 30},
        ],
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_recurrence() -> None:
    """Test google calendar create event with recurrence."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
        "recurrence": {
            "FREQ": "WEEKLY",
            "COUNT": 10,
        },
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_with_conference_data() -> None:
    """Test google calendar create event with conference data."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)
    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "America/Mexico_City",
        "conference_data": True,
    }
    result = tool.run(tool_input)
    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_create_event_timezone_handling() -> None:
    """Test that create_event properly handles timezone in API payload."""
    mock_api_resource = MagicMock()
    tool = CalendarCreateEvent.model_construct(api_resource=mock_api_resource)

    tool_input = {
        "summary": "Event summary",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "Asia/Tokyo",
    }

    result = tool.run(tool_input)

    # Verify API call arguments
    call_args = mock_api_resource.events().insert.call_args
    body = call_args[1]["body"]  # keyword arguments

    # Check if timezone is correctly set
    assert body["start"]["timeZone"] == "Asia/Tokyo"
    assert body["end"]["timeZone"] == "Asia/Tokyo"

    # Check if datetime is preserved (not converted)
    # Should be 14:00 JST, not converted to different time
    assert "2025-07-11T14:00:00+09:00" == body["start"]["dateTime"]
    assert "2025-07-11T15:30:00+09:00" == body["end"]["dateTime"]

    assert tool.args_schema is not None
    assert result.startswith("Event created:")


def test_search_events_timezone_handling() -> None:
    """Test that search_events properly handles timezone in API call."""
    from langchain_google_community.calendar.search_events import CalendarSearchEvents

    mock_api_resource = MagicMock()
    tool = CalendarSearchEvents.model_construct(api_resource=mock_api_resource)

    mock_api_resource.events().list().execute.return_value = {"items": []}

    tool_input = {
        "calendars_info": '[{"id": "primary", "timeZone": "Asia/Tokyo"}]',
        "min_datetime": "2025-07-11 14:00:00",
        "max_datetime": "2025-07-11 15:30:00",
    }

    result = tool.run(tool_input)

    # Verify API call arguments
    call_args = mock_api_resource.events().list.call_args

    # Check if datetime is preserved with correct timezone
    # Should be 14:00 JST, not converted to different time
    assert call_args[1]["timeMin"] == "2025-07-11T14:00:00+09:00"
    assert call_args[1]["timeMax"] == "2025-07-11T15:30:00+09:00"

    assert tool.args_schema is not None
    assert isinstance(result, list)


def test_update_event_timezone_handling() -> None:
    """Test that update_event properly handles timezone in API payload."""
    from langchain_google_community.calendar.update_event import CalendarUpdateEvent

    mock_api_resource = MagicMock()
    tool = CalendarUpdateEvent.model_construct(api_resource=mock_api_resource)

    # Mock existing event
    mock_event = {"start": {"timeZone": "UTC"}, "end": {"timeZone": "UTC"}}
    mock_api_resource.events().get().execute.return_value = mock_event
    mock_api_resource.events().update().execute.return_value = {"htmlLink": "test"}

    tool_input = {
        "event_id": "test_event_id",
        "summary": "Updated Event",
        "start_datetime": "2025-07-11 14:00:00",
        "end_datetime": "2025-07-11 15:30:00",
        "timezone": "Europe/London",
    }

    result = tool.run(tool_input)

    # Verify API call arguments
    call_args = mock_api_resource.events().update.call_args
    body = call_args[1]["body"]

    # Check if timezone is correctly updated
    assert body["start"]["timeZone"] == "Europe/London"
    assert body["end"]["timeZone"] == "Europe/London"

    # Check if datetime is preserved (not converted)
    # Should be 14:00 London time, not converted from system timezone
    assert body["start"]["dateTime"] == "2025-07-11T14:00:00+01:00"
    assert body["end"]["dateTime"] == "2025-07-11T15:30:00+01:00"

    assert tool.args_schema is not None
    assert result.startswith("Event updated:")
