"""Search an event in Google Calendar."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type
from zoneinfo import ZoneInfo

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.calendar.base import CalendarBaseTool


class SearchEventsSchema(BaseModel):
    """Input schema for `CalendarSearchEvents`."""

    calendars_info: str = Field(
        ...,
        description=(
            "A list with the information about all Calendars in Google Calendar"
            "Use the tool 'get_calendars_info' to get it."
        ),
    )

    min_datetime: str = Field(
        ...,
        description=(
            "The start datetime for the events in 'YYYY-MM-DD HH:MM:SS' format. "
            "If you do not know the current datetime, use the tool to get it."
        ),
    )

    max_datetime: str = Field(
        ..., description="The end datetime for the events search."
    )

    max_results: int = Field(
        default=10, description="The maximum number of results to return."
    )

    single_events: bool = Field(
        default=True,
        description=(
            "Whether to expand recurring events into instances and only return single "
            "one-off events and instances of recurring events."
            "'startTime' or 'updated'."
        ),
    )

    order_by: str = Field(
        default="startTime",
        description="The order of the events, either 'startTime' or 'updated'.",
    )

    query: Optional[str] = Field(
        default=None,
        description=(
            "Free text search terms to find events, "
            "that match these terms in the following fields: "
            "summary, description, location, attendee's displayName, attendee's email, "
            "organizer's displayName, organizer's email."
        ),
    )


class CalendarSearchEvents(CalendarBaseTool):  # type: ignore[override, override]
    """Tool that retrieves events from Google Calendar."""

    name: str = "search_events"

    description: str = "Use this tool to search events in the calendar."

    args_schema: Type[SearchEventsSchema] = SearchEventsSchema

    def _get_calendar_timezone(
        self, calendars_info: List[Dict[str, str]], calendar_id: str
    ) -> Optional[str]:
        """Get the timezone of the current calendar."""
        for cal in calendars_info:
            if cal["id"] == calendar_id:
                return cal.get("timeZone")
        return None

    def _get_calendar_ids(self, calendars_info: List[Dict[str, str]]) -> List[str]:
        """Get the calendar IDs."""
        return [cal["id"] for cal in calendars_info]

    def _process_data_events(
        self, events_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Optional[str]]]:
        """Process the data events."""
        simplified_data = []
        for data in events_data:
            event_dict = {
                "id": data.get("id"),
                "htmlLink": data.get("htmlLink"),
                "summary": data.get("summary"),
                "creator": data.get("creator", {}).get("email"),
                "organizer": data.get("organizer", {}).get("email"),
                "start": data.get("start", {}).get("dateTime")
                or data.get("start", {}).get("date"),
                "end": data.get("end", {}).get("dateTime")
                or data.get("end", {}).get("date"),
            }
            simplified_data.append(event_dict)
        return simplified_data

    def _run(
        self,
        calendars_info: str,
        min_datetime: str,
        max_datetime: str,
        max_results: int = 10,
        single_events: bool = True,
        order_by: str = "startTime",
        query: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Optional[str]]]:
        """Run the tool to search events in Google Calendar."""
        try:
            calendars_data = json.loads(calendars_info)
            calendars = self._get_calendar_ids(calendars_data)
            events = []
            for calendar in calendars:
                tz_name = self._get_calendar_timezone(calendars_data, calendar)
                calendar_tz = ZoneInfo(tz_name) if tz_name else None
                time_min = (
                    datetime.strptime(min_datetime, "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=calendar_tz)
                    .isoformat()
                )
                time_max = (
                    datetime.strptime(max_datetime, "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=calendar_tz)
                    .isoformat()
                )
                events_result = (
                    self.api_resource.events()
                    .list(
                        calendarId=calendar,
                        timeMin=time_min,
                        timeMax=time_max,
                        maxResults=max_results,
                        singleEvents=single_events,
                        orderBy=order_by,
                        q=query,
                    )
                    .execute()
                )
                cal_events = events_result.get("items", [])
                events.extend(cal_events)
            return self._process_data_events(events)
        except Exception as error:
            raise Exception(
                f"An error occurred while fetching events: {error}"
            ) from error
