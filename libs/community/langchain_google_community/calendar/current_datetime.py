"""Get the current datetime according to the calendar timezone."""

from datetime import datetime
from typing import Optional, Type
from zoneinfo import ZoneInfo

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.calendar.base import CalendarBaseTool


class CurrentDatetimeSchema(BaseModel):
    """Input schema for `GetCurrentDatetime`."""

    calendar_id: Optional[str] = Field(
        default="primary", description="The calendar ID. Defaults to 'primary'."
    )


class GetCurrentDatetime(CalendarBaseTool):  # type: ignore[override, override]
    """Tool that gets the current datetime according to the calendar timezone."""

    name: str = "get_current_datetime"

    description: str = (
        "Use this tool to get the current datetime according to the calendar timezone."
        "The output datetime format is 'YYYY-MM-DD HH:MM:SS'"
    )

    args_schema: Type[CurrentDatetimeSchema] = CurrentDatetimeSchema

    def get_timezone(self, calendar_id: Optional[str]) -> str:
        """Get the timezone of the specified calendar."""
        calendars = self.api_resource.calendarList().list().execute().get("items", [])
        if not calendars:
            raise ValueError("No calendars found.")
        if calendar_id == "primary":
            return calendars[0]["timeZone"]
        else:
            for item in calendars:
                if item["id"] == calendar_id and item["accessRole"] != "reader":
                    return item["timeZone"]
            raise ValueError(f"Timezone not found for calendar ID: {calendar_id}")

    def _run(
        self,
        calendar_id: Optional[str] = "primary",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool to create an event in Google Calendar."""
        try:
            timezone = self.get_timezone(calendar_id)
            date_time = datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S")
            return f"Time zone: {timezone}, Date and time: {date_time}"
        except Exception as error:
            raise Exception(f"An error occurred: {error}") from error
