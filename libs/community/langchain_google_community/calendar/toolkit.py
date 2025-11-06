from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_google_community.calendar.create_event import CalendarCreateEvent
from langchain_google_community.calendar.current_datetime import GetCurrentDatetime
from langchain_google_community.calendar.delete_event import CalendarDeleteEvent
from langchain_google_community.calendar.get_calendars_info import GetCalendarsInfo
from langchain_google_community.calendar.move_event import CalendarMoveEvent
from langchain_google_community.calendar.search_events import CalendarSearchEvents
from langchain_google_community.calendar.update_event import CalendarUpdateEvent
from langchain_google_community.calendar.utils import build_calendar_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarToolkit(BaseToolkit):
    """Toolkit for interacting with Google Calendar.

    Provides tools for calendar operations including creating, searching,
    updating, moving, and deleting events.


    !!! warning "Security"
        This toolkit contains tools that can read and modify the state of a
        service. For example, it can create, update, and delete calendar events
        on behalf of the associated account.

        See [Security Policy](https://docs.langchain.com/oss/python/security-policy)
        for more information.
    """

    api_resource: Resource = Field(default_factory=build_calendar_service)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            CalendarCreateEvent(api_resource=self.api_resource),
            CalendarSearchEvents(api_resource=self.api_resource),
            CalendarUpdateEvent(api_resource=self.api_resource),
            GetCalendarsInfo(api_resource=self.api_resource),
            CalendarMoveEvent(api_resource=self.api_resource),
            CalendarDeleteEvent(api_resource=self.api_resource),
            GetCurrentDatetime(api_resource=self.api_resource),
        ]
