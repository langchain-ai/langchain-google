from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field

from langchain_google_community.gmail.create_draft import GmailCreateDraft
from langchain_google_community.gmail.get_message import GmailGetMessage
from langchain_google_community.gmail.get_thread import GmailGetThread
from langchain_google_community.gmail.search import GmailSearch
from langchain_google_community.gmail.send_message import GmailSendMessage
from langchain_google_community.gmail.utils import build_gmail_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource  # type: ignore[import]
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


SCOPES = ["https://mail.google.com/"]


class GmailToolkit(BaseToolkit):
    """Toolkit for interacting with Gmail.

    Provides tools for Gmail operations including creating drafts, sending messages,
    searching emails, and retrieving messages and threads.

    !!! warning "Security"

        This toolkit contains tools that can read and modify the state of a
        service. For example, it can send emails and read messages on behalf
        of the associated account.

        See [Security Best Practices](https://docs.langchain.com/oss/python/security-policy)
        for more information.
    """

    api_resource: Resource = Field(default_factory=build_gmail_service)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            GmailCreateDraft(api_resource=self.api_resource),
            GmailSendMessage(api_resource=self.api_resource),
            GmailSearch(api_resource=self.api_resource),
            GmailGetMessage(api_resource=self.api_resource),
            GmailGetThread(api_resource=self.api_resource),
        ]
