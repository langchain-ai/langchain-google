import base64
import email
from email import policy
from typing import Dict, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_google_community.gmail.base import GmailBaseTool
from langchain_google_community.gmail.utils import get_email_body


class SearchArgsSchema(BaseModel):
    """Input schema for `GetMessageTool`."""

    message_id: str = Field(
        ...,
        description="The unique ID of the email message, retrieved from a search.",
    )


class GmailGetMessage(GmailBaseTool):
    """Tool that gets a message by ID from Gmail."""

    name: str = "get_gmail_message"

    description: str = (
        "Use this tool to fetch an email by message ID."
        " Returns the thread ID, snippet, body, subject, and sender."
    )

    args_schema: Type[SearchArgsSchema] = SearchArgsSchema

    def _run(
        self,
        message_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Run the tool."""
        query = (
            self.api_resource.users()
            .messages()
            .get(userId="me", format="raw", id=message_id)
        )
        message_data = query.execute()
        raw_message = base64.urlsafe_b64decode(message_data["raw"])

        email_msg = email.message_from_bytes(raw_message, policy=policy.default)

        subject = email_msg["Subject"]
        sender = email_msg["From"]

        body = get_email_body(email_msg)

        return {
            "id": message_id,
            "threadId": message_data["threadId"],
            "snippet": message_data["snippet"],
            "body": body,
            "subject": subject,
            "sender": sender,
        }
