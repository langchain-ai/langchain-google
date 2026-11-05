"""Test how the Gmail tools read the body and headers of a message."""

import base64
from email.message import EmailMessage
from typing import Any, Dict, Union
from unittest.mock import MagicMock

import pytest

from langchain_google_community.gmail.get_message import GmailGetMessage
from langchain_google_community.gmail.search import GmailSearch

TOOLS = pytest.mark.parametrize("tool", ["search", "get_message"])


def _read(tool: str, message: Union[EmailMessage, bytes]) -> Dict[str, Any]:
    """Read a message with a tool, from a Gmail API that returns it raw."""
    data = message if isinstance(message, bytes) else message.as_bytes()
    api = MagicMock()
    messages = api.users.return_value.messages.return_value
    messages.list.return_value.execute.return_value = {"messages": [{"id": "m1"}]}
    messages.get.return_value.execute.return_value = {
        "raw": base64.urlsafe_b64encode(data).decode(),
        "threadId": "t1",
        "snippet": "",
    }
    # bypass pydantic validation as google-api-python-client is not a package dependency
    if tool == "search":
        search = GmailSearch.model_construct(api_resource=api)
        return search.run({"query": "in:inbox"})[0]
    get_message = GmailGetMessage.model_construct(api_resource=api)
    return get_message.run({"message_id": "m1"})


def _message(subject: str = "Hello") -> EmailMessage:
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["To"] = "me@example.com"
    message["Subject"] = subject
    return message


@TOOLS
def test_reads_an_html_body_that_comes_with_an_attachment(tool: str) -> None:
    """Test an HTML email with a file attached, and no plain text part."""
    message = _message("Invoice")
    message.set_content("<p>Your invoice for September.</p>", subtype="html")
    message.add_attachment(
        b"%PDF-1.4", maintype="application", subtype="pdf", filename="invoice.pdf"
    )

    assert "Your invoice for September." in _read(tool, message)["body"]


@TOOLS
def test_reads_an_html_body_with_inline_images(tool: str) -> None:
    """Test an HTML email with an image it shows, and no plain text part."""
    message = _message("Shipped")
    message.set_content('<p>Your order has shipped.</p><img src="cid:logo">', "html")
    message.add_related(b"\x89PNG", maintype="image", subtype="png", cid="<logo>")

    assert "Your order has shipped." in _read(tool, message)["body"]


@TOOLS
def test_decodes_a_body_in_its_declared_charset(tool: str) -> None:
    """Test a Latin-1 body, which is not valid UTF-8."""
    message = (
        b"From: sender@example.com\n"
        b"Subject: Hello\n"
        b"Content-Type: text/plain; charset=iso-8859-1\n"
        b"Content-Transfer-Encoding: 8bit\n"
        b"\n" + "Grüße aus Reutlingen".encode("latin-1") + b"\n"
    )

    assert _read(tool, message)["body"].strip() == "Grüße aus Reutlingen"


@TOOLS
def test_reads_a_body_in_a_charset_python_does_not_know(tool: str) -> None:
    """Test a body whose declared charset has no Python codec."""
    message = (
        b"From: sender@example.com\n"
        b"Subject: Hello\n"
        b"Content-Type: text/plain; charset=x-unknown\n"
        b"Content-Transfer-Encoding: 8bit\n"
        b"\n" + "Grüße aus Reutlingen".encode() + b"\n"
    )

    assert _read(tool, message)["body"].strip() == "Grüße aus Reutlingen"


@TOOLS
def test_decodes_an_encoded_subject_and_sender(tool: str) -> None:
    """Test headers that carry non-ASCII text as RFC 2047 encoded words."""
    message = _message("Grüße")
    message.replace_header("From", "Jürgen <juergen@example.com>")
    message.set_content("Hello.")
    assert b"=?utf-8?" in message.as_bytes()

    result = _read(tool, message)

    assert result["subject"] == "Grüße"
    assert result["sender"] == "Jürgen <juergen@example.com>"


@TOOLS
def test_reads_an_html_body_as_text(tool: str) -> None:
    """Test that the markup of an HTML body is removed."""
    pytest.importorskip("bs4")
    message = _message("Invoice")
    message.set_content("<p>Your <b>invoice</b> for September.</p>", subtype="html")

    assert _read(tool, message)["body"].strip() == "Your invoice for September."


@TOOLS
def test_keeps_angle_brackets_in_a_plain_body(tool: str) -> None:
    """Test that plain text is not read as HTML."""
    pytest.importorskip("bs4")
    message = _message("Contact")
    message.set_content("Write to <jane@example.com> if 3 < 4.")

    assert _read(tool, message)["body"].strip() == (
        "Write to <jane@example.com> if 3 < 4."
    )


@TOOLS
def test_prefers_the_plain_text_part(tool: str) -> None:
    """Test an email with both a plain text and an HTML version."""
    message = _message()
    message.set_content("The plain version.")
    message.add_alternative("<p>The HTML version.</p>", subtype="html")

    assert _read(tool, message)["body"].strip() == "The plain version."


@TOOLS
def test_an_email_without_a_text_body_has_an_empty_body(tool: str) -> None:
    """Test an email that only carries a file."""
    message = _message("Scan")
    message.add_attachment(
        b"%PDF-1.4", maintype="application", subtype="pdf", filename="scan.pdf"
    )

    assert _read(tool, message)["body"] == ""
