"""Transactional email over SMTP, used to deliver notifications.

Configured entirely from the environment (``SMTP_HOST``/``SMTP_PORT``/``SMTP_USERNAME``/
``SMTP_PASSWORD``/``SMTP_FROM``/``SMTP_TO``/``SMTP_STARTTLS``) so no mail dependency is
needed -- the standard library's ``smtplib`` speaks to any relay. When unconfigured,
sending logs a warning and returns ``False`` rather than raising: email is an
enhancement, not a requirement, and the notification is recorded in the app either way.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from html import escape as html_escape

from .env import env

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    """Whether an SMTP relay is configured (so email can actually be sent)."""
    return bool(os.environ.get("SMTP_HOST"))


def _from_address() -> str:
    return (
        os.environ.get("SMTP_FROM")
        or os.environ.get("SMTP_USERNAME")
        or "no-reply@localhost"
    )


def recipient_address() -> str:
    """Where notification email goes.

    ``SMTP_TO`` when set, else the address it is sent from, which is a working default
    for a relay that accepts mail addressed to its own account.
    """
    return os.environ.get("SMTP_TO") or _from_address()


def send(to: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    """Send an email; return whether it was dispatched.

    Returns ``False`` (and logs) when SMTP is unconfigured or the send fails, never
    raising: the thing being announced is never blocked on delivery.
    """
    if not is_configured():
        logger.warning("SMTP not configured; not sending email to %s (%r)", to, subject)
        return False
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ.get("SMTP_USERNAME")
    password = env("SMTP_PASSWORD")
    use_starttls = os.environ.get("SMTP_STARTTLS", "1") != "0"

    message = EmailMessage()
    message["From"] = _from_address()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body_text)
    if body_html:
        message.add_alternative(body_html, subtype="html")

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            if use_starttls:
                server.starttls(context=ssl.create_default_context())
            if username and password:
                server.login(username, password)
            server.send_message(message)
        return True
    except Exception:
        logger.exception("failed to send email to %s", to)
        return False


def notification_email(
    *, title: str, body: str, event_type: str
) -> tuple[str, str, str]:
    """Build the (subject, text, html) of a notification email.

    Every interpolated value is
    HTML-escaped (titles and bodies carry user-named monitors, models, and error
    text), and control characters are stripped from the subject to prevent header
    injection.
    """
    subject = f"[Elbi] {title}".replace("\r", " ").replace("\n", " ")
    text = (
        f"{title}\n\n"
        + (f"{body}\n\n" if body else "")
        + "Open your Elbi inbox for details and to mark this as read."
    )
    safe_title = html_escape(title)
    safe_body = html_escape(body)
    safe_type = html_escape(event_type)
    html = (
        '<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">'
        f"<p><strong>{safe_title}</strong></p>"
        + (f"<p>{safe_body}</p>" if body else "")
        + f'<p style="color:#666;font-size:13px">Event type: {safe_type}</p>'
        '<p style="color:#666;font-size:14px">Open your Elbi inbox for '
        "details and to mark this as read.</p>"
        "</div>"
    )
    return subject, text, html
