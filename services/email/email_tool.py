# services/email/email_tool.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()  # load env variables from .env


class EmailTool:
    """
    Basic email sending tool (supports plain text and HTML).
    """

    def __init__(self):
        self.smtp_host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("EMAIL_PORT", 587))
        self.username = os.getenv("EMAIL_USER")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.sender_name = os.getenv("EMAIL_SENDER_NAME", self.username)

    def send_email(self, to_email: str, subject: str, body: str, html=False):
        """Send an email."""

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{self.sender_name} <{self.username}>"
        msg["To"] = to_email
        msg["Subject"] = subject

        if html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.username, to_email, msg.as_string())

        return "Email sent successfully"
