# services/agents/email_agents.py

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from services.email.daily_snapshot_tool import DailySnapshotTool
from services.email.email_tool import EmailTool
from typing import Optional


class SendDailySnapshotInput(BaseModel):
    to: str = Field(description="Recipient email address.")


class EmailAgent:
    """
    Agent responsible for generating and sending portfolio snapshot emails.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.snapshot_tool = DailySnapshotTool(portfolio_csv, metadata_csv)
        self.email_service = EmailTool()  # Assumes env vars are loaded for credentials

        self.send_daily_snapshot = StructuredTool.from_function(
            func=self._send_daily_snapshot,
            name="send_daily_snapshot",
            description="Send a daily portfolio snapshot email with total value, P/L per ticker, biggest mover, sector allocation, and daily change.",
            args_schema=SendDailySnapshotInput,
        )

    def _send_daily_snapshot(self, to: str) -> str:
        """
        Internal method to send the daily snapshot.
        """
        try:
            # Build HTML snapshot using the snapshot tool
            body = self.snapshot_tool.build_snapshot()

            # Send email
            self.email_service.send_email(
                to_email=to,
                subject="📈 Daily Portfolio Snapshot",
                body=body,
                html=True,
            )
            return f"Daily snapshot email sent to {to}"
        except Exception as e:
            return f"Failed to send daily snapshot: {str(e)}"

    # -------------------------------
    # New run method for MCPAgent
    # -------------------------------
    def run(self, query: str) -> str:
        """
        Generic run method to satisfy MCPAgent.
        Recognizes simple commands like:
        - "send daily snapshot to someone@example.com"
        """
        query = query.lower().strip()
        email_keywords = [
            "daily snapshot",
            "send email",
            "email report",
            "notify",
            "send me",
            "daily report",
            "mail me",
        ]

        if any(key in query for key in email_keywords):
            # Extract email address from query if present
            import re

            match = re.search(r"to\s+([\w\.\-]+@[\w\.\-]+)", query)
            to_email = match.group(1) if match else "semernahdi25@gmail.com"
            return self._send_daily_snapshot(to_email)

        return f"EmailAgent: unrecognized query '{query}'"
