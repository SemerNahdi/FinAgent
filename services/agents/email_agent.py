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
