# services/email/daily_snapshot_agent.py

from services.email.daily_snapshot_tool import DailySnapshotTool
from services.email.email_tool import EmailTool
from langchain.tools import tool
from typing import Optional


class DailySnapshotAgent:
    """
    Agent to generate and send daily portfolio snapshots via email.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.snapshot_tool = DailySnapshotTool(portfolio_csv, metadata_csv)
        self.email_tool = EmailTool()  # Assumes env vars are loaded for credentials

    @tool
    def send_daily_snapshot(self, to_email: str) -> str:
        """
        Build the daily snapshot and send it via email.

        Args:
            to_email (str): Recipient email address.

        Returns:
            str: Confirmation message.
        """
        try:
            # Build HTML snapshot
            body = self.snapshot_tool.build_snapshot()

            # Send email
            self.email_tool.send_email(
                to_email=to_email,
                subject="📊 Your Daily Portfolio Snapshot",
                body=body,
                html=True,
            )
            return f"Daily snapshot email sent to {to_email}"
        except Exception as e:
            return f"Failed to send daily snapshot: {str(e)}"
