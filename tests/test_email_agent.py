# tests/test_email_agent.py

import pytest
import os

from services.agents.email_agent import EmailAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
portfolio_path = os.path.join(BASE_DIR, "../data/portfolio.csv")
metadata_path = os.path.join(BASE_DIR, "../data/metadata.csv")


@pytest.mark.parametrize("recipient", ["semernahdi25@gmail.com"])
def test_send_daily_snapshot_real(recipient):
    """
    Sends a real daily snapshot email and checks success.
    """
    agent = EmailAgent(portfolio_path, metadata_path)
    result = agent.send_daily_snapshot.invoke({"to": recipient})

    assert result == f"Daily snapshot email sent to {recipient}"
