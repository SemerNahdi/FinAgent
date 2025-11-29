# services/email/daily_snapshot_tool.py

from services.tools.portfolio_tool import PortfolioTool
from services.email.html_templates import snapshot_html_template
from datetime import datetime, timedelta


class DailySnapshotTool:
    """
    Builds daily HTML snapshot reports.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.portfolio_tool = PortfolioTool(portfolio_csv, metadata_csv)

    def build_snapshot(self) -> str:
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        analysis = self.portfolio_tool.analyze(include_changes=yesterday)
        return snapshot_html_template(analysis)
