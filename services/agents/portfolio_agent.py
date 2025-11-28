# services/agents/portfolio_agent.py

from services.tools.portfolio_tool import PortfolioTool


class PortfolioAgent:
    """
    Handles portfolio analysis queries through PortfolioTool.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.tool = PortfolioTool(portfolio_csv, metadata_csv)

    def run(self, query: str = None) -> dict:
        """
        Runs the agent. Currently returns a full analysis.
        The query parameter is kept for future extensions.
        """
        return self.tool.analyze()

    # Exposed helper methods
    def top_holdings(self, n=5):
        return self.tool.top_holdings(n)

    def sector_allocation(self):
        return self.tool.get_sector_allocation()

    def filter_by_date(self, start_date, end_date):
        return self.tool.filter_by_purchase_date(start_date, end_date)
