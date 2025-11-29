# services/agents/stock_agent.py

from services.tools.stock_tool import StockTool
import re


class StockAgent:
    """
    Agent that handles stock-related queries by using StockTool.
    """

    def __init__(self, stock_tool: StockTool):
        self.tool = stock_tool

    def handle_query(self, query: str):
        query = query.lower().strip()

        # Price query
        price_match = re.match(r"(price of|current price of)\s+([A-Za-z]+)", query)
        if price_match:
            ticker = price_match.group(2).upper()
            price = self.tool.get_price(ticker)
            return {"ticker": ticker, "current_price": round(price, 2)}

        # Moving average query
        ma_match = re.match(r"(\d+)[- ]?day ma of\s+([A-Za-z]+)", query)
        if ma_match:
            period = int(ma_match.group(1))
            ticker = ma_match.group(2).upper()
            ma = self.tool.compute_moving_average(ticker, period)
            return {"ticker": ticker, f"{period}_day_ma": round(ma, 2)}

        # Summary query
        summary_match = re.match(r"(summary of|stock summary for)\s+([A-Za-z]+)", query)
        if summary_match:
            ticker = summary_match.group(2).upper()
            return self.tool.get_summary(ticker)

        return {
            "error": "Query not recognized. Try: 'price of TICKER', '5-day MA of TICKER', or 'summary of TICKER'"
        }

    # Add this method so MCPAgent can call it
    def run(self, query: str):
        return self.handle_query(query)
