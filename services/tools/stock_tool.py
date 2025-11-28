# services/tools/stock_tool.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache


class StockTool:
    """
    Finance utility tool used by financial agents.
    Provides stock price, historical data, and analytical metrics.
    """

    def __init__(self):
        pass

    # ----------------------------------------------------------------------
    # INTERNAL DATA FETCHER (WITH CACHING)
    # ----------------------------------------------------------------------
    @lru_cache(maxsize=64)
    def _fetch(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch data once and cache it.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)

            if data.empty:
                raise LookupError(
                    f"No price data available for {ticker} during {period}"
                )

            return data

        except Exception as e:
            raise ConnectionError(f"Failed to fetch data for {ticker}: {e}")

    # ----------------------------------------------------------------------
    # PUBLIC FUNCTIONS
    # ----------------------------------------------------------------------
    def get_price(self, ticker: str) -> float:
        """
        Return the most recent closing price.
        """
        data = self._fetch(ticker, period="5d")
        return float(data["Close"].iloc[-1])

    def get_historical(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Get historical data over exact date range.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start, end=end)

            if data.empty:
                raise LookupError(f"No data found for {ticker} from {start} to {end}")

            return data

        except Exception as e:
            raise ConnectionError(f"Error fetching historical data for {ticker}: {e}")

    def compute_moving_average(self, ticker: str, period: int = 5) -> float:
        """
        Compute moving average using cached data.
        """
        data = self._fetch(ticker, period=f"{period}d")
        return float(data["Close"].tail(period).mean())

    def compute_volatility(self, ticker: str) -> float:
        """
        Annualized volatility based on last month of returns.
        """
        data = self._fetch(ticker, period="1mo")
        returns = data["Close"].pct_change().dropna()

        if returns.empty:
            raise ValueError(f"Not enough data to compute volatility for {ticker}")

        return float(returns.std() * (252**0.5))

    # ----------------------------------------------------------------------
    # SUMMARY METRIC
    # ----------------------------------------------------------------------
    def get_summary(self, ticker: str) -> dict:
        """
        Rollup of core financial metrics for dashboard or MCP agent.
        """
        data = self._fetch(ticker, period="1mo")

        current_price = float(data["Close"].iloc[-1])
        ma_5 = float(data["Close"].tail(5).mean())

        # compute 1-month return
        start_price = float(data["Close"].iloc[0])
        one_month_return = ((current_price - start_price) / start_price) * 100

        volatility = self.compute_volatility(ticker)

        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "5_day_moving_avg": round(ma_5, 2),
            "1_month_return_pct": round(one_month_return, 2),
            "volatility": round(volatility, 4),
        }
