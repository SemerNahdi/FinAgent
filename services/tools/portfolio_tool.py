# services/tools/portfolio_tool.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional


class PortfolioTool:
    """
    Portfolio analysis using CSVs and live prices.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.portfolio_df = pd.read_csv(portfolio_csv)
        self.metadata_df = pd.read_csv(metadata_csv)

        # Merge metadata safely
        self.portfolio_df = self.portfolio_df.merge(
            self.metadata_df, on="Ticker", how="left"
        )

    # ----------------------------------------------------------------------
    # FETCH PRICES
    # ----------------------------------------------------------------------
    def fetch_prices(self) -> Dict[str, float]:
        """Fetch latest prices from Yahoo Finance for tickers."""
        tickers = self.portfolio_df["Ticker"].unique().tolist()
        data = yf.download(tickers, period="1d", progress=False)["Close"]
        prices = {}
        for ticker in tickers:
            if ticker in data.columns:
                prices[ticker] = (
                    data[ticker].dropna().iloc[-1]
                    if not data[ticker].dropna().empty
                    else None
                )
            else:
                prices[ticker] = None
            if prices[ticker] is None:
                raise ValueError(f"No price data found for {ticker}")
        return prices

    @lru_cache(maxsize=10)
    def fetch_historical_prices(self, date: str) -> Dict[str, float]:
        """Fetch historical closing prices for all tickers on a given date."""
        tickers = self.portfolio_df["Ticker"].unique().tolist()
        end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        data = yf.download(tickers, start=date, end=end_date, progress=False)["Close"]
        prices = {}
        for ticker in tickers:
            if ticker in data.columns:
                prices[ticker] = (
                    data[ticker].dropna().iloc[-1]
                    if not data[ticker].dropna().empty
                    else None
                )
            else:
                prices[ticker] = None
        return prices

    # ----------------------------------------------------------------------
    # BASIC PORTFOLIO INFO
    # ----------------------------------------------------------------------
    def has_stock(self, ticker: str) -> bool:
        return ticker.upper() in self.portfolio_df["Ticker"].str.upper().values

    def get_quantity(self, ticker: str) -> int:
        df = self.portfolio_df
        row = df[df["Ticker"].str.upper() == ticker.upper()]
        return int(row["Quantity"].values[0]) if not row.empty else 0

    def get_purchase_info(self, ticker: str) -> dict:
        df = self.portfolio_df
        row = df[df["Ticker"].str.upper() == ticker.upper()]

        if row.empty:
            return {}

        row = row.iloc[0]
        return {
            "Cost_Basis": float(row["Cost_Basis"]),
            "Purchase_Date": row["Purchase_Date"],
            "Company": row.get("Company", ""),
            "Sector": row.get("Sector", ""),
        }

    def get_portfolio_summary(self) -> list:
        return self.portfolio_df.to_dict(orient="records")

    # ----------------------------------------------------------------------
    # ANALYSIS METHODS
    # ----------------------------------------------------------------------
    def get_current_value(self, price_lookup: Dict[str, float]) -> float:
        total = 0.0
        for row in self.portfolio_df.itertuples():
            total += row.Quantity * price_lookup.get(row.Ticker, 0.0)
        return total

    def get_profit_loss(self, price_lookup: Dict[str, float]) -> dict:
        result = {}
        for row in self.portfolio_df.itertuples():
            current = price_lookup.get(row.Ticker, 0.0)
            cost = row.Cost_Basis
            result[row.Ticker] = {
                "quantity": row.Quantity,
                "cost_basis": cost,
                "current_price": current,
                "profit_loss": (current - cost) * row.Quantity,
            }
        return result

    def get_sector_allocation(self) -> dict:
        df = self.portfolio_df
        total_quantity = df["Quantity"].sum()

        allocation = {}
        for row in df.itertuples():
            sector = getattr(row, "Sector", "Unknown")
            allocation[sector] = allocation.get(sector, 0) + row.Quantity

        return {
            s: round((q / total_quantity) * 100, 2)
            for s, q in allocation.items()
            if total_quantity > 0
        }

    def get_purchase_timeline(self) -> pd.DataFrame:
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        return df.sort_values("Purchase_Date", ascending=False)

    def top_holdings(self, n: int = 5) -> pd.DataFrame:
        return self.portfolio_df.nlargest(n, "Quantity")

    def filter_by_purchase_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        mask = (df["Purchase_Date"] >= start_date) & (df["Purchase_Date"] <= end_date)
        return df.loc[mask]

    def get_price_changes(self, reference_date: str) -> Dict[str, Optional[float]]:
        """Compute % change since reference date for each ticker."""
        current_prices = self.fetch_prices()
        ref_prices = self.fetch_historical_prices(reference_date)
        changes = {}
        for ticker, current in current_prices.items():
            ref = ref_prices.get(ticker)
            changes[ticker] = (
                round((current - ref) / ref * 100, 2) if ref and ref > 0 else None
            )
        return changes

    # ----------------------------------------------------------------------
    # PORTFOLIO ANALYSIS SUMMARY
    # ----------------------------------------------------------------------
    def analyze(self, include_changes: Optional[str] = None) -> dict:
        """
        Returns portfolio analysis:
        - total value
        - total cost
        - total profit/loss
        - sector allocation
        - per-stock P/L
        """
        prices = self.fetch_prices()

        total_value = self.get_current_value(prices)
        total_cost = float(
            (self.portfolio_df["Quantity"] * self.portfolio_df["Cost_Basis"]).sum()
        )
        total_gain_loss = total_value - total_cost
        sector_alloc = self.get_sector_allocation()
        pl_details = self.get_profit_loss(prices)

        analysis = {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_gain_loss,
            "sector_allocation": sector_alloc,
            "profit_loss_details": pl_details,
            "portfolio_records": self.get_portfolio_summary(),
        }

        if include_changes:
            analysis["change_since_date"] = self.get_price_changes(include_changes)

        return analysis


# ----------------------------------------------------------------------
# OPTIONAL: SIMPLE LIST-BASED HELPER
# ----------------------------------------------------------------------
def analyze_portfolio(portfolio_list: list) -> dict:
    """
    Lightweight version of analysis for list-based portfolios.
    """
    total_stocks = len(portfolio_list)
    total_quantity = sum(p["Quantity"] for p in portfolio_list)

    sector_allocation = {}
    for p in portfolio_list:
        sector = p.get("Sector", "Unknown")
        sector_allocation[sector] = sector_allocation.get(sector, 0) + p["Quantity"]

    return {
        "total_stocks": total_stocks,
        "total_quantity": total_quantity,
        "sector_allocation": sector_allocation,
        "portfolio_details": portfolio_list,
    }
