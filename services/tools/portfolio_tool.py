# ---------------------------------------------------------
#  PriceService (safe Yahoo Finance abstraction)
# ---------------------------------------------------------

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
import time
import random


class PriceService:
    """Reliable Yahoo Finance fetcher with batching, retries, caching, and warnings."""

    MAX_BATCH = 50

    # ---------------------------------------------------------
    # Retry helper
    # ---------------------------------------------------------
    @staticmethod
    def _sleep_retry(attempt: int):
        delay = min(1.5 * (2**attempt), 8)
        delay += random.uniform(0, 0.3)
        time.sleep(delay)

    # ---------------------------------------------------------
    # Download helper with retries + batching
    # ---------------------------------------------------------
    def _safe_download(self, tickers, start=None, end=None):
        """Download with retries + batching."""

        results = []
        batches = [
            tickers[i : i + self.MAX_BATCH]
            for i in range(0, len(tickers), self.MAX_BATCH)
        ]

        for batch in batches:
            raw = None
            for attempt in range(4):  # up to 4 retries
                try:
                    raw = yf.download(
                        batch,
                        period="1d" if not start else None,
                        start=start,
                        end=end,
                        progress=False,
                        auto_adjust=False,
                    )
                    break
                except Exception:
                    self._sleep_retry(attempt)

            # If download fully failed → empty DF
            if raw is None:
                results.append(pd.DataFrame())
            else:
                results.append(raw)

        if len(results) == 1:
            return results[0]

        return pd.concat(results, axis=1)

    # ---------------------------------------------------------
    # Latest prices
    # ---------------------------------------------------------
    @lru_cache(maxsize=5)
    def get_latest(self, tickers):
        raw = self._safe_download(tickers)

        if raw is None or raw.empty or "Close" not in raw:
            return {t: None for t in tickers}, {
                t: "No price data returned." for t in tickers
            }

        close = raw["Close"]
        prices, warnings = {}, {}

        for t in tickers:
            if t not in close.columns:
                prices[t] = None
                warnings[t] = "Ticker not found in Yahoo response."
                continue

            series = close[t].dropna()
            if series.empty:
                prices[t] = None
                warnings[t] = "No valid closing price."
                continue

            prices[t] = float(series.iloc[-1])

        return prices, warnings

    # ---------------------------------------------------------
    # Historical prices
    # ---------------------------------------------------------
    @lru_cache(maxsize=50)
    def get_historical(self, tickers, date: str):
        start = date
        end = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        raw = self._safe_download(tickers, start=start, end=end)

        if raw is None or raw.empty or "Close" not in raw:
            return {t: None for t in tickers}, {
                t: "No historical data returned." for t in tickers
            }

        close = raw["Close"]
        prices, warnings = {}, {}

        for t in tickers:
            if t not in close.columns:
                prices[t] = None
                warnings[t] = "No historical column for ticker."
                continue

            series = close[t].dropna()
            if series.empty:
                prices[t] = None
                warnings[t] = "No valid historical close price."
                continue

            prices[t] = float(series.iloc[-1])

        return prices, warnings


# ---------------------------------------------------------
#  PortfolioTool (updated to use PriceService)
# ---------------------------------------------------------


class PortfolioTool:
    """
    Portfolio analysis using CSVs + safe Yahoo price fetching.
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.portfolio_df = pd.read_csv(portfolio_csv)
        self.metadata_df = pd.read_csv(metadata_csv)

        # merge metadata
        self.portfolio_df = self.portfolio_df.merge(
            self.metadata_df, on="Ticker", how="left"
        )

        # new price service
        self.prices = PriceService()

    # ---------------------------------------------------------
    # BASIC STRUCTURE HELPERS
    # ---------------------------------------------------------
    def _tickers(self):
        return self.portfolio_df["Ticker"].unique().tolist()

    # ---------------------------------------------------------
    # UPDATED PRICE FETCHERS
    # ---------------------------------------------------------
    def fetch_prices(self):
        return self.prices.get_latest(tuple(self._tickers()))

    def fetch_historical_prices(self, date: str):
        return self.prices.get_historical(tuple(self._tickers()), date)

    # ---------------------------------------------------------
    # ANALYSIS HELPERS
    # ---------------------------------------------------------
    def get_current_value(self, price_lookup):
        total = 0.0
        for row in self.portfolio_df.itertuples():
            price = price_lookup.get(row.Ticker, 0)
            if price is not None:
                total += row.Quantity * price
        return total

    def get_profit_loss(self, price_lookup):
        result = {}
        for row in self.portfolio_df.itertuples():
            current = price_lookup.get(row.Ticker)
            if current is None:
                result[row.Ticker] = {
                    "quantity": row.Quantity,
                    "cost_basis": row.Cost_Basis,
                    "current_price": None,
                    "profit_loss": None,
                }
                continue

            gain = (current - row.Cost_Basis) * row.Quantity
            result[row.Ticker] = {
                "quantity": row.Quantity,
                "cost_basis": row.Cost_Basis,
                "current_price": current,
                "profit_loss": gain,
            }
        return result

    def get_sector_allocation(self):
        df = self.portfolio_df
        total_quantity = df["Quantity"].sum()
        if total_quantity == 0:
            return {}

        allocation = {}
        for row in df.itertuples():
            sector = getattr(row, "Sector", "Unknown")
            # Skip nan/None sectors
            if sector is None or (isinstance(sector, float) and sector != sector):
                sector = "Unknown"
            allocation[sector] = allocation.get(sector, 0) + row.Quantity

        return {s: round((q / total_quantity) * 100, 2) for s, q in allocation.items()}

    def top_holdings(self, n: int = 5):
        """Get top N holdings by current value"""
        prices, _ = self.fetch_prices()

        holdings = []
        for row in self.portfolio_df.itertuples():
            price = prices.get(row.Ticker)
            if price is not None:
                value = row.Quantity * price
                holdings.append(
                    {
                        "ticker": row.Ticker,
                        "quantity": row.Quantity,
                        "price": price,
                        "value": value,
                        "sector": getattr(row, "Sector", "Unknown"),
                    }
                )

        # Sort by value descending
        holdings.sort(key=lambda x: x["value"], reverse=True)

        return holdings[:n]

    def get_portfolio_summary(self):
        """Get full portfolio records"""
        return self.portfolio_df.to_dict(orient="records")

    def filter_by_purchase_date(self, start_date: str, end_date: str):
        """Filter portfolio by purchase date range"""
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        filtered = df[(df["Purchase_Date"] >= start) & (df["Purchase_Date"] <= end)]
        return filtered.to_dict(orient="records")

    def get_purchase_timeline(self):
        """Get purchases organized by date"""
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        df = df.sort_values("Purchase_Date")
        return df[["Ticker", "Quantity", "Cost_Basis", "Purchase_Date"]].to_dict(
            orient="records"
        )

    def get_price_changes(self, date: str):
        """Get price changes since a specific date"""
        current_prices, _ = self.fetch_prices()
        historical_prices, _ = self.fetch_historical_prices(date)

        changes = {}
        for ticker in self._tickers():
            current = current_prices.get(ticker)
            historical = historical_prices.get(ticker)

            if current and historical:
                pct_change = ((current - historical) / historical) * 100
                changes[ticker] = {
                    "historical_price": historical,
                    "current_price": current,
                    "change_pct": round(pct_change, 2),
                    "change_amount": round(current - historical, 2),
                }

        return changes

    # ---------------------------------------------------------
    # MAIN ANALYSIS
    # ---------------------------------------------------------
    def analyze(self, include_changes: str = None):
        tickers = self._tickers()

        prices, price_warnings = self.fetch_prices()
        total_value = self.get_current_value(prices)
        total_cost = float(
            (self.portfolio_df["Quantity"] * self.portfolio_df["Cost_Basis"]).sum()
        )

        analysis = {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_value - total_cost,
            "sector_allocation": self.get_sector_allocation(),
            "profit_loss": self.get_profit_loss(prices),
            "warnings": price_warnings,
            "portfolio": self.portfolio_df.to_dict(orient="records"),
        }

        if include_changes:
            hist, hwarn = self.fetch_historical_prices(include_changes)
            changes = {}
            for t in tickers:
                cur = prices.get(t)
                old = hist.get(t)
                if cur and old:
                    changes[t] = round((cur - old) / old * 100, 2)
                else:
                    changes[t] = None

            analysis["change_since_date"] = changes
            analysis["historical_warnings"] = hwarn

        return analysis
