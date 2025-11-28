# services/tools/portfolio_tool.py
import pandas as pd
import yfinance as yf


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
    def fetch_prices(self) -> dict:
        """Fetch latest prices from Yahoo Finance for tickers."""
        prices = {}
        for ticker in self.portfolio_df["Ticker"]:
            stock = yf.Ticker(ticker)
            price = stock.info.get("regularMarketPrice")
            if price is None:
                raise ValueError(f"No price data found for {ticker}")
            prices[ticker] = price
        return prices

    # ----------------------------------------------------------------------
    # BASIC PORTFOLIO INFO
    # ----------------------------------------------------------------------
    def has_stock(self, ticker: str) -> bool:
        return ticker.upper() in self.portfolio_df["Ticker"].values

    def get_quantity(self, ticker: str) -> int:
        df = self.portfolio_df
        row = df[df["Ticker"] == ticker.upper()]
        return int(row["Quantity"].values[0]) if not row.empty else 0

    def get_purchase_info(self, ticker: str) -> dict:
        df = self.portfolio_df
        row = df[df["Ticker"] == ticker.upper()]

        if row.empty:
            return {}

        row = row.iloc[0]
        return {
            "Cost_Basis": float(row["Cost_Basis"]),
            "Purchase_Date": row["Purchase_Date"],
            "Company": row["Company"] if "Company" in df.columns else "",
            "Sector": row["Sector"] if "Sector" in df.columns else "",
        }

    def get_portfolio_summary(self) -> list:
        return self.portfolio_df.to_dict(orient="records")

    # ----------------------------------------------------------------------
    # ANALYSIS METHODS
    # ----------------------------------------------------------------------
    def get_current_value(self, price_lookup: dict) -> float:
        total = 0
        for row in self.portfolio_df.itertuples():
            total += row.Quantity * price_lookup.get(row.Ticker, 0)
        return total

    def get_profit_loss(self, price_lookup: dict) -> dict:
        result = {}
        for row in self.portfolio_df.itertuples():
            current = price_lookup.get(row.Ticker, 0)
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

        return {s: round((q / total_quantity) * 100, 2) for s, q in allocation.items()}

    def get_purchase_timeline(self) -> pd.DataFrame:
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        return df.sort_values("Purchase_Date", ascending=False)

    def top_holdings(self, n=5) -> pd.DataFrame:
        return self.portfolio_df.nlargest(n, "Quantity")

    def filter_by_purchase_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.portfolio_df.copy()
        df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
        mask = (df["Purchase_Date"] >= start_date) & (df["Purchase_Date"] <= end_date)
        return df.loc[mask]

    # ----------------------------------------------------------------------
    # PORTFOLIO ANALYSIS SUMMARY
    # ----------------------------------------------------------------------
    def analyze(self) -> dict:
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

        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_gain_loss": total_gain_loss,
            "sector_allocation": sector_alloc,
            "profit_loss_details": pl_details,
            "portfolio_records": self.get_portfolio_summary(),
        }


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
