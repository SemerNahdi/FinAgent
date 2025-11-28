# tests/test_stock_tool.py
# python -m pytest tests/test_stock_tool.py -vv

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from services.tools.stock_tool import StockTool


# Helper DataFrame for mocking yfinance returns
def mock_history_df():
    return pd.DataFrame(
        {"Close": [100, 102, 101, 103, 104], "Open": [99, 101, 100, 102, 103]}
    )


# ----------------------------------------------------------------------
# TEST: get_price
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_get_price(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    price = tool.get_price("AAPL")

    assert price == 104.0


# ----------------------------------------------------------------------
# TEST: historical data
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_get_historical(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    df = tool.get_historical("MSFT", "2024-01-01", "2024-02-01")

    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns


# ----------------------------------------------------------------------
# TEST: moving average
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_compute_moving_average(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    ma = tool.compute_moving_average("TSLA", period=5)

    expected = sum([100, 102, 101, 103, 104]) / 5
    assert ma == expected


# ----------------------------------------------------------------------
# TEST: volatility
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_compute_volatility(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    vol = tool.compute_volatility("AMZN")

    assert vol > 0
    assert isinstance(vol, float)


# ----------------------------------------------------------------------
# TEST: get_summary
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_get_summary(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    summary = tool.get_summary("GOOG")

    assert summary["ticker"] == "GOOG"
    assert "current_price" in summary
    assert "5_day_moving_avg" in summary
    assert "1_month_return_pct" in summary
    assert "volatility" in summary


# ----------------------------------------------------------------------
# TEST: caching works
# ----------------------------------------------------------------------
@patch("yfinance.Ticker")
def test_fetch_caching(mock_ticker):
    mock_obj = MagicMock()
    mock_obj.history.return_value = mock_history_df()
    mock_ticker.return_value = mock_obj

    tool = StockTool()
    tool._fetch("NFLX", "1mo")
    tool._fetch("NFLX", "1mo")  # second call should use cache

    assert mock_obj.history.call_count == 1
