import pandas as pd
import pytest
from unittest.mock import patch
from services.tools.portfolio_tool import PortfolioTool


# python -m pytest tests/test_portfolio_agent.py -vv
@pytest.fixture
def sample_portfolio_csv(tmp_path):
    p = tmp_path / "portfolio.csv"
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT"],
            "Quantity": [10, 5],
            "Cost_Basis": [100.0, 200.0],
            "Purchase_Date": ["2021-01-01", "2022-01-01"],
        }
    )
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def sample_metadata_csv(tmp_path):
    p = tmp_path / "metadata.csv"
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT"],
            "Company": ["Apple Inc", "Microsoft Corp"],
            "Sector": ["Tech", "Tech"],
        }
    )
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def tool(sample_portfolio_csv, sample_metadata_csv):
    return PortfolioTool(sample_portfolio_csv, sample_metadata_csv)


def test_has_stock(tool):
    assert tool.has_stock("AAPL") is True
    assert tool.has_stock("TSLA") is False


def test_get_quantity(tool):
    assert tool.get_quantity("AAPL") == 10
    assert tool.get_quantity("MSFT") == 5
    assert tool.get_quantity("TSLA") == 0


def test_get_purchase_info(tool):
    info = tool.get_purchase_info("AAPL")
    assert info["Company"] == "Apple Inc"
    assert info["Sector"] == "Tech"
    assert info["Cost_Basis"] == 100.0


def test_get_sector_allocation(tool):
    alloc = tool.get_sector_allocation()
    assert alloc["Tech"] == 100.0  # Both stocks are tech


def test_filter_by_purchase_date(tool):
    df = tool.filter_by_purchase_date("2021-01-01", "2021-12-31")
    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "AAPL"


@patch("services.tools.portfolio_tool.yf.Ticker")
def test_fetch_prices(mock_yf, tool):
    mock_yf.return_value.info = {"regularMarketPrice": 150}
    prices = tool.fetch_prices()
    assert prices["AAPL"] == 150
    assert prices["MSFT"] == 150


@patch("services.tools.portfolio_tool.PortfolioTool.fetch_prices")
def test_analyze(mock_fetch, tool):
    mock_fetch.return_value = {"AAPL": 150, "MSFT": 250}

    result = tool.analyze()

    assert result["total_cost"] == (10 * 100 + 5 * 200)
    assert result["total_value"] == (10 * 150 + 5 * 250)
    assert "sector_allocation" in result
    assert "profit_loss_details" in result
