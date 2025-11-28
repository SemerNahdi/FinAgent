from unittest.mock import patch
from services.agents.portfolio_agent import PortfolioAgent


@patch("services.tools.portfolio_tool.PortfolioTool.analyze")
def test_agent_run(mock_analyze, tmp_path):
    mock_analyze.return_value = {"total_value": 123}

    portfolio_csv = tmp_path / "portfolio.csv"
    metadata_csv = tmp_path / "metadata.csv"

    portfolio_csv.write_text("Ticker,Quantity,Cost_Basis,Purchase_Date\nAAPL,10,100,2021-01-01")
    metadata_csv.write_text("Ticker,Company,Sector\nAAPL,Apple Inc,Tech")

    agent = PortfolioAgent(portfolio_csv, metadata_csv)
    result = agent.run()

    assert result["total_value"] == 123
    mock_analyze.assert_called_once()


@patch("services.tools.portfolio_tool.PortfolioTool.top_holdings")
def test_agent_top_holdings(mock_top, tmp_path):
    mock_top.return_value = "top5"

    p = tmp_path / "p.csv"
    m = tmp_path / "m.csv"
    p.write_text("Ticker,Quantity,Cost_Basis,Purchase_Date\nAAPL,10,100,2021-01-01")
    m.write_text("Ticker,Company,Sector\nAAPL,Apple Inc,Tech")

    agent = PortfolioAgent(p, m)
    assert agent.top_holdings() == "top5"
