# test_mcp.py
# python -m tests.test_mcp

import asyncio
from services.agents.email_agent import EmailAgent
from services.mcp_agent import MCPAgent
from services.agents.rag_agent import RAGAgent
from services.agents.portfolio_agent import PortfolioAgent
from services.agents.stock_agent import StockAgent
from services.tools.stock_tool import StockTool


async def test():
    portfolio_csv = "data/portfolio.csv"
    metadata_csv = "data/metadata.csv"

    agents = {
        "rag": RAGAgent(),
        "portfolio": PortfolioAgent(
            portfolio_csv=portfolio_csv, metadata_csv=metadata_csv
        ),
        "stock": StockAgent(StockTool()),
        "email": EmailAgent(portfolio_csv=portfolio_csv, metadata_csv=metadata_csv),
    }
    mcp = MCPAgent(agents)

    q = "Sends daily snapshots"
    print(await mcp.run(q))


if __name__ == "__main__":
    asyncio.run(test())
