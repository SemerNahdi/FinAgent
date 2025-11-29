# backend/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

from services.agents.rag_agent import RAGAgent
from services.agents.portfolio_agent import PortfolioAgent
from services.agents.stock_agent import StockAgent
from services.agents.email_agent import EmailAgent
from services.mcp_agent import MCPAgent
from services.tools.stock_tool import StockTool

router = APIRouter()

# CSV paths
PORTFOLIO_CSV = "data/portfolio.csv"
METADATA_CSV = "data/metadata.csv"

# Initialize agents
stock_agent = StockAgent(StockTool())
rag_agent = RAGAgent()
portfolio_agent = PortfolioAgent(portfolio_csv=PORTFOLIO_CSV, metadata_csv=METADATA_CSV)
email_agent = EmailAgent(portfolio_csv=PORTFOLIO_CSV, metadata_csv=METADATA_CSV)

# Initialize polished MCP agent

mcp = MCPAgent(
    agents={
        "stock": stock_agent,
        "rag": rag_agent,
        "portfolio": portfolio_agent,
        "email": email_agent,
    }
)


class Query(BaseModel):
    question: str


@router.post("/ask")
async def ask(query: Query):
    try:
        response = await mcp.run(query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
