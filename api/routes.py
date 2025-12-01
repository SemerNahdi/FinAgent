# backend/api/routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from services.agents.rag_agent import RAGAgent
from services.agents.portfolio_agent import PortfolioAgent
from services.agents.stock_agent import StockAgent
from services.agents.email_agent import EmailAgent
from services.agents.websearch_agent import WebSearchAgent
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
websearch_agent = WebSearchAgent()

# Initialize enhanced MCP agent with intelligent routing and caching
mcp = MCPAgent(
    agents={
        "stock": stock_agent,
        "rag": rag_agent,
        "portfolio": portfolio_agent,
        "email": email_agent,
        "websearch": websearch_agent,
    },
    max_concurrent=5,  # Max 5 agents running in parallel
    timeout=30.0,  # 30 second timeout per agent
    confidence_threshold=0.4,  # Only route to agents with 40%+ confidence
    enable_cache=True,  # Enable response caching
)


class Query(BaseModel):
    question: str


class QueryResponse(BaseModel):
    response: str
    agents_used: Optional[list] = None
    cache_hit: Optional[bool] = None


class CacheStats(BaseModel):
    enabled: bool
    size: Optional[int] = None
    entries: Optional[list] = None


@router.post("/ask", response_model=QueryResponse)
async def ask(query: Query):
    """
    Main query endpoint with enhanced intelligent routing.

    The orchestrator will:
    1. Analyze query intent to determine relevant agents
    2. Execute agents in parallel (respecting dependencies)
    3. Return cached results when available
    4. Merge responses intelligently
    """
    try:
        response, agents_used = await mcp.run(query.question)
        return {"response": response, "agents_used": agents_used}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """
    Get current cache statistics.
    Useful for monitoring cache performance.
    """
    try:
        stats = mcp.get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached responses.
    Useful for testing or when you need fresh data.
    """
    try:
        mcp.clear_cache()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify service is running.
    """
    return {
        "status": "healthy",
        "agents": list(mcp.agents.keys()),
        "cache_enabled": mcp.enable_cache,
        "max_concurrent": mcp.max_concurrent,
    }
