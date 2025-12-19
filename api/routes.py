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
from services.tools.groq_wrapper import GroqLLM
from services.language.detect import get_language_service

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
    max_concurrent=5,
    timeout=30.0,
    confidence_threshold=0.4,
    enable_cache=True,
)

# Initialize language detection service
groq_llm = GroqLLM()
language_service = get_language_service()


class Query(BaseModel):
    question: str


class LanguageInfo(BaseModel):
    language: str
    dialect: Optional[str] = None
    confidence: float
    is_dialect: bool
    detected_by: str


class QueryResponse(BaseModel):
    response: str
    sources: list
    agents_used: Optional[list] = None
    cache_hit: Optional[bool] = None
    language_info: Optional[LanguageInfo] = None


class CacheStats(BaseModel):
    enabled: bool
    size: Optional[int] = None
    entries: Optional[list] = None


@router.post("/ask", response_model=QueryResponse)
async def ask(query: Query, style: str = "professional"):
    """
    Process user query with automatic language detection and dialect-aware responses.

    Flow:
    1. Detect language and dialect (local first, Groq fallback)
    2. Get appropriate response language
    3. Process query with MCP agents
    4. Return response in detected language/dialect
    """
    try:
        # Step 1: Detect language and dialect
        detection_result = await language_service.detect(query.question)

        # Step 2: Get response language and instruction
        response_language = language_service.get_response_language(detection_result)

        # Step 3: Process query with MCP agents
        # Pass the response instruction to ensure proper language/dialect in response
        final_result, agents_used, cache_hit = await mcp.run(
            query.question,
            language=response_language,
            style=style,
        )

        # Step 4: Return response with language info
        return {
            "response": final_result.get("response", ""),
            "sources": final_result.get("sources", []),
            "agents_used": agents_used,
            "cache_hit": cache_hit,
            "language_info": {
                "language": detection_result.language,
                "dialect": detection_result.dialect,
                "confidence": detection_result.confidence,
                "is_dialect": detection_result.is_dialect,
                "detected_by": detection_result.detected_by,
            },
        }

    except Exception as e:
        print(f"Error in ask endpoint: {e}")
        # Error message in English as fallback
        return {
            "response": (
                "An error occurred while processing your query. "
                "Please contact the technical team at semernahdi25@gmail.com for assistance."
            ),
            "sources": [],
            "agents_used": [],
            "cache_hit": None,
            "language_info": None,
        }


@router.post("/detect-language")
async def detect_language(query: Query):
    """
    Standalone endpoint to test language detection.
    Useful for debugging and monitoring detection accuracy.
    """
    try:
        result = await language_service.detect(query.question)
        response_lang = language_service.get_response_language(result)
        instruction = language_service.format_response_instruction(result)

        return {
            "detected_language": result.language,
            "dialect": result.dialect,
            "confidence": result.confidence,
            "is_dialect": result.is_dialect,
            "detected_by": result.detected_by,
            "response_language": response_lang,
            "response_instruction": instruction,
        }
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
        "language_detection": "enabled",
    }
