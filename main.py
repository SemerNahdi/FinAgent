import asyncio
from services.agents import (
    daily_snapshot_agent,
    portfolio_agent,
    rag_agent,
    stock_agent,
)
from services.mcp_agent import MCPAgent

agents = {
    "portfolio": portfolio_agent,
    "rag": rag_agent,
    "email": daily_snapshot_agent,
    "stock": stock_agent,
}

mcp = MCPAgent(agents)

result = asyncio.run(mcp.run("Send today's portfolio snapshot to my email"))
print(result)
