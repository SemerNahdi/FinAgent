# services/mcp_agent.py
import asyncio
import json
import logging
from services.tools.groq_wrapper import GroqLLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import re

# services/agents/mcp_agent_polished.py
import asyncio
import re


class MCPAgent:
    """
    Multi-agent orchestrator that merges responses from stock, RAG, portfolio, and email agents
    into a single polished, human-readable format with markdown formatting.
    """

    def __init__(self, agents: dict):
        """
        agents: dict with keys 'stock', 'rag', 'portfolio', 'email' mapping to agent instances
        """
        self.agents = agents

    # ------------------------
    # Agent handlers
    # ------------------------
    async def _handle_stock(self, query):
        try:
            res = self.agents["stock"].run(query)
            if not res or "error" in res:
                return ""
            # Price
            if "current_price" in res:
                return f"- **Stock Price:** The current price of **{res['ticker']}** is **${res['current_price']}**."
            # Moving average
            elif any(k.endswith("_day_ma") for k in res.keys()):
                period_key = [k for k in res if k.endswith("_day_ma")][0]
                return f"- **{period_key.replace('_','-').capitalize()}:** {res['ticker']} = **${res[period_key]}**."
            # Summary or other
            else:
                return f"- **Stock Info ({res.get('ticker','')}):** {res}"
        except Exception:
            return ""

    async def _handle_rag(self, query):
        try:
            res = await self.agents["rag"].run_async(query)
            if res:
                return f"- **RAG Insight:** {res}"
            return ""
        except Exception:
            return ""

    async def _handle_portfolio(self, query):
        try:
            analysis = self.agents["portfolio"].run(query)
            if not analysis:
                return ""
            top = self.agents["portfolio"].top_holdings(5)
            sectors = self.agents["portfolio"].sector_allocation()

            top_holdings_text = "\n".join(
                [f"  - {t['ticker']}: ${t['value']}" for t in top]
            )
            sector_text = "\n".join([f"  - {s}: {p}%" for s, p in sectors.items()])

            return (
                f"- **Portfolio Summary:**\n"
                f"  **Top 5 Holdings:**\n{top_holdings_text}\n"
                f"  **Sector Allocation:**\n{sector_text}"
            )
        except Exception:
            return ""

    async def _handle_email(self, query):
        try:
            res = self.agents["email"].run(query)
            if "unrecognized" in res:
                return ""
            return f"- **Email Action:** {res}"
        except Exception:
            return ""

    # ------------------------
    # Query splitting
    # ------------------------
    def _split_query(self, query):
        """
        Splits multi-part queries into agent-specific chunks
        """
        chunks = []

        if re.search(
            r"(price of|current price of|\d+[- ]?day ma|summary of|stock summary)",
            query,
            re.I,
        ):
            chunks.append(("stock", query))

        if re.search(
            r"(explain|what is|who|give info|finance|investment|analysis)", query, re.I
        ):
            chunks.append(("rag", query))

        if re.search(
            r"(portfolio|top holdings|sector allocation|filter by date)", query, re.I
        ):
            chunks.append(("portfolio", query))

        if "daily snapshot" in query.lower():
            chunks.append(("email", query))

        return chunks

    # ------------------------
    # Merge responses into human-readable markdown
    # ------------------------
    def _merge_responses(self, responses: list) -> str:
        """
        Merge all agent outputs into a single human-readable markdown string.
        Supports bullets, bold/italic formatting, links, and structured sections.
        """
        # Filter empty
        valid = [r for r in responses if r]
        if not valid:
            return "Sorry, I couldn't find an answer. Try rephrasing your query."

        # Join with two line breaks for readability
        return "\n\n".join(valid)

    # ------------------------
    # Main entry
    # ------------------------
    async def run(self, query: str) -> str:
        chunks = self._split_query(query)
        tasks = []

        for agent_type, chunk_query in chunks:
            if agent_type == "stock":
                tasks.append(self._handle_stock(chunk_query))
            elif agent_type == "rag":
                tasks.append(self._handle_rag(chunk_query))
            elif agent_type == "portfolio":
                tasks.append(self._handle_portfolio(chunk_query))
            elif agent_type == "email":
                tasks.append(self._handle_email(chunk_query))

        results = await asyncio.gather(*tasks)
        return self._merge_responses(results)
