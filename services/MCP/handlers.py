# File: handlers.py
# Contains handler functions for different agents

from typing import Dict, List, Optional
import asyncio
import logging
from .enums import AgentType  # Assuming enums.py is in the same package

logger = logging.getLogger(__name__)


async def handle_generic(
    self, agent_type: AgentType, query: str  # MCPAgent instance
) -> Optional[Dict]:
    cached = self.cache_manager.get_cached(agent_type.value, query)
    if cached:
        return cached

    async with self._semaphore:
        agent = self.agents.get(agent_type.value)
        if not agent:
            self._log(f"❌ No agent found for type {agent_type.value}")
            return None

        try:
            # Run agent (async if available)
            if hasattr(agent, "run_async"):
                res = await self._call_with_timeout(agent.run_async(query))
            else:
                res = await asyncio.get_event_loop().run_in_executor(
                    None, agent.run, query
                )

            if res is None:
                return None

            sources = []

            # ----------------------------
            # RAG Agent
            # ----------------------------
            if agent_type == AgentType.RAG:
                # Expect dict with 'answer' and 'sources'
                if not isinstance(res, dict):
                    res = {"answer": str(res), "sources": []}
                sources = res.get("sources", [])
                result = {
                    "type": "rag",
                    "answer": res.get("answer", ""),
                    "sources": sources,
                    "query": query,
                    "raw_data": res,
                }

                # Print sources to terminal
                print(f"\n🟣 [RAG Agent] Sources for query: '{query}'")
                for i, src in enumerate(sources, start=1):
                    print(
                        f"  📄 {i}. {src.get('source','Unknown')} (score: {src.get('score',0)})"
                    )

            # ----------------------------
            # WebSearch Agent
            # ----------------------------
            elif agent_type == AgentType.WEBSEARCH:
                content_lines = []
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, dict):
                            summary = item.get("summary", "")
                            src = item.get("source", "Unknown")
                            content_lines.append(f"- {summary} [{src}]({src})")
                            sources.append(
                                {"source": src, "score": item.get("score", 0)}
                            )
                        else:
                            content_lines.append(f"- {str(item)}")
                            sources.append({"source": str(item), "score": 0})
                else:
                    content_lines.append(f"- {str(res)}")
                    sources.append({"source": str(res), "score": 0})

                content = "\n".join(content_lines)
                result = {
                    "type": "websearch",
                    "content": content,
                    "sources": sources,
                    "query": query,
                    "raw_data": res,
                }

                # Print sources
                print(f"\n🟢 [WebSearch Agent] Sources for query: '{query}'")
                for i, src in enumerate(sources, start=1):
                    print(
                        f"  📄 {i}. {src.get('source','Unknown')} (score: {src.get('score',0)})"
                    )

            # ----------------------------
            # Other Agents (portfolio, stock, email, etc.)
            # ----------------------------
            else:
                if isinstance(res, (dict, list)):
                    content = res
                else:
                    content = str(res)
                sources = [{"source": agent_type.value, "score": 0}]
                result = {
                    "type": agent_type.value,
                    "content": content,
                    "sources": sources,
                    "query": query,
                    "raw_data": res,
                }

                # Print sources
                print(
                    f"\n🔵 [{agent_type.value.capitalize()} Agent] Sources for query: '{query}'"
                )
                for i, src in enumerate(sources, start=1):
                    print(
                        f"  📄 {i}. {src.get('source','Unknown')} (score: {src.get('score',0)})"
                    )

            # ----------------------------
            # Debug Logs
            # ----------------------------
            self._log(f"[RAW OUTPUT] {agent_type.value}: {res}")
            self._log(f"[NORMALIZED OUTPUT] {agent_type.value}: {result}")

            # Cache and return
            self.cache_manager.set_cached(agent_type.value, query, result)
            return result

        except Exception as e:
            self._log(f"{agent_type.value} error: {e}")
            result = {
                "type": agent_type.value,
                "content": f"Error occurred: {e}. Please contact the technical team at semernahdi25@gmail.com",
                "sources": [{"source": agent_type.value, "score": 0}],
                "query": query,
                "raw_data": None,
            }
            self.cache_manager.set_cached(agent_type.value, query, result)
            return result


async def handle_portfolio(
    self, agent_type: AgentType, query: str  # MCPAgent instance
) -> Optional[str]:
    cached = self.cache_manager.get_cached(agent_type.value, query)
    if cached:
        return cached
    async with self._semaphore:
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.agents["portfolio"].run, query
            )
            if not isinstance(results, list):
                results = [{"method": "analyze", "result": results or {}, "meta": {}}]
            parts = ["**Portfolio Summary:**"]
            for action in results:
                result = action.get("result", {})
                if not result:
                    continue
                method = action.get("method", "")
                if method == "analyze" and isinstance(result, dict):
                    parts.append(
                        f"**Total Value:** ${result.get('total_value', 0):,.2f}"
                    )
                    parts.append(f"**Total Cost:** ${result.get('total_cost', 0):,.2f}")
                    gain = result.get("total_gain_loss", 0)
                    sign = "+" if gain >= 0 else ""
                    parts.append(f"**Gain/Loss:** {sign}${gain:,.2f}")
                elif method == "top_holdings" and isinstance(result, list):
                    parts.append(
                        f"**Top Holdings ({len(result)}):** "
                        + ", ".join(h.get("ticker", "") for h in result)
                    )
                elif method == "sector_allocation" and isinstance(result, dict):
                    parts.append(
                        "**Sector Allocation:** "
                        + ", ".join(f"{s}: {p}%" for s, p in result.items())
                    )
                elif isinstance(result, list):
                    parts.append(f"**Holdings:** {len(result)} items")
                elif isinstance(result, str):
                    parts.append(result)
            result_text = "\n".join(parts)
            if len(parts) > 1:
                self.cache_manager.set_cached(agent_type.value, query, result_text)
                return result_text
        except Exception as e:
            self._log(f"Portfolio error: {e}")
    return None


async def handle_email(
    self, agent_type: AgentType, query: str  # MCPAgent instance
) -> Optional[str]:
    async with self._semaphore:
        try:
            normalized = (
                "daily snapshot"
                if any(
                    kw in query.lower() for kw in ["send", "email", "mail", "report"]
                )
                else query
            )
            res = await asyncio.get_event_loop().run_in_executor(
                None, self.agents["email"].run, normalized
            )
            if (
                res
                and "error" not in str(res).lower()
                and "unrecognized" not in str(res).lower()
            ):
                return f"**Email:** {res}"
        except Exception as e:
            self._log(f"Email error: {e}")
    return None
