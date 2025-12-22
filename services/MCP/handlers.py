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
                            # Skip "no news found" messages
                            if "no recent news" in summary.lower() or "no news found" in summary.lower():
                                continue
                            content_lines.append(f"- {summary} [{src}]({src})")
                            sources.append(
                                {"source": src, "score": item.get("score", 0)}
                            )
                        else:
                            item_str = str(item)
                            # Skip "no news found" messages
                            if "no recent news" in item_str.lower() or "no news found" in item_str.lower():
                                continue
                            content_lines.append(f"- {item_str}")
                            sources.append({"source": item_str, "score": 0})
                else:
                    res_str = str(res)
                    # Skip "no news found" messages
                    if "no recent news" not in res_str.lower() and "no news found" not in res_str.lower():
                        content_lines.append(f"- {res_str}")
                        sources.append({"source": res_str, "score": 0})

                # Only return if we have actual content
                if not content_lines:
                    return None

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
            # Other Agents (stock, email, etc.)
            # ----------------------------
            else:
                if isinstance(res, (dict, list)):
                    content = res
                else:
                    content = str(res)
                
                # Filter out error messages for stock agent
                if agent_type == AgentType.STOCK:
                    content_str = str(content).lower()
                    error_indicators = [
                        "not recognized",
                        "try queries such",
                        "error occurred",
                        "please contact",
                    ]
                    if any(indicator in content_str for indicator in error_indicators):
                        return None
                
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
) -> Optional[Dict]:
    cached = self.cache_manager.get_cached(agent_type.value, query)
    if cached:
        if isinstance(cached, dict):
            return cached
        # Handle legacy string cache
        return {
            "type": "portfolio",
            "content": cached,
            "sources": [{"source": "portfolio", "score": 0}],
            "query": query,
            "raw_data": cached,
        }
    
    async with self._semaphore:
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.agents["portfolio"].run, query
            )
            if not isinstance(results, list):
                results = [{"method": "analyze", "result": results or {}, "meta": {}}]
            
            parts = []
            sources = []
            detailed_data = []
            
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
                    detailed_data.append({"method": "analyze", "data": result})
                    
                elif method == "top_holdings" and isinstance(result, list):
                    holdings_list = []
                    for h in result:
                        ticker = h.get("ticker", "")
                        shares = h.get("shares", 0)
                        value = h.get("value", 0)
                        # Only show shares if > 0, otherwise just show value
                        if shares > 0:
                            holdings_list.append(f"**{ticker}**: ${value:,.2f} ({shares} shares)")
                        else:
                            holdings_list.append(f"**{ticker}**: ${value:,.2f}")
                    
                    parts.append(f"**Top {len(result)} Holdings:**")
                    parts.append("\n".join(holdings_list))
                    detailed_data.append({"method": "top_holdings", "data": result})
                    sources.extend([{"source": f"portfolio_{h.get('ticker', '')}", "score": 0} for h in result])
                    
                elif method == "sector_allocation" and isinstance(result, dict):
                    parts.append("**Sector Allocation:**")
                    parts.append("\n".join(f"- {s}: {p}%" for s, p in result.items()))
                    detailed_data.append({"method": "sector_allocation", "data": result})
                    
                elif isinstance(result, list):
                    parts.append(f"**Holdings:** {len(result)} items")
                    detailed_data.append({"method": "holdings", "data": result})
                    
                elif isinstance(result, str):
                    parts.append(result)
            
            if not parts:
                return None
                
            content = "**Portfolio Summary:**\n" + "\n".join(parts)
            
            if not sources:
                sources = [{"source": "portfolio", "score": 0}]
            
            result_dict = {
                "type": "portfolio",
                "content": content,
                "sources": sources,
                "query": query,
                "raw_data": detailed_data if detailed_data else results,
            }
            
            # Print sources
            print(f"\n🔵 [Portfolio Agent] Sources for query: '{query}'")
            for i, src in enumerate(sources, start=1):
                print(f"  📄 {i}. {src.get('source','Unknown')} (score: {src.get('score',0)})")
            
            self.cache_manager.set_cached(agent_type.value, query, result_dict)
            return result_dict
            
        except Exception as e:
            self._log(f"Portfolio error: {e}")
            result = {
                "type": "portfolio",
                "content": f"Error occurred: {e}. Please contact the technical team at semernahdi25@gmail.com",
                "sources": [{"source": "portfolio", "score": 0}],
                "query": query,
                "raw_data": None,
            }
            self.cache_manager.set_cached(agent_type.value, query, result)
            return result


async def handle_email(
    self, agent_type: AgentType, query: str  # MCPAgent instance
) -> Optional[Dict]:
    cached = self.cache_manager.get_cached(agent_type.value, query)
    if cached:
        if isinstance(cached, dict):
            return cached
        # Handle legacy string cache
        return {
            "type": "email",
            "content": cached,
            "sources": [{"source": "email", "score": 0}],
            "query": query,
            "raw_data": cached,
        }
    
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
                content = f"**Email:** {res}"
                result = {
                    "type": "email",
                    "content": content,
                    "sources": [{"source": "email", "score": 0}],
                    "query": query,
                    "raw_data": res,
                }
                
                # Print sources
                print(f"\n🔵 [Email Agent] Sources for query: '{query}'")
                print(f"  📄 1. email (score: 0)")
                
                self.cache_manager.set_cached(agent_type.value, query, result)
                return result
        except Exception as e:
            self._log(f"Email error: {e}")
            result = {
                "type": "email",
                "content": f"Error occurred: {e}. Please contact the technical team at semernahdi25@gmail.com",
                "sources": [{"source": "email", "score": 0}],
                "query": query,
                "raw_data": None,
            }
            self.cache_manager.set_cached(agent_type.value, query, result)
            return result
    return None
