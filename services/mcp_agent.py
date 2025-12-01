import asyncio
import re
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import time
import traceback


class AgentType(Enum):
    STOCK = "stock"
    RAG = "rag"
    PORTFOLIO = "portfolio"
    EMAIL = "email"
    WEBSEARCH = "websearch"


@dataclass
class AgentConfig:
    """Configuration for agent routing and execution"""

    type: AgentType
    pattern: re.Pattern
    priority: int
    dependencies: Set[AgentType]
    required_keywords: Set[str]
    cache_ttl: int = 0


class QueryIntent:
    """Analyzes query intent for smarter routing"""

    INTENT_KEYWORDS = {
        AgentType.STOCK: {
            "strong": {
                "price",
                "ticker",
                "quote",
                "trading",
                "share",
                "stock",
                "ma",
                "moving average",
            },
            "weak": {"current", "value", "worth"},
        },
        AgentType.PORTFOLIO: {
            "strong": {
                "portfolio",
                "holdings",
                "my stocks",
                "allocation",
                "positions",
                "my holdings",
            },
            "weak": {"performance", "return", "total"},
        },
        AgentType.WEBSEARCH: {
            "strong": {"news", "latest", "recent", "breaking", "headlines", "update"},
            "weak": {"today", "this week", "current"},
        },
        AgentType.RAG: {
            "strong": {
                "explain",
                "what is",
                "define",
                "how does",
                "why",
                "tell me about",
            },
            "weak": {"information", "details", "about"},
        },
        AgentType.EMAIL: {
            "strong": {"email", "send", "notify", "report", "snapshot", "mail"},
            "weak": {"daily", "summary"},
        },
    }

    @classmethod
    def analyze(cls, query: str) -> Dict[AgentType, float]:
        """Returns confidence scores (0-1) for each agent type."""
        query_lower = query.lower()
        scores = defaultdict(float)

        for agent_type, keywords in cls.INTENT_KEYWORDS.items():
            # Strong keyword match = +0.7
            for kw in keywords["strong"]:
                if kw in query_lower:
                    scores[agent_type] += 0.7

            # Weak keyword match = +0.3
            for kw in keywords["weak"]:
                if kw in query_lower:
                    scores[agent_type] += 0.3

        # Cap scores at 1.0
        return {k: min(v, 1.0) for k, v in scores.items()}


class MCPAgent:
    """
    Enhanced multi-agent orchestrator with intelligent routing,
    dependency management, caching, and optimized parallel execution.
    """

    def __init__(
        self,
        agents: dict,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        confidence_threshold: float = 0.4,
        enable_cache: bool = True,
    ):
        self.agents = agents
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache = {} if enable_cache else None
        self._cache_timestamps = {} if enable_cache else None

        # Agent configurations
        self.agent_configs = {
            AgentType.STOCK: AgentConfig(
                type=AgentType.STOCK,
                pattern=re.compile(
                    r"(price|current price|\d+[- ]?day|ma|moving average|stock summary|ticker|quote)",
                    re.I,
                ),
                priority=1,
                dependencies=set(),
                required_keywords={"ticker", "stock", "price"},
                cache_ttl=60,
            ),
            AgentType.WEBSEARCH: AgentConfig(
                type=AgentType.WEBSEARCH,
                pattern=re.compile(
                    r"(news|latest|update|headlines|recent|breaking)", re.I
                ),
                priority=2,
                dependencies=set(),
                required_keywords={"news", "latest", "update"},
                cache_ttl=300,
            ),
            AgentType.PORTFOLIO: AgentConfig(
                type=AgentType.PORTFOLIO,
                pattern=re.compile(
                    r"(portfolio|top holdings|sector allocation|my stocks|my holdings)",
                    re.I,
                ),
                priority=3,
                dependencies=set(),
                required_keywords={"portfolio", "holdings"},
                cache_ttl=120,
            ),
            AgentType.RAG: AgentConfig(
                type=AgentType.RAG,
                pattern=re.compile(
                    r"(explain|what is|who|give info|define|describe|tell me)", re.I
                ),
                priority=4,
                dependencies=set(),
                required_keywords={"explain", "what", "define"},
                cache_ttl=600,
            ),
            AgentType.EMAIL: AgentConfig(
                type=AgentType.EMAIL,
                pattern=re.compile(
                    r"(daily snapshot|send email|email report|notify|mail)", re.I
                ),
                priority=5,
                dependencies={AgentType.PORTFOLIO},
                required_keywords={"email", "send", "notify"},
                cache_ttl=0,
            ),
        }

    # ------------------------
    # Caching utilities
    # ------------------------
    def _get_cache_key(self, agent_type: str, query: str) -> str:
        return f"{agent_type}:{hash(query.lower().strip())}"

    def _get_cached(self, agent_type: str, query: str) -> Optional[str]:
        if not self.enable_cache or not self._cache:
            return None

        try:
            config = self.agent_configs.get(AgentType(agent_type))
            if not config or config.cache_ttl == 0:
                return None

            cache_key = self._get_cache_key(agent_type, query)
            if cache_key in self._cache:
                age = time.time() - self._cache_timestamps[cache_key]
                if age < config.cache_ttl:
                    return self._cache[cache_key]
                else:
                    del self._cache[cache_key]
                    del self._cache_timestamps[cache_key]
        except:
            pass

        return None

    def _set_cached(self, agent_type: str, query: str, result: str):
        if not self.enable_cache or not self._cache:
            return

        try:
            config = self.agent_configs.get(AgentType(agent_type))
            if not config or config.cache_ttl == 0:
                return

            cache_key = self._get_cache_key(agent_type, query)
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        except:
            pass

    # ------------------------
    # Agent handlers with proper async
    # ------------------------
    async def _call_with_timeout(self, coro, timeout: float = None):
        try:
            return await asyncio.wait_for(coro, timeout=timeout or self.timeout)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    async def _handle_stock(self, query: str) -> Optional[str]:
        cached = self._get_cached("stock", query)
        if cached:
            return cached

        async with self._semaphore:
            try:
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(None, self.agents["stock"].run, query)

                if not res or "error" in str(res).lower():
                    return None

                result = ""
                if isinstance(res, dict):
                    if "current_price" in res:
                        result = f"**Stock Price:** {res['ticker']} is currently trading at **${res['current_price']}**"
                    elif any(k.endswith("_day_ma") for k in res.keys()):
                        period_key = next(k for k in res if k.endswith("_day_ma"))
                        period = period_key.replace("_", "-").replace(
                            "day-ma", "day MA"
                        )
                        result = f"**Moving Average:** {res['ticker']} {period} = **${res[period_key]:.2f}**"
                    else:
                        result = f"**Stock Info:** {res}"
                else:
                    result = f"**Stock Info:** {res}"

                if result:
                    self._set_cached("stock", query, result)
                return result
            except Exception as e:
                print(f"Stock agent error: {e}")
                return None

    async def _handle_rag(self, query: str) -> Optional[str]:
        cached = self._get_cached("rag", query)
        if cached:
            return cached

        async with self._semaphore:
            try:
                # Check if agent has async method
                if hasattr(self.agents["rag"], "run_async"):
                    res = await self._call_with_timeout(
                        self.agents["rag"].run_async(query)
                    )
                else:
                    loop = asyncio.get_event_loop()
                    res = await loop.run_in_executor(
                        None, self.agents["rag"].run, query
                    )

                if res:
                    result = f"**Information:** {res}"
                    self._set_cached("rag", query, result)
                    return result
                return None
            except Exception as e:
                print(f"RAG agent error: {e}")
                return None

    async def _handle_portfolio(self, query: str) -> Optional[str]:
        cached = self._get_cached("portfolio", query)
        if cached:
            return cached

        async with self._semaphore:
            try:
                loop = asyncio.get_event_loop()

                # Run all portfolio operations concurrently
                tasks = [
                    loop.run_in_executor(None, self.agents["portfolio"].run, query),
                    loop.run_in_executor(
                        None, self.agents["portfolio"].top_holdings, 5
                    ),
                    loop.run_in_executor(
                        None, self.agents["portfolio"].sector_allocation
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                analysis, top, sectors = results

                if isinstance(analysis, Exception) or not analysis:
                    return None

                parts = ["**Portfolio Summary:**"]

                if top and not isinstance(top, Exception) and len(top) > 0:
                    holdings = "\n".join(
                        [f"  • {t['ticker']}: ${t['value']:.2f}" for t in top]
                    )
                    parts.append(f"\n**Top 5 Holdings:**\n{holdings}")

                if sectors and not isinstance(sectors, Exception) and len(sectors) > 0:
                    sector_text = "\n".join(
                        [f"  • {s}: {p:.1f}%" for s, p in sectors.items()]
                    )
                    parts.append(f"\n**Sector Allocation:**\n{sector_text}")

                result = "\n".join(parts)
                self._set_cached("portfolio", query, result)
                return result
            except Exception as e:
                print(f"Portfolio agent error: {e}")
                return None

    async def _handle_email(self, query: str) -> Optional[str]:
        async with self._semaphore:
            try:
                # Normalize email queries to help the agent understand
                normalized_query = query.lower()
                if any(
                    word in normalized_query
                    for word in ["send", "email", "mail", "report"]
                ):
                    # Convert to format email agent expects
                    normalized_query = "daily snapshot"

                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None, self.agents["email"].run, normalized_query
                )

                print(f"Email agent raw response: {res}")  # Debug

                if (
                    res
                    and "unrecognized" not in str(res).lower()
                    and "error" not in str(res).lower()
                ):
                    return f"**Email:** {res}"
                else:
                    print(f"Email agent rejected response: {res}")
                return None
            except Exception as e:
                print(f"Email agent error: {e}")
                traceback.print_exc()
                return None

    async def _handle_websearch(self, query: str) -> Optional[str]:
        cached = self._get_cached("websearch", query)
        if cached:
            return cached

        async with self._semaphore:
            try:
                loop = asyncio.get_event_loop()
                res = await loop.run_in_executor(
                    None, self.agents["websearch"].run, query
                )

                if res:
                    result = f"**Latest News:**\n{res}"
                    self._set_cached("websearch", query, result)
                    return result
                return None
            except Exception as e:
                print(f"WebSearch agent error: {e}")
                return None

    # ------------------------
    # Intelligent routing
    # ------------------------
    def _route_query(self, query: str) -> List[Tuple[AgentType, str, float]]:
        """Route query to appropriate agents based on intent and patterns."""
        intent_scores = QueryIntent.analyze(query)
        selected_agents = []

        for agent_type, config in self.agent_configs.items():
            confidence = intent_scores.get(agent_type, 0.0)

            # Boost confidence if pattern matches
            if config.pattern.search(query):
                confidence = max(confidence, 0.6)

            # Add agent if confidence exceeds threshold
            if confidence >= self.confidence_threshold:
                selected_agents.append((agent_type, query, confidence))

        # If no agents selected, use RAG as fallback
        if not selected_agents:
            selected_agents.append((AgentType.RAG, query, 0.5))

        # Sort by confidence (desc) then priority (asc)
        selected_agents.sort(key=lambda x: (-x[2], self.agent_configs[x[0]].priority))

        return selected_agents

    # ------------------------
    # Execution
    # ------------------------
    async def _execute_agents(
        self, agents_to_run: List[Tuple[AgentType, str, float]]
    ) -> List[str]:
        """Execute agents respecting dependencies."""
        handler_map = {
            AgentType.STOCK: self._handle_stock,
            AgentType.RAG: self._handle_rag,
            AgentType.PORTFOLIO: self._handle_portfolio,
            AgentType.EMAIL: self._handle_email,
            AgentType.WEBSEARCH: self._handle_websearch,
        }

        # Separate independent and dependent agents
        independent = []
        dependent = []

        for item in agents_to_run:
            agent_type = item[0]
            query = item[1]
            conf = item[2]

            config = self.agent_configs[agent_type]
            if config.dependencies:
                dependent.append((agent_type, query, conf))
            else:
                independent.append((agent_type, query, conf))

        results = []

        # Run independent agents in parallel
        if independent:
            tasks = [
                handler_map[agent_type](query)
                for agent_type, query, conf in independent
            ]
            results.extend(await asyncio.gather(*tasks))

        # Run dependent agents after
        if dependent:
            tasks = [
                handler_map[agent_type](query) for agent_type, query, conf in dependent
            ]
            results.extend(await asyncio.gather(*tasks))

        return results

    # ------------------------
    # Response merging
    # ------------------------
    def _merge_responses(self, responses: List[str]) -> str:
        valid = [r for r in responses if r]
        if not valid:
            return (
                "I couldn't find a relevant answer. Could you rephrase your question?"
            )

        # Remove duplicates
        seen = set()
        unique = []
        for response in valid:
            key = response[:50]
            if key not in seen:
                seen.add(key)
                unique.append(response)

        return "\n\n".join(unique)

    # ------------------------
    # Main entry
    # ------------------------
    async def run(self, query: str) -> Tuple[str, List[str]]:
        """
        Main orchestrator entry point.
        Returns: (response, list of agents that were selected)
        """
        agents_to_run = self._route_query(query)

        # Track which agents were selected
        selected_agent_names = [agent_type.value for agent_type, _, _ in agents_to_run]

        print(f"Query: {query}")
        print(f"Selected agents: {selected_agent_names}")
        print(
            f"Agent confidence scores: {[(a.value, f'{c:.2f}') for a, _, c in agents_to_run]}"
        )

        results = await self._execute_agents(agents_to_run)

        print(f"Results from agents: {[r[:100] if r else 'None' for r in results]}")

        response = self._merge_responses(results)
        return response, selected_agent_names

    # ------------------------
    # Utilities
    # ------------------------
    def clear_cache(self):
        if self._cache:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_cache_stats(self) -> Dict:
        if not self._cache:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": len(self._cache),
            "entries": list(self._cache.keys()),
        }
