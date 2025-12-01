import re
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import asyncio
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentType(Enum):
    STOCK = "stock"
    RAG = "rag"
    PORTFOLIO = "portfolio"
    EMAIL = "email"
    WEBSEARCH = "websearch"


@dataclass
class AgentConfig:
    type: AgentType
    pattern: re.Pattern
    priority: int
    dependencies: set[AgentType]
    required_keywords: set[str]
    cache_ttl: int = 0


class QueryIntent:
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
        query_lower = query.lower()
        scores = defaultdict(float)
        for agent_type, keywords in cls.INTENT_KEYWORDS.items():
            scores[agent_type] += sum(
                0.7 for kw in keywords["strong"] if kw in query_lower
            )
            scores[agent_type] += sum(
                0.3 for kw in keywords["weak"] if kw in query_lower
            )
        return {k: min(v, 1.0) for k, v in scores.items()}


class MCPAgent:
    """
    Refactored MCPAgent:
      - Centralized handler map with generic handler for most agents to reduce code duplication.
      - Improved portfolio response parsing to handle multiple action types (e.g., top_holdings, sector_allocation) more robustly.
      - Simplified email normalization and added flexibility for query passing.
      - Added debug logging similar to previous agents.
      - Optimized routing and execution: Used list comprehensions, removed unnecessary prints/tracebacks.
      - Scalability: Agent configs as dict for easy extension; generic handler allows adding new agents without new methods.
      - Cache uses frozenset for keys if needed; added safe pop.
      - Removed redundancies in response merging (used set for seen).
      - Aligned with refactored PortfolioAgent (handles list of dicts, extracts based on method).
    """

    def __init__(
        self,
        agents: Dict,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        confidence_threshold: float = 0.4,
        enable_cache: bool = True,
        debug: bool = False,
    ):
        self.agents = agents
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache
        self.debug = debug
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache = {} if enable_cache else None
        self._cache_timestamps = {} if enable_cache else None
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
        self.handler_map = {
            AgentType.STOCK: self._handle_generic,
            AgentType.RAG: self._handle_generic,
            AgentType.PORTFOLIO: self._handle_portfolio,
            AgentType.EMAIL: self._handle_email,
            AgentType.WEBSEARCH: self._handle_generic,
        }

    def _log(self, *msg):
        if self.debug:
            logger.info(" ".join(map(str, msg)))

    def _get_cache_key(self, agent_type: str, query: str) -> str:
        return f"{agent_type}:{hash(query.lower().strip())}"

    def _get_cached(self, agent_type: str, query: str) -> Optional[str]:
        if not self.enable_cache or not self._cache:
            return None
        config = self.agent_configs.get(AgentType(agent_type))
        if not config or config.cache_ttl == 0:
            return None
        key = self._get_cache_key(agent_type, query)
        if (
            key in self._cache
            and time.time() - self._cache_timestamps[key] < config.cache_ttl
        ):
            return self._cache[key]
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)
        return None

    def _set_cached(self, agent_type: str, query: str, result: str):
        if not self.enable_cache or not self._cache:
            return
        config = self.agent_configs.get(AgentType(agent_type))
        if not config or config.cache_ttl == 0:
            return
        key = self._get_cache_key(agent_type, query)
        self._cache[key] = result
        self._cache_timestamps[key] = time.time()

    async def _call_with_timeout(self, coro, timeout: float = None):
        try:
            return await asyncio.wait_for(coro, timeout or self.timeout)
        except (asyncio.TimeoutError, Exception):
            return None

    async def _handle_generic(self, agent_type: AgentType, query: str) -> Optional[str]:
        cached = self._get_cached(agent_type.value, query)
        if cached:
            return cached
        async with self._semaphore:
            agent = self.agents[agent_type.value]
            try:
                if hasattr(agent, "run_async"):
                    res = await self._call_with_timeout(agent.run_async(query))
                else:
                    res = await asyncio.get_event_loop().run_in_executor(
                        None, agent.run, query
                    )
                if res and "error" not in str(res).lower():
                    result = f"**{agent_type.value.capitalize()} Info:** {res}"
                    self._set_cached(agent_type.value, query, result)
                    return result
            except Exception as e:
                self._log(f"{agent_type.value} error: {e}")
        return None

    async def _handle_portfolio(
        self, agent_type: AgentType, query: str
    ) -> Optional[str]:
        cached = self._get_cached(agent_type.value, query)
        if cached:
            return cached
        async with self._semaphore:
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None, self.agents["portfolio"].run, query
                )
                if not isinstance(results, list):
                    results = [
                        {"method": "analyze", "result": results or {}, "meta": {}}
                    ]
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
                        parts.append(
                            f"**Total Cost:** ${result.get('total_cost', 0):,.2f}"
                        )
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
                    self._set_cached(agent_type.value, query, result_text)
                    return result_text
            except Exception as e:
                self._log(f"Portfolio error: {e}")
        return None

    async def _handle_email(self, agent_type: AgentType, query: str) -> Optional[str]:
        async with self._semaphore:
            try:
                normalized = (
                    "daily snapshot"
                    if any(
                        kw in query.lower()
                        for kw in ["send", "email", "mail", "report"]
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

    def _route_query(self, query: str) -> List[tuple[AgentType, str, float]]:
        intent_scores = QueryIntent.analyze(query)
        selected = []
        for agent_type, config in self.agent_configs.items():
            confidence = intent_scores.get(agent_type, 0.0)
            if config.pattern.search(query):
                confidence = max(confidence, 0.5)
            if confidence >= self.confidence_threshold:
                selected.append((agent_type, query, confidence))
        if not selected:
            selected = [(AgentType.RAG, query, 0.5)]
        selected.sort(key=lambda x: (-x[2], self.agent_configs[x[0]].priority))
        return selected

    async def _execute_agents(
        self, agents_to_run: List[tuple[AgentType, str, float]]
    ) -> List[str]:
        independent = [
            (at, q, c)
            for at, q, c in agents_to_run
            if not self.agent_configs[at].dependencies
        ]
        dependent = [
            (at, q, c)
            for at, q, c in agents_to_run
            if self.agent_configs[at].dependencies
        ]
        results = []
        for group in [independent, dependent]:
            if group:
                tasks = [self.handler_map[at](at, q) for at, q, c in group]
                results.extend([r for r in await asyncio.gather(*tasks) if r])
        return results

    def _merge_responses(self, responses: List[str]) -> str:
        if not responses:
            return "Couldn't find relevant info. Rephrase?"
        seen = set()
        unique = [r for r in responses if r[:50] not in seen and not seen.add(r[:50])]
        return "\n\n".join(unique)

    async def run(self, query: str) -> tuple[str, List[str]]:
        agents_to_run = self._route_query(query)
        selected_names = [at.value for at, _, _ in agents_to_run]
        self._log(
            f"Query: {query}",
            f"Selected: {selected_names}",
            f"Scores: {[(at.value, c) for at, _, c in agents_to_run]}",
        )
        results = await self._execute_agents(agents_to_run)
        self._log(f"Results: {[r[:50] if r else 'None' for r in results]}")
        response = self._merge_responses(results)
        return response, selected_names

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_cache_stats(self) -> Dict:
        return {
            "enabled": self.enable_cache,
            "size": len(self._cache) if self._cache else 0,
            "entries": list(self._cache.keys()) if self._cache else [],
        }
