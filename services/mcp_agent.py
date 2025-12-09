# File: mcp_agent.py
# Main MCPAgent class that ties everything together
# Imports from other files

import re
from typing import Dict, List, Optional, Tuple
import asyncio
import time
import logging
from collections import defaultdict

from services.tools.groq_wrapper import GroqLLM
from services.MCP.enums import AgentType, AgentConfig
from services.MCP.intent import QueryIntent
from services.MCP.cache import CacheManager
from services.MCP.handlers import handle_generic, handle_portfolio, handle_email
from services.MCP.prompts import build_agent_summary, create_system_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        self.llm = GroqLLM()
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        self._semaphore = asyncio.Semaphore(max_concurrent)
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
        self.cache_manager = CacheManager(enable_cache, self.agent_configs)
        self.handler_map = {
            AgentType.STOCK: handle_generic,
            AgentType.RAG: handle_generic,
            AgentType.PORTFOLIO: handle_portfolio,
            AgentType.EMAIL: handle_email,
            AgentType.WEBSEARCH: handle_generic,
        }

    def _log(self, *msg):
        if self.debug:
            logger.info(" ".join(map(str, msg)))

    async def _call_with_timeout(self, coro, timeout: float = None):
        try:
            return await asyncio.wait_for(coro, timeout or self.timeout)
        except (asyncio.TimeoutError, Exception):
            return None

    def _route_query(self, query: str) -> List[Tuple[AgentType, str, float]]:
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
        self, agents_to_run: List[Tuple[AgentType, str, float]]
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
                tasks = [self.handler_map[at](self, at, q) for at, q, c in group]
                results.extend([r for r in await asyncio.gather(*tasks) if r])
        return results

    def clear_cache(self):
        self.cache_manager.clear_cache()

    def get_cache_stats(self) -> Dict:
        return self.cache_manager.get_cache_stats()

    async def _generate_final_answer(
        self,
        query: str,
        agent_results: List[Dict],
        language: str = "English",
        style: str = "professional",
    ) -> str:
        """
        Generate a final Markdown response using Groq LLM based on all agent outputs.
        """
        agent_summary = build_agent_summary(query, agent_results)
        system_prompt = create_system_prompt(language=language, style=style)

        final_prompt = (
            f"{system_prompt}\n\n"
            f"{agent_summary}\n\n"
            f"All agent sources have been included as Markdown links above.\n"
            f"TASK: Synthesize the information above into a single coherent Markdown response with inline citations."
        )

        answer = await self.llm.call_async(
            prompt=final_prompt, model="llama-3.3-70b-versatile", max_tokens=1024
        )

        if isinstance(answer, dict):
            return str(answer.get("plan") or answer.get("content") or "")
        elif isinstance(answer, list):
            return "\n".join(str(a) for a in answer)
        elif isinstance(answer, str):
            return answer
        else:
            self._log(f"Unexpected LLM output type: {type(answer).__name__}")
            return str(answer)

    async def _merge_responses(
        self,
        query: str,
        agent_results: List[Dict],
        language: str = "English",
        style: str = "professional",
    ) -> Dict:
        """
        Deduplicate agent results, collect sources, and synthesize final answer.
        """
        if not agent_results:
            return {"response": "Couldn't find relevant info. Rephrase?", "sources": []}

        seen = set()
        unique_results = []
        all_sources = []
        for r in agent_results:
            if not isinstance(r, dict):
                r = {"content": str(r), "answer": str(r), "sources": []}
            key = str(r.get("content") or r.get("answer", ""))[:50]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
                # Collect sources
                all_sources.extend(r.get("sources", []))

        final_answer = await self._generate_final_answer(
            query, unique_results, language, style
        )

        return {"response": final_answer, "sources": all_sources}

    async def run(
        self, query: str, language: str = "English", style: str = "professional"
    ) -> Tuple[Dict, List[str], bool]:
        # Check cache
        cached_response = self.cache_manager.get_cached("final_response", query)
        if cached_response:
            return cached_response, [], True

        # Determine which agents to run
        agents_to_run = self._route_query(query)
        selected_names = [at.value for at, _, _ in agents_to_run]

        # Execute agents concurrently
        results = await self._execute_agents(agents_to_run)

        # Print sources per agent
        for res in results:
            agent_type = (
                res.get("type", "unknown") if isinstance(res, dict) else "unknown"
            )
            sources = res.get("sources", []) if isinstance(res, dict) else []
            print(f"\n🟢 [AGENT: {agent_type.upper()}] Sources:")

            if sources:
                for i, src in enumerate(sources, start=1):
                    # Ensure dictionary format for printing
                    if isinstance(src, dict):
                        src_name = src.get("source", "Unknown")
                        score = src.get("score", 0)
                        print(f"  🔹 {i}. {src_name} (score: {score})")
                    else:
                        print(f"  🔹 {i}. {src}")
            else:
                print("  ⚠️ No sources found.")

        # Merge responses & aggregate sources
        final_result = await self._merge_responses(query, results, language, style)

        # Cache final result
        self.cache_manager.set_cached("final_response", query, final_result)

        return final_result, selected_names, False
