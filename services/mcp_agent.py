# File: mcp_agent.py

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

        self._log(
            f"Intent scores: {[(k.value, v) for k, v in intent_scores.items() if v > 0]}"
        )

        for agent_type, config in self.agent_configs.items():
            confidence = intent_scores.get(agent_type, 0.0)
            # Boost confidence if pattern matches
            if config.pattern.search(query):
                confidence = max(confidence, 0.5)

            # Lower threshold for multi-intent queries (allow more agents)
            threshold = self.confidence_threshold
            if len([s for s in intent_scores.values() if s > 0]) > 1:
                # Multi-intent query - be more permissive
                threshold = max(0.2, threshold - 0.1)

            if confidence >= threshold:
                selected.append((agent_type, query, confidence))
                self._log(
                    f"Selected agent: {agent_type.value} (confidence: {confidence:.2f})"
                )

        if not selected:
            self._log("No agents selected - query may be out of scope")
            return []

        selected.sort(key=lambda x: (-x[2], self.agent_configs[x[0]].priority))
        self._log(f"Final agent selection: {[at.value for at, _, _ in selected]}")
        return selected

    async def _execute_agents(
        self, agents_to_run: List[Tuple[AgentType, str, float]]
    ) -> List[Dict]:
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
                agent_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, res in enumerate(agent_results):
                    if isinstance(res, Exception):
                        self._log(f"Agent {group[i][0].value} raised exception: {res}")
                        continue
                    if res is None:
                        continue
                    # Normalize to dict if needed
                    if not isinstance(res, dict):
                        res = {
                            "type": group[i][0].value,
                            "content": str(res),
                            "sources": [],
                            "query": group[i][1],
                            "raw_data": res,
                        }
                    results.append(res)
        return results

    def clear_cache(self):
        self.cache_manager.clear_cache()

    def get_cache_stats(self) -> Dict:
        return self.cache_manager.get_cache_stats()

    async def _generate_no_agent_response(
        self,
        query: str,
        language: str = "English",
        style: str = "professional",
    ) -> str:
        """
        Generate a graceful response when no agents are selected for the query.
        Uses LLM to create a contextually appropriate response.
        """
        query_lower = query.lower().strip()

        # Detect simple greetings
        greeting_patterns = [
            "hey",
            "hi",
            "hello",
            "greetings",
            "what's up",
            "howdy",
            "who are you",
            "what are you",
            "introduce yourself",
            "tell me about yourself",
        ]

        is_greeting = any(pattern in query_lower for pattern in greeting_patterns)

        if is_greeting:
            # Simple, friendly greeting response
            if "who are you" in query_lower or "what are you" in query_lower:
                return "Hey! I'm your helpful finance assistant. I can help you with stock prices, portfolio analysis, market news, and other financial topics. What would you like to know?"
            else:
                return "Hey! I'm here to help with your finance questions. What can I help you with today?"

        system_prompt = create_system_prompt(language=language, style=style)

        no_agent_prompt = (
            f"{system_prompt}\n\n"
            f"User Query: {query}\n\n"
            f"TASK: Respond to the user's query in a concise, friendly way:\n"
            f"- If it's a greeting, respond simply (1-2 sentences max)\n"
            f"- If it's a non-financial question, politely explain you specialize in financial topics (1-2 sentences)\n"
            f"- Keep it SHORT and natural - no long explanations or lists\n"
            f"- Do NOT mention agents, tools, or technical details\n"
            f"- Respond in {language} using {style} style\n"
            f"- Be conversational and friendly"
        )

        answer = await self.llm.call_async(
            prompt=no_agent_prompt, model="llama-3.3-70b-versatile", max_tokens=200
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
            f"USER QUERY:\n{query}\n\n"
            f"{agent_summary}\n\n"
            f"TASK:\n"
            f"- Answer the user query using ONLY the information above.\n"
            f"- Follow all system rules strictly.\n"
            f"- Output a clean, well-structured Markdown response.\n"
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

        # Handle case when no agents are selected
        if not agents_to_run:
            self._log("No agents selected for query, generating graceful response")
            graceful_response = await self._generate_no_agent_response(
                query, language, style
            )
            final_result = {
                "response": graceful_response,
                "sources": [],
            }
            # Cache the response
            self.cache_manager.set_cached("final_response", query, final_result)
            return final_result, [], False

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
