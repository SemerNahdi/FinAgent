# services/mcp_agent.py
import asyncio
import json
import logging
from services.tools.groq_wrapper import GroqLLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MCPAgent:
    """
    Async MCP Agent using Groq for context-aware tool selection.
    """

    def __init__(self, agents: dict):
        self.agents = agents
        self.llm = GroqLLM()  # Groq wrapper

        # Descriptions for each agent/tool
        self.tool_descriptions = {
            "portfolio": (
                "Analyzes user portfolios, computes total value, profit/loss, "
                "sector allocation, top holdings, and historical performance."
            ),
            "rag": "Answers queries from uploaded PDFs, CSVs, and JSONs.",
            "stock": "Provides stock prices, top market movers, and market summaries.",
            "summarizer": "Summarizes financial reports or extracted content.",
            "email": "Sends daily snapshots, alerts, and custom emails.",
        }

    async def run_agent(self, agent_name: str, query: str):
        agent = self.agents.get(agent_name)
        if not agent:
            return f"[Agent '{agent_name}' not found]"
        try:
            if asyncio.iscoroutinefunction(agent.run):
                result = await agent.run(query)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent.run, query)
            return f"[{agent_name}]: {result}"
        except Exception as e:
            logger.error(f"Error running agent '{agent_name}': {e}")
            return f"[{agent_name} error: {str(e)}]"

    async def run(self, query: str):
        """
        Run the MCP agent for a user query:
        1. Ask Groq which agents/tools to call.
        2. Run all selected agents asynchronously.
        3. Aggregate results.
        """
        plan = await self.plan_tools(query)
        tasks = [self.run_agent(agent_name, query) for agent_name in plan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to readable messages
        normalized_results = [
            str(r) if isinstance(r, Exception) else r for r in results
        ]

        return self.aggregate(normalized_results)

    async def plan_tools(self, query: str):
        """
        Ask Groq which tools to call based on query + current context.
        Returns a normalized list of agent names.
        """
        # Gather context
        context = {
            "portfolio": getattr(
                self.agents.get("portfolio"), "get_portfolio_summary", lambda: []
            )(),
            "stock_agents": getattr(
                self.agents.get("stock"), "get_top_movers", lambda: []
            )(),
            "email_tools": ["daily_snapshot", "alerts"],
            "available_agents": list(self.agents.keys()),
        }

        descriptions_text = "\n".join(
            f"{name}: {desc}"
            for name, desc in self.tool_descriptions.items()
            if name in context["available_agents"]
        )

        # Groq prompt with strict JSON and example
        prompt = f"""
You are a multi-agent planner for a finance assistant. 
Available agents with descriptions:
{descriptions_text}

Portfolio snapshot: {context['portfolio']}
Top stock movers: {context['stock_agents']}
Email capabilities: {context['email_tools']}

Determine which agents should handle the following user query:
"{query}"

IMPORTANT:
- Respond ONLY with a valid JSON array of strings.
- Each string must be the exact name of an agent (e.g., "rag", "portfolio", "stock", "email", "summarizer").
- Do NOT include any extra text or explanations outside the JSON array.
- Example output: ["rag", "email"]
"""
        logger.info("Sending prompt to Groq for planning...")
        logger.debug(f"Prompt: {prompt}")

        try:
            tools_response = await self.llm.call_json_async(prompt)
            # Normalize response: always a list
            if isinstance(tools_response, dict) and "plan" in tools_response:
                tools = tools_response["plan"]
            elif isinstance(tools_response, list):
                tools = tools_response
            elif isinstance(tools_response, str):
                tools = [tools_response]
            else:
                tools = ["rag"]

            if not tools:
                logger.warning("Groq returned empty list, defaulting to ['rag']")
                tools = ["rag"]

            logger.info(f"Groq selected tools: {tools}")
        except Exception as e:
            logger.error(f"Groq planning failed, defaulting to ['rag']: {e}")
            tools = ["rag"]

        return tools

    def aggregate(self, results: list):
        """
        Combine multiple agent outputs into a single formatted string.
        """
        return "\n\n".join(results)
