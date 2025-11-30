# services/agents/websearch_agent.py
from services.tools.websearch_tool import search_financial_news


class WebSearchAgent:
    """
    Agent to fetch current financial news and provide human-readable summaries with sources.
    """

    def run(self, query: str):
        """
        Search the web for financial news.
        Returns a summary of top news + sources in bullet points.
        """
        results = search_financial_news(query)
        if not results:
            return "No recent news found for your query."

        return results
