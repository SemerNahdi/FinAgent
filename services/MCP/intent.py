# File: intent.py
# Contains QueryIntent class for analyzing query intents

from typing import Dict
from collections import defaultdict
from .enums import AgentType  # Assuming enums.py is in the same package


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
                "Proxy Statements",
                "Annual Reports",
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

        # Count matches for each agent type
        for agent_type, keywords in cls.INTENT_KEYWORDS.items():
            strong_matches = sum(1 for kw in keywords["strong"] if kw in query_lower)
            weak_matches = sum(1 for kw in keywords["weak"] if kw in query_lower)

            # Boost score for multiple strong matches
            if strong_matches > 0:
                scores[agent_type] += 0.7 * strong_matches + (
                    0.2 * (strong_matches - 1)
                )
            if weak_matches > 0:
                scores[agent_type] += 0.3 * weak_matches

        # Special handling for stock queries (look for ticker symbols or stock names)
        if any(
            word in query_lower
            for word in ["stock", "price", "ticker", "quote", "shares"]
        ):
            # Check for common stock names
            common_stocks = [
                "tesla",
                "apple",
                "microsoft",
                "google",
                "amazon",
                "meta",
                "nvidia",
                "tsla",
                "aapl",
                "msft",
                "googl",
                "amzn",
                "nvda",
            ]
            if any(stock in query_lower for stock in common_stocks):
                scores[AgentType.STOCK] = max(scores[AgentType.STOCK], 0.8)

        # Special handling for portfolio queries
        if any(
            word in query_lower
            for word in ["my", "holdings", "portfolio", "top", "allocation"]
        ):
            scores[AgentType.PORTFOLIO] = max(scores[AgentType.PORTFOLIO], 0.6)

        # Special handling for email queries
        if any(
            word in query_lower
            for word in ["send", "email", "mail", "snapshot", "report", "daily"]
        ):
            scores[AgentType.EMAIL] = max(scores[AgentType.EMAIL], 0.7)

        # Special handling for news queries
        if any(
            word in query_lower
            for word in ["news", "latest", "recent", "update", "headlines"]
        ):
            scores[AgentType.WEBSEARCH] = max(scores[AgentType.WEBSEARCH], 0.7)

        return {k: min(v, 1.0) for k, v in scores.items()}
