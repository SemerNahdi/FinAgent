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