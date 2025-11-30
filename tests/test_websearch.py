# tests/test_websearch.py
import pytest
from services.tools import websearch_tool
from services.agents.websearch_agent import WebSearchAgent


# python -m pytest tests/test_websearch.py -vv
# -----------------------------
# Tests for the tool
# -----------------------------
def test_search_financial_news_success(monkeypatch):
    # Mock requests.get to avoid real API calls
    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "articles": [
                    {
                        "title": "Stock rises",
                        "description": "Stock XYZ rises 5%",
                        "url": "https://example.com/1",
                    },
                    {
                        "title": "Market update",
                        "description": "",
                        "url": "https://example.com/2",
                    },
                ]
            }

    monkeypatch.setattr(
        websearch_tool.requests, "get", lambda url, params: MockResponse()
    )

    result = websearch_tool.search_financial_news("stock XYZ")
    assert "- Stock XYZ rises 5%" in result
    assert "[Source 1](https://example.com/1)" in result


def test_search_financial_news_no_articles(monkeypatch):
    class MockResponse:
        status_code = 200

        def json(self):
            return {"articles": []}

    monkeypatch.setattr(
        websearch_tool.requests, "get", lambda url, params: MockResponse()
    )

    result = websearch_tool.search_financial_news("nonexistent query")
    assert result == "No recent news found for your query."


def test_search_financial_news_api_fail(monkeypatch):
    class MockResponse:
        status_code = 500

    monkeypatch.setattr(
        websearch_tool.requests, "get", lambda url, params: MockResponse()
    )

    result = websearch_tool.search_financial_news("anything")
    assert result == "Could not fetch news at this time."


# -----------------------------
# Tests for the agent
# -----------------------------
def test_websearch_agent(monkeypatch):
    # Mock the underlying tool
    monkeypatch.setattr(
        "services.agents.websearch_agent.search_financial_news",
        lambda query: "- Mock news\n\n[Source 1](https://mock.com)",
    )
    agent = WebSearchAgent()
    result = agent.run("Tesla news")
    assert "- Mock news" in result
    assert "[Source 1](https://mock.com)" in result
