# services/tools/websearch_tool.py
import os
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def search_financial_news(query: str, max_results=5):
    """
    Search for financial news using NewsAPI.
    Returns a summary of top articles with sources.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return "Could not fetch news at this time."

    data = response.json()
    articles = data.get("articles", [])
    if not articles:
        return "No recent news found for your query."

    # Build summary
    summary_lines = []
    sources = []
    for article in articles:
        title = article.get("title", "No title")
        desc = article.get("description", "")
        url = article.get("url", "")
        summary_lines.append(f"- {desc or title}")
        sources.append(url)

    summary_text = "\n".join(summary_lines)
    sources_text = "\n".join(
        [f"[Source {i+1}]({url})" for i, url in enumerate(sources)]
    )

    return (
        f"**Latest news highlights:**\n{summary_text}\n\n**Sources:**\n{sources_text}"
    )
