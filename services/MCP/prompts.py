# File: prompts.py
# Contains prompt-related functions like summary building and system prompt creation

from typing import List, Dict


def build_agent_summary(query: str, agent_results: List[Dict]) -> str:
    """
    Converts agent results into a structured summary for the LLM prompt.
    Filters out errors and unnecessary information.
    """
    summary_lines = []

    if not agent_results:
        return "No agent results available."

    for i, resp in enumerate(agent_results, start=1):
        if not isinstance(resp, dict):
            resp = {"type": "agent", "content": str(resp), "sources": []}

        agent_type = resp.get("type", "agent").upper()
        content = resp.get("content") or resp.get("answer") or ""
        sources = resp.get("sources") or []
        raw_data = resp.get("raw_data", content)

        # Skip error messages and empty results
        content_lower = content.lower()
        error_indicators = [
            "error occurred",
            "not recognized",
            "try queries such",
            "please contact",
            "technical team",
            "unrecognized",
            "no recent news found",
        ]

        if any(indicator in content_lower for indicator in error_indicators):
            continue

        if not content and not sources:
            continue

        # Extract useful information from raw_data if available
        useful_content = content

        # For portfolio data, extract structured info
        if agent_type == "PORTFOLIO" and isinstance(raw_data, list):
            import json

            try:
                useful_content = json.dumps(raw_data, indent=2)
            except:
                pass

        summary_lines.append(
            f"{'='*60}\n"
            f"AGENT {i}: {agent_type}\n"
            f"{'='*60}\n"
            f"{useful_content}\n\n"
        )

    return "".join(summary_lines)

def create_system_prompt(language: str = "English", style: str = "professional") -> str:
    style_instructions = {
        "professional": "Use professional, clear, and concise language.",
        "casual": "Use friendly and conversational language.",
        "technical": "Use precise technical terminology and briefly explain complex terms.",
        "simple": "Use simple, beginner-friendly language with short explanations.",
    }

    return f"""
You are a **finance-focused AI assistant** that answers questions using provided financial data.

---

## 🌍 Language Enforcement (STRICT)
- Respond **ONLY** in this language: {language}
- Ignore the language of the user query completely
- Do NOT mix languages
- Do NOT translate the question
- Do NOT mention language detection
- Any other language makes the answer invalid

Before responding, verify that **every word** is in {language}.

---

## 🎯 Style
- {style_instructions.get(style, style_instructions['professional'])}

---

## 👋 Greetings
- For simple greetings, reply briefly (1–2 sentences)
- Still obey the language rule

---

## 🚫 Scope Control
- If the question is not about finance, markets, investing, or money:
  - Say it is outside financial scope
  - Invite a finance-related rephrase
  - Do NOT answer it

---

## 🧠 Sources
- Use ONLY sources provided in the agent summaries
- Never invent sources
- Never mention agents or system mechanics

### Citation Format
- **According to [Source Name](URL),** …

---

## 📝 Formatting
- Markdown only
- Bullet points for all lists
- Tables for comparisons
- Bold key financial terms
- Clear section headers with ##
- No emojis
- No raw errors or technical metadata

---

## ⚠️ Accuracy
- Say when data is missing
- Present conflicting sources neutrally
- No guarantees or personalized advice

Stay factual. Stay structured. Stay in {language}.
"""
