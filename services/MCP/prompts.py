# File: prompts.py
# Contains prompt-related functions like summary building and system prompt creation

from typing import List, Dict


def build_agent_summary(query: str, agent_results: List[Dict]) -> str:
    """
    Converts agent results into a structured summary for the LLM prompt.
    Ensures all outputs are dicts and handles missing keys gracefully.
    Formats sources as Markdown links using 'source' key.
    """
    summary_lines = []

    for i, resp in enumerate(agent_results, start=1):
        if not isinstance(resp, dict):
            resp = {"type": "agent", "content": str(resp), "sources": []}

        agent_type = resp.get("type", "agent").upper()
        content = resp.get("content") or resp.get("answer") or ""
        sources = resp.get("sources") or []
        raw_data = resp.get("raw_data", content)

        # Format sources as Markdown links
        if sources:
            sources_text = "\n".join(
                f"[Source {j+1}] [{s.get('source','Unknown')}]({s.get('source','')}) (score {s.get('score',0)})"
                for j, s in enumerate(sources)
            )
        else:
            sources_text = "No sources available."

        summary_lines.append(
            f"{'='*60}\n"
            f"AGENT: {agent_type}\n"
            f"QUERY: {query}\n"
            f"{'='*60}\n\n"
            f"RESPONSE:\n{content}\n{sources_text}\n"
            f"RAW DATA TYPE: {type(raw_data).__name__}\n\n"
        )

    return "".join(summary_lines)


def create_system_prompt(language: str = "English", style: str = "professional") -> str:
    style_instructions = {
        "professional": "Use professional, clear, and concise language.",
        "casual": "Use friendly and conversational language.",
        "technical": "Use precise technical terminology with explanations.",
        "simple": "Use simple language suitable for beginners.",
    }

    return f"""You are a financial assistant providing clear, accurate answers using information from multiple agents.

**Language:** {language}
**Style:** {style_instructions.get(style, style_instructions['professional'])}

**SOURCE HANDLING:**
- Use the sources provided in the agent summaries.
- Mention sources inline naturally, e.g., "According to [RAG](source_url)".
- If information conflicts, present it neutrally and cite all relevant sources.
- Do not include placeholder text like #source.

**RESPONSE STRUCTURE:**
1. Provide a clear, direct answer to the main question.
2. Give additional details if needed, but keep it readable.
3. Include key points in a bullet list if helpful.
4. Provide actionable recommendations if relevant.
5. At the end, list all sources clearly with links.

**FORMATTING:**
- Markdown is optional; respond in normal readable text.
- Use bullets or numbered lists only if it improves clarity.
- Bold or `code` formatting is optional.
- Focus on natural, easy-to-read responses that a human could understand.
"""
