# services/rag/rag_agent.py
import asyncio
from services.rag.rag_tool import RAGTool


class RAGAgent:
    def __init__(
        self,
        data_dir: str = "./data",
        index_dir: str = "./data/finance_agent_index",
        model: str = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        groq_api_key: str = None,
    ):
        self.rag = RAGTool(
            data_dir=data_dir,
            index_dir=index_dir,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            groq_api_key=groq_api_key,
        )

    # services/rag/rag_agent.py


import asyncio
from services.rag.rag_tool import RAGTool


class RAGAgent:
    def __init__(
        self,
        data_dir: str = "./data",
        index_dir: str = "./data/finance_agent_index",
        model: str = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        groq_api_key: str = None,
    ):
        self.rag = RAGTool(
            data_dir=data_dir,
            index_dir=index_dir,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            groq_api_key=groq_api_key,
        )

    def run(self, query: str, top_k: int = 3):
        """Sync RAG execution with sources returned."""
        # Retrieve top hits
        hits = self.rag.index.search(query, k=top_k)
        prompt = self.rag._build_prompt(query, hits)

        # Extract sources from hits
        sources = []
        for i, hit in enumerate(hits):
            md = hit.get("metadata", {})
            meta_info = md.get("meta", {})
            source_name = meta_info.get("source", md.get("doc_id", "Unknown"))
            content_snippet = md.get("content", "")[:200] + (
                "..." if len(md.get("content", "")) > 200 else ""
            )
            sources.append(
                {
                    "source": source_name,
                    "content": content_snippet,
                    "score": hit.get("score", 0),
                }
            )

        # Query LLM
        response = self.rag.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()

        # Fallback if no sources
        if not sources:
            sources.append(
                {
                    "source": "LLM Generated",
                    "content": answer[:200] + ("..." if len(answer) > 200 else ""),
                    "score": 0,
                }
            )

        # Print sources to terminal with emojis
        print(f"\n🟣 [RAG Agent] Sources for query: '{query}'")
        for i, src in enumerate(sources, start=1):
            print(f"  📄 {i}. {src['source']} (score: {src['score']})")

        return {"answer": answer, "sources": sources, "query": query}

    async def run_async(self, query: str, top_k: int = 3):
        """Async wrapper so MCPAgent can await this agent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query, top_k)

    def get_portfolio_summary(self):
        """MCPAgent may call this. RAG returns empty."""
        return []

    async def run_async(self, query: str, top_k: int = 3) -> dict:
        """Async wrapper so MCPAgent can await this agent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query, top_k)

    def get_portfolio_summary(self):
        """MCPAgent may call this. RAG returns empty."""
        return []
