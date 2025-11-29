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
        """Sync RAG execution."""
        hits = self.rag.index.search(query, k=top_k)
        prompt = self.rag._build_prompt(query, hits)

        response = self.rag.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    async def run_async(self, query: str, top_k: int = 3):
        """Async wrapper so MCPAgent can await this agent."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, query, top_k)

    def get_portfolio_summary(self):
        """MCPAgent may call this. RAG returns empty."""
        return []
