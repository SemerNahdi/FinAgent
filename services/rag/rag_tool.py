# services/rag/rag_tool.py
from pathlib import Path
from typing import List, Dict
import hashlib

from dotenv import load_dotenv
from services.rag.parser import parse_file, list_files
from services.rag.chunking import chunk_content
from services.rag.embeddings import FaissIndex
from groq import Groq
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

class RAGTool:
    def __init__(
        self,
        data_dir: str = "./data",
        index_dir: str = "./data/finance_agent_index",
        model: str = DEFAULT_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        groq_api_key: str = None,
    ):
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))

        # Load existing index or create new
        if self.index_dir.exists() and any(self.index_dir.iterdir()):
            self.index = FaissIndex.load(self.index_dir, model_name=self.model)
        else:
            self.index = FaissIndex(model_name=self.model)

    def _safe_chunk_folder(self, file_name: str) -> Path:
        short_name = file_name[:20].replace(" ", "_")
        folder_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
        return self.data_dir / "chunks" / f"{short_name}_{folder_hash}"

    def add_file(self, file_path: str):
        """Parse, chunk, and embed a single file."""
        f = Path(file_path)
        if not f.exists() or f.suffix.lower() not in [".pdf", ".csv", ".json"]:
            print(f"⏭️ Skipped {f.name} (unsupported type)")
            return

        doc_id = f.stem
        if doc_id in {meta.get("doc_id") for meta in self.index.metadatas}:
            print(f"⚠️ Already ingested {f.name}")
            return

        text = parse_file(f)
        if not text:
            print(f"⚠️ Empty file: {f.name}")
            return

        chunks = chunk_content(text, self.chunk_size, self.chunk_overlap)
        folder = self._safe_chunk_folder(f.name)
        folder.mkdir(parents=True, exist_ok=True)

        metadatas = []
        for i, chunk in enumerate(chunks):
            safe_name = "".join(c if c.isalnum() else "_" for c in f.name)
            chunk_file = folder / f"{safe_name}_chunk{i+1}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")
            metadatas.append({
                "doc_id": doc_id,
                "chunk_id": i,
                "meta": {"source": f.name},
                "content": chunk
            })

        self.index.add(chunks, metadatas)
        self.index.save(self.index_dir)
        print(f"✅ Ingested {f.name}: {len(chunks)} chunks")

    def add_directory(self, data_dir: str = None):
        """Ingest all files from a directory."""
        data_dir = Path(data_dir or self.data_dir)
        for f in list_files(data_dir, exts=["pdf", "csv", "json"]):
            self.add_file(f)

    def query(self, q: str, top_k: int = 5):
        """Retrieve top chunks and query LLM."""
        hits = self.index.search(q, k=top_k)
        prompt = self._build_prompt(q, hits)
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(self, query: str, retrieved: List[Dict]) -> str:
        parts = ["You are a helpful financial assistant. Use the sources to answer concisely.", "\n\n---\nSources:\n"]
        for i, hit in enumerate(retrieved):
            md = hit["metadata"]
            src = md.get("meta", {}).get("source", md.get("doc_id", "unknown"))
            snippet = md.get("content", "")[:1200]
            parts.append(f"[{i}] Source: {src}  (score: {hit['score']:.3f})\nChunk ID: {md.get('chunk_id')}\nContent: {snippet}\n\n")
        parts.append(f"\nQuestion:\n{query}\n\nAnswer:")
        return "\n".join(parts)
