# services/rag/rag_tool.py
from typing import List, Dict
from pathlib import Path
import os
import hashlib

from groq import Groq

from services.rag.parser import parse_file, list_files
from services.rag.chunking import chunk_content
from services.rag.embeddings import FaissIndex
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configurable paths
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "./data"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "./data/finance_agent_index"))
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_SAVE_DIR = DATA_DIR / "chunks"
CHUNK_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_chunk_folder(file_name: str) -> Path:
    """Generate a safe folder name using hash to avoid Windows path issues."""
    short_name = file_name[:20].replace(" ", "_")
    folder_hash = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    return CHUNK_SAVE_DIR / f"{short_name}_{folder_hash}"


def ingest_new_files(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR):
    """
    Incremental ingestion: only new files not already in FAISS index are processed.
    """
    supported_exts = ["pdf", "csv", "json"]
    files = list_files(data_dir, exts=supported_exts)

    # Load existing index or create new
    if index_dir.exists() and any(index_dir.iterdir()):
        index = FaissIndex.load(index_dir, model_name=DEFAULT_MODEL)
        ingested_ids = {meta.get("doc_id") for meta in index.metadatas}
        print(f"📂 Loaded existing FAISS index with {len(ingested_ids)} documents")
    else:
        index = FaissIndex(model_name=DEFAULT_MODEL)
        ingested_ids = set()
        print("🆕 Created new FAISS index")

    new_docs = 0
    new_chunks = 0

    for f in files:
        doc_id = f.stem
        if doc_id in ingested_ids:
            print(f"⚠️ Skipping already ingested file: {f.name}")
            continue

        try:
            text = parse_file(f)
            if not text:
                print(f"⚠️ Skipping empty file: {f.name}")
                continue

            # Chunk document
            doc_chunks = chunk_content(
                text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
            )
            if not doc_chunks:
                print(f"⚠️ No chunks generated for: {f.name}")
                continue

            # Save chunks
            folder = _safe_chunk_folder(f.name)
            folder.mkdir(parents=True, exist_ok=True)
            for i, chunk in enumerate(doc_chunks):
                safe_name = "".join(c if c.isalnum() else "_" for c in f.name)
                chunk_file = folder / f"{safe_name}_chunk{i+1}.txt"
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    cf.write(chunk)

            # Add chunks to index
            metadatas = [
                {
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "meta": {"source": f.name},
                    "content": chunk,
                }
                for i, chunk in enumerate(doc_chunks)
            ]

            index.add(doc_chunks, metadatas)

            print(f"✅ Ingested {f.name}: {len(doc_chunks)} chunks")
            new_docs += 1
            new_chunks += len(doc_chunks)

        except Exception as e:
            print(f"❌ Failed to ingest {f.name}: {e}")

    # Save FAISS index after processing new files only
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save(index_dir)
    print(f"💾 FAISS index updated: {new_docs} new docs, {new_chunks} new chunks")
    return {"new_docs": new_docs, "new_chunks": new_chunks}


# ------------------- Retrieval -------------------
def load_index(index_dir: Path = INDEX_DIR):
    if index_dir.exists() and any(index_dir.iterdir()):
        return FaissIndex.load(index_dir, model_name=DEFAULT_MODEL)
    else:
        print("⚠️ No FAISS index found. Creating empty index.")
        return FaissIndex(model_name=DEFAULT_MODEL)


def retrieve(query: str, top_k: int = 5, index_dir: Path = INDEX_DIR):
    index = load_index(index_dir)
    return index.search(query, k=top_k)


def build_prompt(
    query: str, retrieved: List[Dict], system_instructions: str = None
) -> str:
    system = (
        system_instructions
        or "You are a helpful financial assistant. Use the sources to answer concisely."
    )

    parts = [system, "\n\n---\nSources:\n"]

    for i, hit in enumerate(retrieved):
        md = hit["metadata"]
        src = md.get("meta", {}).get("source", md.get("doc_id", "unknown"))
        chunk_text = md.get("content", "")

        # Keep chunks reasonable size
        snippet = chunk_text[:1200]

        parts.append(
            f"[{i}] Source: {src}  (score: {hit['score']:.3f})\n"
            f"Chunk ID: {md.get('chunk_id')}\n"
            f"Content: {snippet}\n\n"
        )

    parts.append("\nQuestion:\n" + query + "\n\nAnswer:")
    return "\n".join(parts)


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(
    prompt: str, model: str = "llama-3.3-70b-versatile", max_tokens: int = 512
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def answer_query(
    query: str,
    top_k: int = 5,
    system_instructions: str = None,
    index_dir: Path = INDEX_DIR,
):
    hits = retrieve(query, top_k=top_k, index_dir=index_dir)
    prompt = build_prompt(query, hits, system_instructions=system_instructions)
    resp = call_llm(prompt)
    return {"query": query, "prompt": prompt, "answer": resp, "retrieved": hits}


# ------------------- CLI -------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ingest", action="store_true", help="Ingest new files incrementally"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive query mode"
    )
    args = parser.parse_args()

    # Ingest first if requested
    if args.ingest:
        print("Incremental ingestion starting...")
        stats = ingest_new_files()
        print(stats)

    # Interactive multi-query loop
    if args.interactive:
        index_loaded = False
        print("Entering interactive query mode. Type 'exit' to quit.\n")
        while True:
            query = input("Enter your question: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Exiting interactive mode.")
                break
            out = answer_query(query)
            print("\nPROMPT:")
            print(out["prompt"][:2000])
            print("\nANSWER:")
            print(out["answer"])
            print("\n" + "-" * 60 + "\n")


# python -m services.rag.rag_tool --ingest
# python -m services.rag.rag_tool --ingest --interactive
