# services/rag/indexing.py (updated to include ingest_file and load existing index)
import os
from typing import Dict
from services.rag.parser import parse_file
from services.rag.chunking import chunk_content
from services.rag.embeddings import FaissIndex as EmbeddingsManager

CHUNK_SAVE_DIR = "data/chunks"  # folder to save chunks


def ingest_file(file_path: str, em: EmbeddingsManager):
    """
    Ingest a single file, save chunks, and add to embeddings.
    Skips if already ingested.
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1][1:].lower()
    if ext not in ["pdf", "csv", "json"]:
        print(f"⏭️ Skipped {file_name} (unsupported type: {ext})")
        return

    # Skip if already ingested
    ingested_files = {meta["file"] for meta in em.metadatas}
    if file_name in ingested_files:
        print(f"✅⏭️ {file_name} already ingested, skipping...")
        return

    try:
        # Parse
        content = parse_file(file_path)
        if not content:
            print(f"⚠️ {file_name} parsed content is empty, skipping...")
            return
        print(f"📄 [PARSE] {file_name} parsed successfully")

        # Chunk
        chunks = chunk_content(content)
        if not chunks:
            print(f"⚠️ {file_name} generated no chunks, skipping...")
            return
        print(f"✂️ [CHUNK] {file_name} created {len(chunks)} chunks")

        # Save chunks
        save_path = os.path.join(os.getcwd(), CHUNK_SAVE_DIR, file_name)  # full path
        os.makedirs(save_path, exist_ok=True)

        for i, chunk in enumerate(chunks):
            # replace invalid filename chars if any
            safe_file_name = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in file_name
            )
            chunk_file = os.path.join(save_path, f"{safe_file_name}_chunk{i+1}.txt")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)

        print(f"💾 [CHUNKS] {len(chunks)} chunks saved to {save_path}")

        # Embeddings
        metadata = [
            {"doc_id": file_name, "chunk_id": i, "source": file_name, "content": chunk}
            for i, chunk in enumerate(chunks)
        ]

        em.add(chunks, metadata)
        print(f"🧠 [EMBEDDING] {file_name} embeddings added")

    except Exception as e:
        print(f"❌ [FAILURE] {file_name}: {str(e)}")
        raise  # Re-raise for test_rag to catch


def ingest_directory(data_dir: str, index_path: str = "faiss_index"):
    """
    Ingest all supported files in a directory, save chunks, and update embeddings.
    Only new files are processed.
    """
    # Load existing index if it exists, else create new
    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        em = EmbeddingsManager.load(index_path)
        print(f"📂 Loaded existing FAISS index from {index_path}")
    else:
        em = EmbeddingsManager()
        print(f"🆕 Created new FAISS index")

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        ingest_file(file_path, em)

    # Save the FAISS index after all new files
    os.makedirs(index_path, exist_ok=True)
    em.save(index_path)
    print(f"💾 [SAVE] FAISS index saved to {index_path}")
