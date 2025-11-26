# test_rag.py 
import os
import traceback

# services/rag/test_rag.py
from services.rag.indexing import ingest_file, CHUNK_SAVE_DIR
from services.rag.embeddings import FaissIndex as EmbeddingsManager


DATA_DIR = r"C:\Users\ASUS\Desktop\uni\ING5\Deep Learning\Finance agent\data"
TEST_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")  # data/faiss_index


def test_files_in_directory(data_dir, index_path):
    # Load existing index if it exists, else create new
    index_file = os.path.join(index_path, "index.faiss")
    if os.path.exists(index_file):
        em = EmbeddingsManager.load(index_path)
        print(f"📂 Loaded existing FAISS index from {index_path}")
    else:
        em = EmbeddingsManager()
        print(f"🆕 Created new FAISS index")

    results = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        try:
            ingest_file(file_path, em)
            results.append(f"✅ {file_name}: SUCCESS")
        except Exception as e:
            error_trace = traceback.format_exc()
            results.append(f"❌ {file_name}: FAILURE - {str(e)}\n{error_trace}")

    # Save the FAISS index after all files
    os.makedirs(index_path, exist_ok=True)
    em.save(index_path)
    print(f"💾 [SAVE] FAISS index saved to {index_path}")

    return results


if __name__ == "__main__":
    results = test_files_in_directory(DATA_DIR, TEST_INDEX_PATH)
    print("\n=== SUMMARY ===")
    for res in results:
        print(res)
