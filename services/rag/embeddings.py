# backend/services/rag/embeddings.py
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import pickle
from dotenv import load_dotenv

# Load .env once
load_dotenv()


class FaissIndex:
    def __init__(self, model_name: Optional[str] = None, dim: Optional[int] = None):
        # Use constructor arg > .env > hardcoded default
        model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.dim = dim or self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadatas = []

    def _encode(self, texts: List[str]):
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms
        return embs.astype("float32")

    def add(self, texts: List[str], metadatas: List[Dict]):
        embs = self._encode(texts)
        self.index.add(embs)
        self.metadatas.extend(metadatas)

    def search(self, query: str, k: int = 5):
        q_emb = self._encode([query])
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append({"score": float(score), "metadata": self.metadatas[idx]})
        return results

    # ------------------------------
    # Save and Load methods
    # ------------------------------
    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadatas.pkl", "wb") as fh:
            pickle.dump(self.metadatas, fh)

    @classmethod
    def load(cls, path: str, model_name: Optional[str] = None):
        p = Path(path)
        # Use constructor arg > .env > default
        model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        inst = cls(model_name=model_name)
        inst.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "metadatas.pkl", "rb") as fh:
            inst.metadatas = pickle.load(fh)
        return inst
