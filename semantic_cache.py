import numpy as np
import faiss
from datetime import datetime
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings

class SemanticCache:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600
    ):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.cache_entries = []
        embedding_dim = 384
        self.index = faiss.IndexFlatIP(embedding_dim)

    def _get_embedding(self, text: str) -> np.ndarray:
        return np.array(self.embedding_model.embed_query(text), dtype=np.float32)

    def search(self, query: str) -> Optional[dict]:
        if not self.cache_entries:
            return None
        query_emb = self._get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, 1)
        if len(indices[0]) == 0 or indices[0][0] == -1:
            return None
        score = distances[0][0]
        idx = indices[0][0]
        if score >= self.similarity_threshold:
            entry = self.cache_entries[idx]
            if (datetime.now() - entry["timestamp"]).total_seconds() < self.ttl_seconds:
                return entry
        return None

    def add(self, question: str, answer: str, sources: Optional[List[str]] = None):
        self.cache_entries = [e for e in self.cache_entries if e["question"] != question]
        emb = self._get_embedding(question)
        entry = {
            "question": question,
            "answer": answer,
            "embedding": emb,
            "sources": sources or [],
            "timestamp": datetime.now()
        }
        self.index.add(emb.reshape(1, -1))
        self.cache_entries.append(entry)

    def get_stats(self) -> dict:
        return {"total_entries": len(self.cache_entries), "threshold": self.similarity_threshold, "ttl_seconds": self.ttl_seconds}

    def clear(self):
        self.cache_entries = []
        self.index.reset()

    def __len__(self):
        return len(self.cache_entries)