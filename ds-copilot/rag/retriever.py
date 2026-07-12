import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"


class KnowledgeRetriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_PATH}. Run build_index.py first."
            )
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, k=3):
        if not query or not query.strip():
            return []
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.chunks[idx])
        return results


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    test_queries = [
        "KeyError: 'regionn'",
        "how to group and sum a column",
        "TypeError unsupported operand types",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, k=2)
        for r in results:
            print(f"  -> {r[:100]}...")
