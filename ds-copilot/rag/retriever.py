import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

INDEX_DIR = os.path.join("rag", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"  # must match the model used in build_index.py


class KnowledgeRetriever:
    """
    Loads the FAISS index + chunk store built by build_index.py and retrieves
    the most relevant pandas/debugging knowledge chunks for a given query
    (e.g. an error message or a natural-language question).
    """

    def __init__(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                f"Index not found at {INDEX_PATH}. Run build_index.py first "
                "(from the project root, so the 'rag/index' path resolves correctly)."
            )

        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, k: int = 3) -> list:
        """
        Returns the top-k most relevant knowledge chunks (as plain strings)
        for the given query, ordered by relevance (closest first).
        Uses L2 distance since build_index.py builds an IndexFlatL2 index.
        """
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