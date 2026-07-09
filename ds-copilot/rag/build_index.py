import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

CORPUS_PATH = os.path.join("docs_corpus", "pandas_knowledge.txt")
INDEX_DIR = os.path.join("rag", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality embedding model


def load_and_chunk_corpus(path: str) -> list:
    """
    Splits the corpus into chunks based on 'TOPIC:' markers,
    so each chunk is a self-contained knowledge unit.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = content.split("TOPIC:")
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            chunks.append("TOPIC:" + chunk)

    return chunks


def build_index():
    print("Loading corpus...")
    chunks = load_and_chunk_corpus(CORPUS_PATH)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved to {INDEX_PATH}")
    print(f"Chunks saved to {CHUNKS_PATH}")
    print("Done!")


if __name__ == "__main__":
    build_index()