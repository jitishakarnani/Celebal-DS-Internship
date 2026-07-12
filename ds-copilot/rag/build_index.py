import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

CORPUS_PATH = os.path.join("docs_corpus", "pandas_knowledge.txt")
OFFICIAL_DOCS_DIR = os.path.join("docs_corpus", "official_docs")
INDEX_DIR = os.path.join("rag", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_chunks = content.split("TOPIC:")
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            chunks.append("TOPIC:" + chunk)
    return chunks


def load_and_chunk_official_docs(folder, chunk_size=800, overlap=100):
    chunks = []
    if not os.path.isdir(folder):
        return chunks
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                overlap_text = current[-overlap:] if current else ""
                current = (overlap_text + "\n\n" + para).strip() if overlap_text else para
        if current:
            chunks.append(current)
    return chunks


def build_index():
    print("Loading hand-written knowledge base...")
    chunks = load_and_chunk_corpus(CORPUS_PATH)
    print(f"Created {len(chunks)} chunks from pandas_knowledge.txt.")

    print("Loading official documentation...")
    doc_chunks = load_and_chunk_official_docs(OFFICIAL_DOCS_DIR)
    if not doc_chunks:
        print("  No official docs found. Run docs_corpus/fetch_official_docs.py first.")
    print(f"Created {len(doc_chunks)} chunks from official docs.")

    chunks.extend(doc_chunks)
    print(f"Total chunks: {len(chunks)}")

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
