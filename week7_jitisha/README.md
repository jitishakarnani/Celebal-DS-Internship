# Document Question Answering System (RAG)

## How to run

1. Open `RAG_Document_QA.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
2. Run the first cell to install dependencies.
3. In the **Configuration** cell:
   - Set `PDF_PATH` to your PDF file (notes, resume, research paper, book — anything).
   - Set `GEMINI_API_KEY` to your key from https://aistudio.google.com/app/apikey (free tier works fine).
4. Run all cells top to bottom.
5. In the "Run It" section, ask questions about your document with `rag.ask("your question")`.

## What it does

Implements the full RAG pipeline: PDF ingestion → text chunking → embedding creation
(`sentence-transformers`, local) → vector storage (FAISS) → query embedding → context
retrieval (top-k similarity search) → grounded answer generation (Gemini 2.5 Flash).

## Files

- `RAG_Document_QA.ipynb` — the full, runnable notebook (submit this)
- `README.md` — this file

## Notes for submission

- No internet-hosted vector DB needed (FAISS runs locally) — nothing to set up beyond the two pip
  installs and a Gemini API key.
- The embedding model downloads once (~80MB) the first time you run the notebook, so make sure
  you have an internet connection when you first execute it.
