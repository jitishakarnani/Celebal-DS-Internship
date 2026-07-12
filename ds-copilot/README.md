# Autonomous Data Science Co-Pilot

An AI agent that lets you upload a dataset, ask questions about it in plain English, and get back charts and insights - automatically. It writes its own Python/Pandas code, runs it in a sandbox, and self-corrects using a RAG pipeline over official Python and Pandas documentation if the generated code fails.

Built as part of a Data Science internship at Celebal Technologies.

## What it does

1. Upload a CSV, Excel (.xlsx), or JSON file through the web UI.
2. Ask a question in plain English - e.g. "What is the total revenue by region?"
3. The agent generates Python/Pandas code for your question using Gemini 2.5 Flash.
4. The code runs in an isolated sandbox (subprocess, timeout-protected).
5. If it fails, a self-healing agent retrieves relevant guidance from a RAG index built over official Python and Pandas documentation, and rewrites the code.
6. You get back a chart and a plain-English insight, in a chat interface.

## Architecture

Question and dataset go into core_agent.py, the main orchestrator. It generates code using Gemini, then sandbox.py runs that code in an isolated subprocess and captures stdout, stderr, and any chart produced. On success, the insight and chart go back to the user. On failure, self_heal.py takes over: it retrieves relevant guidance from rag/retriever.py, which searches a FAISS index built by rag/build_index.py from the docs_corpus folder, then generates corrected code and retries.

Knowledge base (docs_corpus folder) has two sources, both indexed together: pandas_knowledge.txt, a hand-written cheatsheet of common pandas errors like KeyError, TypeError, and groupby usage; and official_docs, which contains official Python and Pandas documentation downloaded and cleaned by docs_corpus/fetch_official_docs.py, covering groupby, merging, missing data, indexing, reshaping, and built-in exceptions.

## Project structure

agent folder contains core_agent.py (main orchestrator), sandbox.py (isolated subprocess code execution), and self_heal.py (RAG-grounded self-correction on failure).
rag folder contains build_index.py (builds the FAISS index) and retriever.py (top-k retrieval at query time).
docs_corpus folder contains pandas_knowledge.txt, fetch_official_docs.py, and the official_docs subfolder.
utils folder contains chart_utils.py and file_loader.py.
app.py is the Streamlit UI. requirements.txt lists dependencies. .env holds the GOOGLE_API_KEY and is not committed.

## Setup

1. Clone and enter the project: cd ds-copilot
2. Create and activate a virtual environment: python -m venv venv, then venv\Scripts\Activate.ps1
3. Install dependencies: pip install -r requirements.txt
4. Add your Gemini API key to a .env file in the project root: GOOGLE_API_KEY=your-key-here. Get a free key at aistudio.google.com/apikey.
5. Download the documentation corpus: python docs_corpus\fetch_official_docs.py
6. Build the RAG index: python rag\build_index.py
7. Run the app: streamlit run app.py

## Tech stack

LLM: Gemini 2.5 Flash, via langchain-google-genai.
Embeddings: all-MiniLM-L6-v2, via sentence-transformers.
Vector store: FAISS.
Sandboxing: Python subprocess, isolated temp directory, timeout-protected.
UI: Streamlit.
Data handling: pandas, openpyxl for Excel, matplotlib and seaborn for charts.

## Example questions to try

- What is the total revenue by region?
- Show me the monthly sales trend over time
- Which category has the highest average revenue?
- Are there any missing values in this dataset?
- Which product generated the most revenue overall?

## Known limitations

- Sandbox execution is process-isolated but not fully sandboxed against malicious code, fine for trusted internal use, not for untrusted public deployment.
- Self-healing retries a maximum of 2 times before surfacing the error to the user.
- Large files over 500MB may hit Streamlit's upload size limit.
