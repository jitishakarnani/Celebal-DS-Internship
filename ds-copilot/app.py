import os
import tempfile
from pathlib import Path

import streamlit as st

from agent.core_agent import CoreAgent

st.set_page_config(page_title="Data Science Co-Pilot", page_icon="📊", layout="wide")
st.title("📊 Autonomous Data Science Co-Pilot")
st.caption("Upload a dataset, ask questions in plain English, get charts and insights.")

if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_filename" not in st.session_state:
    st.session_state.loaded_filename = None


def load_dataset(uploaded_file):
    """Saves the uploaded file to disk and initializes a CoreAgent for it."""
    tmp_dir = tempfile.mkdtemp()
    save_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.agent = CoreAgent(save_path)
    st.session_state.loaded_filename = uploaded_file.name
    st.session_state.messages = []


with st.sidebar:
    st.header("1. Upload your dataset")
    uploaded_file = st.file_uploader("CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.loaded_filename:
        with st.spinner("Loading dataset and setting up the agent..."):
            try:
                load_dataset(uploaded_file)
                st.success(f"Loaded {uploaded_file.name}")
            except EnvironmentError as e:
                st.error(str(e))
                st.session_state.agent = None
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                st.session_state.agent = None

    if st.session_state.agent is not None:
        st.divider()
        st.subheader("Dataset preview")
        df = st.session_state.agent.df
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{len(df)} rows x {len(df.columns)} columns")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

if st.session_state.agent is None:
    st.info("👈 Upload a dataset from the sidebar to get started.")
    st.markdown("""
    **Try asking things like:**
    - "What is the total revenue by region?"
    - "Show me the trend over time for sales"
    - "Which category has the highest average value?"
    - "Are there any missing values in this dataset?"
    """)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_bytes"):
                st.image(msg["chart_bytes"])
            if msg.get("code"):
                with st.expander("View generated code"):
                    st.code(msg["code"], language="python")
            if msg.get("healed"):
                st.caption(f"🔧 Self-corrected after {msg['heal_attempts']} attempt(s)")

    question = st.chat_input("Ask a question about your data...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            result = None
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.answer(question)
                except Exception as e:
                    st.error(f"Something went wrong while processing that: {e}")

            if result is not None:
                if result["success"]:
                    response_text = result["insight"] or "Here's what I found:"
                    st.markdown(response_text)

                    if result["chart_bytes"]:
                        st.image(result["chart_bytes"])

                    with st.expander("View generated code"):
                        st.code(result["code"], language="python")

                    if result["healed"]:
                        st.caption(f"🔧 Self-corrected after {result['heal_attempts']} attempt(s)")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "chart_bytes": result["chart_bytes"],
                        "code": result["code"],
                        "healed": result["healed"],
                        "heal_attempts": result["heal_attempts"],
                    })
                else:
                    error_msg = (
                        "I couldn't answer that even after trying to self-correct. "
                        "Could you rephrase the question, or check that it relates to "
                        "columns in your dataset?"
                    )
                    st.error(error_msg)
                    with st.expander("View error details"):
                        st.code(result["stderr"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "chart_bytes": None,
                        "code": result["code"],
                        "healed": False,
                        "heal_attempts": result["heal_attempts"],
                    })
