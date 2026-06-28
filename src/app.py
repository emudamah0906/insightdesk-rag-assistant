"""
Streamlit chat UI for InsightDesk.

A clickable front-end over the same RAGPipeline: chat history, the grounded answer,
expandable retrieved sources with scores, and live confidence/latency/provider — the
kind of demo you screen-share to a client to make the value tangible.

Setup:  pip install streamlit
Run:    streamlit run src/app.py
        (then open the URL it prints, usually http://localhost:8501)

Works offline with the mock LLM; set ANTHROPIC_API_KEY (or LLM_PROVIDER=bedrock) for
real generation — the sidebar shows which provider is live.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st
from rag import RAGPipeline, CONF_THRESHOLD
from rerank import rerank
from embeddings import cosine
from llm import active_provider

st.set_page_config(page_title="InsightDesk Assistant", page_icon="🛟", layout="centered")


@st.cache_resource
def get_pipeline(mode: str) -> RAGPipeline:
    return RAGPipeline(mode=mode)


# ---- sidebar / controls
st.sidebar.title("🛟 InsightDesk")
st.sidebar.caption("Enterprise RAG support assistant")
mode = st.sidebar.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"], index=0,
                            help="hybrid = dense + BM25 fused with RRF")
provider = active_provider()
st.sidebar.metric("LLM provider", provider)
if provider == "mock":
    st.sidebar.info("Offline extractive mock. Set ANTHROPIC_API_KEY or "
                    "LLM_PROVIDER=bedrock for real generation.")
st.sidebar.caption(f"Confidence gate: answers below {CONF_THRESHOLD:.2f} cosine escalate to a human.")
if st.sidebar.button("Clear chat"):
    st.session_state.history = []

rag = get_pipeline(mode)
st.sidebar.caption(f"{len(rag.retriever.chunks)} chunks indexed from data/kb/")

st.title("How can I help?")
st.caption("Try: *How long do refunds take?* · *What does a 429 error mean?* · *Do you ship to the EU?*")

if "history" not in st.session_state:
    st.session_state.history = []

# ---- replay history
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ---- new question
q = st.chat_input("Ask about billing, technical, account, or shipping…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering…"):
            res = rag.answer(q, verbose=False)
            # recompute the reranked chunks to show retrieval transparency
            retrieved = rag.retriever.search(q, k=5, mode=mode)
            reranked = rerank(q, retrieved, top_n=3)
            qv = rag.retriever.embedder.embed(q)

        st.markdown(res["answer"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", res["confidence"])
        c2.metric("Latency", f"{res['latency_ms']} ms")
        c3.metric("Sources", len(res["sources"]))

        with st.expander("🔎 Retrieved context (why it answered this)"):
            for ch in reranked:
                st.markdown(f"**[{ch.source}]** · `{ch.id}` · sim={cosine(qv, ch.vec):.3f}")
                st.caption(ch.text)

    st.session_state.history.append({"role": "assistant", "content": res["answer"]})
