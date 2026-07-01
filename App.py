"""
app.py — Streamlit chatbot over the UNIZULU Richards Bay documents.

Run locally:
    streamlit run app.py

Requires:
    - data/processed/index.pkl  (built by build_index.py)
    - ANTHROPIC_API_KEY set in .env (local) or Streamlit secrets (cloud)
"""

import os
import pickle
from pathlib import Path

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import anthropic

load_dotenv()

INDEX_PATH = Path("data/processed/index.pkl")
TOP_K = 6
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a helpful assistant for engineering students at the \
UNIZULU Richards Bay campus. Answer questions ONLY using the provided context \
chunks from the official lecture timetable, general calendar, and Engineering \
Department handbook. If the answer isn't in the context, say so clearly and \
suggest the student check with the department directly — do not guess or \
invent module codes, dates, times, or venues. When you give an answer that \
involves a specific class, date, or venue, name the source document you took \
it from. Keep answers concise and student-friendly."""


def get_api_key():
    # Streamlit Cloud: st.secrets. Local: .env / environment variable.
    if "ANTHROPIC_API_KEY" in st.secrets if hasattr(st, "secrets") else False:
        return st.secrets["ANTHROPIC_API_KEY"]
    return os.environ.get("ANTHROPIC_API_KEY")


@st.cache_resource(show_spinner=False)
def load_index():
    if not INDEX_PATH.exists():
        return None
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def retrieve(index, query: str, k: int = TOP_K):
    vec = index["vectorizer"].transform([query])
    sims = cosine_similarity(vec, index["matrix"]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        if sims[i] <= 0:
            continue
        chunk = index["chunks"][i]
        results.append({**chunk, "score": float(sims[i])})
    return results


def build_prompt(question: str, context_chunks: list) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']} — {c['location']}]\n{c['text']}"
        for c in context_chunks
    )
    return (
        f"Context from official UNIZULU documents:\n\n{context_block}\n\n"
        f"---\n\nStudent question: {question}"
    )


def main():
    st.set_page_config(page_title="UNIZULU Richards Bay Assistant", page_icon="🎓")
    st.title("🎓 UNIZULU Richards Bay — Engineering Assistant")
    st.caption(
        "Answers are grounded in the 2026 lecture timetable, the 2026 general "
        "calendar, and the Engineering Department handbook."
    )

    index = load_index()
    if index is None:
        st.error(
            "No index found. Run `python extract_data.py` then "
            "`python build_index.py` before starting the app."
        )
        st.stop()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found. Add it to a local .env file, or to "
            "Streamlit Cloud's Secrets if deployed."
        )
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "Ask about lecture times, venues, exam dates, module requirements..."
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                chunks = retrieve(index, question)

            if not chunks:
                answer = (
                    "I couldn't find anything relevant to that in the timetable, "
                    "calendar, or handbook. Try rephrasing, or check with the "
                    "department directly."
                )
                st.markdown(answer)
            else:
                prompt = build_prompt(question, chunks)
                with st.spinner("Thinking..."):
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=1000,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    )
                answer = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                st.markdown(answer)

                with st.expander("Sources used"):
                    for c in chunks:
                        st.markdown(
                            f"**{c['source']}** — {c['location']} "
                            f"(relevance {c['score']:.2f})"
                        )
                        st.caption(c["text"][:300] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
