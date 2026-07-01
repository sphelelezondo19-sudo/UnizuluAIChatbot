"""
ask.py — quick command-line test of the chatbot, no Streamlit needed.

Usage:
    python ask.py "When is 5EEE401L scheduled on Monday?"
"""

import os
import pickle
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

INDEX_PATH = Path("data/processed/index.pkl")
TOP_K = 6
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a helpful assistant for engineering students at the \
UNIZULU Richards Bay campus. Answer questions ONLY using the provided context \
chunks from the official lecture timetable, general calendar, and Engineering \
Department handbook. If the answer isn't in the context, say so clearly. \
Cite which source document you used."""


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    if not INDEX_PATH.exists():
        sys.exit("No index found. Run extract_data.py then build_index.py first.")

    with open(INDEX_PATH, "rb") as f:
        index = pickle.load(f)

    vec = index["vectorizer"].transform([question])
    sims = cosine_similarity(vec, index["matrix"]).flatten()
    top_idx = sims.argsort()[::-1][:TOP_K]
    chunks = [index["chunks"][i] for i in top_idx if sims[i] > 0]

    if not chunks:
        print("No relevant context found.")
        return

    context_block = "\n\n".join(
        f"[Source: {c['source']} — {c['location']}]\n{c['text']}" for c in chunks
    )
    prompt = f"Context:\n\n{context_block}\n\n---\n\nQuestion: {question}"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY in your .env file first.")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    print("\n".join(b.text for b in response.content if b.type == "text"))


if __name__ == "__main__":
    main()
