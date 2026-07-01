"""
build_index.py

Reads data/processed/corpus.jsonl (produced by extract_data.py) and builds a
TF-IDF search index over all chunks, saved to data/processed/index.pkl.

Run:
    python build_index.py
"""

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_PATH = Path("data/processed/corpus.jsonl")
INDEX_PATH = Path("data/processed/index.pkl")


def main():
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"{CORPUS_PATH} not found. Run `python extract_data.py` first."
        )

    chunks = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        raise SystemExit("Corpus is empty — nothing to index.")

    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.9,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": chunks,
        }, f)

    print(f"Indexed {len(chunks)} chunks -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
