"""
knowledge_builder.py

Builds the RAG knowledge base (SQuAD + a curated set of Wikipedia topics,
chunked and embedded into ChromaDB). Runs automatically on every server
startup, since Render's free tier wipes local files (including
chroma_db/) on every redeploy or spin-down/spin-up cycle.

Wikipedia coverage is fetched by explicit topic title via Wikipedia's API
rather than streaming a random slice of the full dump — this is both much
faster (a couple dozen direct API calls vs. streaming through thousands
of dump rows) and guarantees that common general-knowledge questions
(e.g. "What is AI?") are actually answerable, rather than depending on
whether the right article happened to land in a random slice.
"""

import os
import shutil
import itertools

import numpy as np
import requests

import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 100

SQUAD_ROW_LIMIT = int(os.getenv("SQUAD_ROW_LIMIT", "5000"))

# Curated Wikipedia topics fetched directly by title via Wikipedia's API,
# instead of streaming through a random slice of the dump and hoping the
# right articles happen to be included. This is both faster (a couple
# dozen direct API calls vs. streaming thousands of dump rows) and more
# reliable — it guarantees coverage of common general-knowledge questions
# (like "What is AI?") rather than leaving it to chance.
CURATED_WIKI_TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Computer science",
    "Natural language processing",
    "Neural network",
    "Photosynthesis",
    "Prime number",
    "Climate change",
    "World War II",
    "Solar System",
    "DNA",
    "Democracy",
    "Evolution",
    "Internet",
    "Renewable energy",
    "Quantum mechanics",
    "Human brain",
    "Economics",
    "William Shakespeare",
    "Ancient Rome",
    "Mount Everest",
    "Python (programming language)",
    "Photograph",
    "Gravity",
    "The Solar System",
]

_model = None

# One shared ChromaDB client for the whole app — both this file (building
# the knowledge base) and retrieval_agent.py (querying it) use this exact
# same connection. Previously each created its own separate client, which
# could fail to see the other's writes reliably (two connections to the
# same on-disk store, opened at different times). Sharing one connection
# avoids that entirely.
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_model():
    """Loads the embedding model once and reuses it across the app,
    since it's also used by retrieval_agent.py at query time."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _fetch_wikipedia_article(title: str) -> str:
    """Fetches the full plain-text extract of one Wikipedia article by
    title via Wikipedia's public API. Returns an empty string if the
    article isn't found or the request fails, rather than raising —
    a single missing/renamed topic shouldn't break the whole build."""
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
            },
            headers={"User-Agent": "ai-response-evaluator/1.0"},
            timeout=15,
        )
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except Exception as e:
        print(f"  Warning: failed to fetch Wikipedia article '{title}': {type(e).__name__}")
    return ""


def _fetch_curated_wikipedia_articles() -> list[str]:
    print(f"Fetching {len(CURATED_WIKI_TOPICS)} curated Wikipedia articles...")
    texts = []
    for i, title in enumerate(CURATED_WIKI_TOPICS, start=1):
        extract = _fetch_wikipedia_article(title)
        if extract:
            texts.append(extract)
        print(f"  Wikipedia: {i}/{len(CURATED_WIKI_TOPICS)} ({title})")
    return texts


def build_knowledge_base(force_rebuild: bool = False):
    """
    Builds (or reuses) the ChromaDB knowledge base. If the collection
    already has data and force_rebuild is False, this is a no-op — this
    matters locally, where you don't want to re-download and re-embed
    everything on every restart, but on Render, the collection is always
    empty on a fresh container, so this runs fresh every time there.
    """
    if force_rebuild and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    client = _chroma_client
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Knowledge base already populated with {collection.count()} chunks — skipping rebuild.")
        return collection

    print("Building knowledge base...")

    print(f"Loading SQuAD (first {SQUAD_ROW_LIMIT} rows)...")
    # streaming=True here too — non-streaming loads were caching far more
    # data to disk than needed for a small slice, which caused a separate
    # "temporary storage exceeded" failure during deploy
    squad_stream = load_dataset("rajpurkar/squad", split="train", streaming=True)

    squad_rows = []
    for i, row in enumerate(itertools.islice(squad_stream, SQUAD_ROW_LIMIT), start=1):
        squad_rows.append(row)
        if i % 5000 == 0 or i == SQUAD_ROW_LIMIT:
            print(f"  SQuAD: {i}/{SQUAD_ROW_LIMIT} rows loaded")

    squad_contexts = list(set(row["context"] for row in squad_rows))

    wiki_texts = _fetch_curated_wikipedia_articles()

    all_documents = squad_contexts + wiki_texts

    all_chunks = []
    for doc in all_documents:
        all_chunks.extend(_chunk_text(doc))

    print(f"Embedding {len(all_chunks)} chunks...")
    model = get_embedding_model()

    # Manual batching with explicit progress prints — sentence-transformers'
    # built-in tqdm progress bar often doesn't render cleanly in Render's
    # log viewer (it relies on carriage-return updates, which get lost in
    # a piped/non-interactive log stream), making a long embedding step
    # look stuck even when it's progressing normally.
    EMBED_BATCH_SIZE = 64  # raised from 16 now that Standard tier has headroom
    embeddings_list = []
    for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[i:i + EMBED_BATCH_SIZE]
        batch_embeddings = model.encode(batch, batch_size=EMBED_BATCH_SIZE)
        embeddings_list.extend(batch_embeddings)
        done = min(i + EMBED_BATCH_SIZE, len(all_chunks))
        if done % 1000 < EMBED_BATCH_SIZE or done == len(all_chunks):
            print(f"  Embedded {done}/{len(all_chunks)} chunks")

    embeddings = np.array(embeddings_list)

    BATCH_SIZE = 5000
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        collection.add(
            ids=[str(j) for j in range(i, i + len(batch_chunks))],
            embeddings=batch_embeddings.tolist(),
            documents=batch_chunks,
        )
        print(f"  Indexed {i + len(batch_chunks)}/{len(all_chunks)} chunks")

    print(
        f"Knowledge base built: {len(squad_contexts)} SQuAD contexts, "
        f"{len(wiki_texts)} Wikipedia articles, {len(all_chunks)} total chunks."
    )
    return collection


if __name__ == "__main__":
    build_knowledge_base(force_rebuild=True)