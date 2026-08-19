"""
knowledge_builder.py

Builds the RAG knowledge base (SQuAD + Wikipedia, chunked and embedded
into ChromaDB). Runs automatically on every server startup, since
Render's free tier wipes local files (including chroma_db/) on every
redeploy or spin-down/spin-up cycle.

WIKI_ARTICLE_LIMIT is intentionally small by default (150, down from the
original 1000) so this rebuild finishes quickly on Render's constrained
free-tier CPU. Increase it via the WIKI_ARTICLE_LIMIT environment
variable once persistent storage is in place, if broader coverage is
needed later.
"""

import os
import shutil
import itertools

import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 100

WIKI_ARTICLE_LIMIT = int(os.getenv("WIKI_ARTICLE_LIMIT", "60"))
SQUAD_ROW_LIMIT = int(os.getenv("SQUAD_ROW_LIMIT", "200"))

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
    squad_rows = list(itertools.islice(squad_stream, SQUAD_ROW_LIMIT))
    squad_contexts = list(set(row["context"] for row in squad_rows))

    print(f"Loading {WIKI_ARTICLE_LIMIT} Wikipedia articles...")
    # streaming=True avoids downloading full data shards just to grab a
    # small slice — this was the main source of slow startup, since even
    # requesting only 60 articles via train[:60] could pull down much
    # more data than needed without streaming
    wiki_stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wiki_texts = [row["text"] for row in itertools.islice(wiki_stream, WIKI_ARTICLE_LIMIT)]

    all_documents = squad_contexts + wiki_texts

    all_chunks = []
    for doc in all_documents:
        all_chunks.extend(_chunk_text(doc))

    print(f"Embedding {len(all_chunks)} chunks...")
    model = get_embedding_model()
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=16)

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