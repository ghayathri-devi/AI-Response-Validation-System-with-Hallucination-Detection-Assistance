import os
import shutil

import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 100

WIKI_ARTICLE_LIMIT = int(os.getenv("WIKI_ARTICLE_LIMIT", "150"))
SQUAD_ROW_LIMIT = int(os.getenv("SQUAD_ROW_LIMIT", "500"))

_model = None


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

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(f"Knowledge base already populated with {collection.count()} chunks — skipping rebuild.")
        return collection

    print("Building knowledge base...")

    print(f"Loading SQuAD (first {SQUAD_ROW_LIMIT} rows)...")
    squad = load_dataset("rajpurkar/squad", split=f"train[:{SQUAD_ROW_LIMIT}]")
    squad_contexts = list(set(squad["context"]))

    print(f"Loading {WIKI_ARTICLE_LIMIT} Wikipedia articles...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split=f"train[:{WIKI_ARTICLE_LIMIT}]")
    wiki_texts = list(wiki["text"])

    all_documents = squad_contexts + wiki_texts

    all_chunks = []
    for doc in all_documents:
        all_chunks.extend(_chunk_text(doc))

    print(f"Embedding {len(all_chunks)} chunks...")
    model = get_embedding_model()
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)

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