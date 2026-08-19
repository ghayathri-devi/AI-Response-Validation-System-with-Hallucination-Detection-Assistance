from knowledge_builder import _chroma_client, COLLECTION_NAME, get_embedding_model


def retrieve_context(question: str, top_k: int = 2) -> list[str]:
    model = get_embedding_model()
    query_embedding = model.encode([question]).tolist()

    # Fetch the collection fresh on every call rather than caching it once
    # at import time — the background thread that builds the knowledge
    # base runs after this module is first imported, so a cached reference
    # taken too early could miss data added afterward. get_collection() on
    # an already-open client is a cheap call, not a new connection.
    collection = _chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    if not results.get("documents"):
        return []

    return results["documents"][0]