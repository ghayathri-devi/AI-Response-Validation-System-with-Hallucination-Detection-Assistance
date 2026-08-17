import chromadb

from knowledge_builder import CHROMA_PATH, COLLECTION_NAME, get_embedding_model

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_context(question: str, top_k: int = 2) -> list[str]:
    model = get_embedding_model()
    query_embedding = model.encode([question]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    if not results.get("documents"):
        return []

    return results["documents"][0]