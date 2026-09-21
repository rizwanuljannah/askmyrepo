from pymilvus import MilvusClient

from embed_repo import get_embedding

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "code_chunks"
TOP_K = 3  # how many matching chunks to retrieve

client = MilvusClient(uri=MILVUS_URI)


def search_chunks(question, top_k=TOP_K):
    """Embed the question, search Milvus, and return the top matching chunks."""
    question_embedding = get_embedding(question)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[question_embedding],
        limit=top_k,
        output_fields=["file_path", "chunk_index", "chunk_text"],
    )

    # results[0] is the list of matches for our one question
    return results[0]


if __name__ == "__main__":
    question = "How does this app connect to a database?"

    matches = search_chunks(question)

    print(f"Question: {question}\n")
    print(f"Top {len(matches)} matching chunk(s):\n")

    for i, match in enumerate(matches):
        print(f"--- Match {i+1} (similarity score: {match['distance']:.4f}) ---")
        print(f"File: {match['entity']['file_path']}")
        print(f"Chunk text:\n{match['entity']['chunk_text'][:300]}")
        print()