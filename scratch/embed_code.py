import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


# Test it
sample_text = "def add(a, b):\n    return a + b"
embedding = get_embedding(sample_text)

print(f"Embedding length: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")