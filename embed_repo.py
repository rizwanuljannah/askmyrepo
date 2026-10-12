import os
import time
import threading
import requests

from chunk_repo import clone_repo, walk_and_chunk

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def get_embedding(text):
   _throttle_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.7  # ~85 requests/minute, under the 100/min free limit


def _wait_turn():
    with _throttle_lock:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def get_embedding(text, max_retries=4):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768,
    }
    for attempt in range(max_retries + 1):
        _wait_turn()
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        if response.status_code == 429 and attempt < max_retries:
            time.sleep(5 * (attempt + 1))  # back off, then retry
            continue
        break
    if response.status_code != 200:
        print(f"Gemini embedding error (status {response.status_code}): {response.text}")
    response.raise_for_status()
    return response.json()["embedding"]["values"]

def embed_all_chunks(repo_url, clone_dir):
    """Clone the repo, chunk every file, and embed every chunk.
    Returns a list of records, each with the chunk's text, its
    embedding, and metadata about where it came from."""
    clone_repo(repo_url, clone_dir)
    file_chunks = walk_and_chunk(clone_dir)  # {file_path: [chunk1, chunk2, ...]}

    records = []
    for filepath, chunks in file_chunks.items():
        for i, chunk_text in enumerate(chunks):
            try:
                embedding = get_embedding(chunk_text)
            except Exception as e:
                print(f"FAILED to embed chunk {i} of {filepath}: {e}")
                print(f"Chunk length: {len(chunk_text)} characters")
                print(f"Chunk preview: {chunk_text[:200]!r}")
                raise
            records.append({
                "file_path": filepath,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "embedding": embedding,
            })

    return records


if __name__ == "__main__":
    DEFAULT_REPO_URL = "https://github.com/heroku/java-getting-started"
    DEFAULT_CLONE_DIR = "target_repo_JAVA"

    records = embed_all_chunks(DEFAULT_REPO_URL, DEFAULT_CLONE_DIR)

    print(f"\n=== Embedded {len(records)} chunk(s) total ===\n")

    bad_records = [r for r in records if len(r["embedding"]) != 768]
    if bad_records:
        print(f"WARNING: {len(bad_records)} chunk(s) had an unexpected embedding length!")
    else:
        print("All embeddings are 768-dimensional, as expected.")

    if records:
        sample = records[0]
        print("\n=== Sample record ===")
        print(f"File: {sample['file_path']}")
        print(f"Chunk index: {sample['chunk_index']}")
        print(f"Chunk text (first 200 chars): {sample['chunk_text'][:200]}")
        print(f"Embedding length: {len(sample['embedding'])}")
        print(f"First 5 embedding values: {sample['embedding'][:5]}")