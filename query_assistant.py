import requests

from search_chunks import search_chunks

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
GENERATE_MODEL = "llama3.2:3b"


def build_prompt(question, matches):
    """Combine the question and retrieved chunks into one prompt for llama3."""
    context_parts = []
    for match in matches:
        file_path = match["entity"]["file_path"]
        chunk_text = match["entity"]["chunk_text"]
        context_parts.append(f"--- From {file_path} ---\n{chunk_text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the code context below to answer the question. If the context
doesn't contain enough information to answer, say so honestly instead
of guessing. When you reference something, mention which file it came from.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask_question(question):
    matches = search_chunks(question)
    prompt = build_prompt(question, matches)

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": GENERATE_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    answer = response.json()["response"]

    # Collect the unique file paths used, for a citation list
    cited_files = sorted(set(m["entity"]["file_path"] for m in matches))

    return answer, cited_files


if __name__ == "__main__":
    question = "How does this app connect to a database?"

    answer, cited_files = ask_question(question)

    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for f in cited_files:
        print(f"  - {f}")