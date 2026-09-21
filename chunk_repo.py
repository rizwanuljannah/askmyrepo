import os
import git

from chunk_code import get_chunks, LANGUAGE_MAP

# File extensions we should NEVER try to read as text (binary/media/data files)
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".exe", ".dll", ".so", ".pyc",
    ".csv", ".parquet", ".pkl", ".pt", ".h5", ".onnx",
}

# Exact filenames we should skip, regardless of extension
# (lock files are noisy/auto-generated, .env files often contain secrets)
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    ".env", ".env.local", ".env.production",
}

# Folders we never want to walk into
SKIP_DIRS = {
    ".git", "venv", "__pycache__", "node_modules", ".idea", ".vscode",
    "dist", "build", ".next", "target", "bin", "obj",
}


def clone_repo(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        print("Cloning repo...")
        git.Repo.clone_from(repo_url, clone_dir)
        print("Done cloning.")
    else:
        print("Repo already exists, skipping clone.")


def walk_and_chunk(clone_dir):
    all_results = {}  # file_path -> list of chunks

    for root, dirs, files in os.walk(clone_dir):
        # Skip unwanted folders (modify dirs in-place so os.walk skips them)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            ext = os.path.splitext(filename)[1]

            if filename in SKIP_FILENAMES:
                print(f"Skipping non-useful file: {filename}")
                continue

            if ext in SKIP_EXTENSIONS:
                print(f"Skipping binary/data file: {filename}")
                continue

            filepath = os.path.join(root, filename)

            try:
                chunks = get_chunks(filepath)
            except UnicodeDecodeError:
                print(f"Skipping likely binary file (couldn't read as text): {filename}")
                continue
            except Exception as e:
                print(f"Could not process {filepath}: {e}")
                continue

            if chunks:
                all_results[filepath] = chunks

    return all_results


if __name__ == "__main__":
    # Default settings when running this file directly (not through the app)
    DEFAULT_REPO_URL = "https://github.com/heroku/java-getting-started"
    DEFAULT_CLONE_DIR = "target_repo_JAVA"

    clone_repo(DEFAULT_REPO_URL, DEFAULT_CLONE_DIR)
    results = walk_and_chunk(DEFAULT_CLONE_DIR)

    print(f"\n=== Summary: {len(results)} file(s) produced chunks ===\n")
    for filepath, chunks in results.items():
        print(f"{filepath}: {len(chunks)} chunk(s)")

    # Show the actual content of chunks for one file as a sample.
    # Prefer a real source-code file (one whose extension has a tree-sitter
    # grammar) over config/text files, since that's what we actually want
    # to eyeball for correctness.
    if results:
        code_files = [
            fp for fp in results
            if os.path.splitext(fp)[1] in LANGUAGE_MAP
        ]
        sample_file = code_files[0] if code_files else list(results.keys())[0]

        print(f"\n=== Sample chunks from {sample_file} ===\n")
        for i, chunk in enumerate(results[sample_file]):
            print(f"--- Chunk {i+1} ---")
            print(chunk[:300])  # limit to first 300 chars for readability
            print()