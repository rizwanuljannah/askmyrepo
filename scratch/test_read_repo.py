import os
import git

# A small public repo to test with
REPO_URL = "https://github.com/navdeep-G/setup.py"
CLONE_DIR = "test_repo"

# Clone only if not already cloned
if not os.path.exists(CLONE_DIR):
    print("Cloning repo...")
    git.Repo.clone_from(REPO_URL, CLONE_DIR)
    print("Done cloning.")
else:
    print("Repo already exists, skipping clone.")

# Walk through the repo and list all Python files
print("\nPython files found:")
for root, dirs, files in os.walk(CLONE_DIR):
    # skip hidden folders like .git
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            print(filepath)