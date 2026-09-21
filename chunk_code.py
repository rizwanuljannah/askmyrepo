import os
import json
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

# Map file extensions to their tree-sitter language
LANGUAGE_MAP = {
    ".py": Language(tspython.language()),
    ".js": Language(tsjavascript.language()),
    ".ts": Language(tstypescript.language_typescript()),
    ".java": Language(tsjava.language()),
    ".c": Language(tsc.language()),
    ".cpp": Language(tscpp.language()),
    ".go": Language(tsgo.language()),
}

# Which node types count as "a function/method" for each language
NODE_TYPES_MAP = {
    ".py": ("function_definition",),
    ".js": ("function_declaration", "method_definition"),
    ".ts": ("function_declaration", "method_definition"),
    ".java": ("method_declaration",),
    ".c": ("function_definition",),
    ".cpp": ("function_definition",),
    ".go": ("function_declaration", "method_declaration"),
}

print("Languages loaded:", list(LANGUAGE_MAP.keys()))


def chunk_source_with_grammar(source_code, ext):
    """Parse source_code using the tree-sitter grammar for `ext` and
    return one chunk per function/method found."""
    target_node_types = NODE_TYPES_MAP[ext]

    parser = Parser(LANGUAGE_MAP[ext])
    tree = parser.parse(bytes(source_code, "utf-8"))

    chunks = []

    def walk(node):
        if node.type in target_node_types:
            # Capture this function/method, and DON'T look inside it further
            chunk_text = source_code[node.start_byte:node.end_byte]
            chunks.append(chunk_text)
            return
        # Not a function itself — keep looking inside (e.g. inside a class)
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not chunks:
        # No functions/methods found (common in scripts/notebooks that are
        # just top-level sequential code) — fall back so this content isn't lost
        return fallback_chunk(source_code)

    return chunks


def extract_notebook_cells(file_path):
    """Read a .ipynb file's JSON structure and return a list of each
    individual code cell's source (skipping markdown/empty cells)."""
    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell_code = "".join(cell.get("source", []))
            if cell_code.strip():
                cells.append(cell_code)
    return cells


def get_chunks(file_path):
    ext = os.path.splitext(file_path)[1]

    if ext == ".ipynb":
        print("Notebook detected - extracting code cells...")
        cells = extract_notebook_cells(file_path)
        if not cells:
            print("No code cells found in notebook.")
            return []

        chunks = []
        for cell_code in cells:
            # Try to find functions within this cell; if none, the cell
            # itself becomes one chunk (via chunk_source_with_grammar's
            # own fallback for "no functions found")
            chunks.extend(chunk_source_with_grammar(cell_code, ".py"))
        return chunks

    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    if ext not in LANGUAGE_MAP:
        print(f"No grammar for {ext}, using fallback chunking.")
        return fallback_chunk(source_code)

    return chunk_source_with_grammar(source_code, ext)


def fallback_chunk(source_code, lines_per_chunk=40):
    lines = source_code.splitlines()
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i:i + lines_per_chunk]
        chunks.append("\n".join(chunk_lines))
    return chunks


# Test it (only runs when this file is executed directly, not when imported)
if __name__ == "__main__":
    result = get_chunks("target_repo/Bitcoin_Market_Sentiment_Analysis.ipynb")
    for i, chunk in enumerate(result):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()