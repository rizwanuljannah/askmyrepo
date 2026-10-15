from dotenv import load_dotenv
load_dotenv()

import os
import re
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import MilvusClient, DataType

from chunk_repo import clone_repo, walk_and_chunk
from embed_repo import get_embedding
from query_assistant import ask_question

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "")
COLLECTION_NAME = "code_chunks_v2"
EMBEDDING_DIM = 768

# Only plain GitHub repo URLs are accepted (also keeps the Milvus filter string safe)
REPO_URL_PATTERN = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+$")

st.set_page_config(
    page_title="Codebase Dev Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Session state
#   chats: {repo_url: {"messages": [ {"role", "content", "sources"?} ]}}
#   active_repo: the repo whose chat is currently open
# ----------------------------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_repo" not in st.session_state:
    st.session_state.active_repo = None


# ----------------------------------------------------------------------
# Dark theme CSS (this app is dark-only)
# ----------------------------------------------------------------------
def inject_css():
    bg = "#0f1115"
    bg_secondary = "#171a21"
    bg_chat_user = "#2b2f3a"
    bg_chat_assistant = "#1b1e26"
    text_primary = "#e8e9ec"
    text_secondary = "#9aa0ac"
    border = "#2a2e37"
    accent = "#6c8cff"
    input_bg = "#171a21"

    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            .stApp {{
                background-color: {bg} !important;
                color: {text_primary} !important;
            }}

            .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
            .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
            .stMarkdown, .stMarkdown p, .stCaption, [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"],
            [data-testid="stWidgetLabel"] p {{
                color: {text_primary} !important;
            }}

            .stApp [data-testid="stCaptionContainer"],
            .stApp small,
            .app-header span.subtitle {{
                color: {text_secondary} !important;
            }}

            section[data-testid="stSidebar"] {{
                background-color: {bg_secondary} !important;
                border-right: 1px solid {border};
            }}
            section[data-testid="stSidebar"] * {{
                color: {text_primary} !important;
            }}

            .stTextInput input, .stTextArea textarea {{
                background-color: {input_bg} !important;
                color: {text_primary} !important;
                border: 1px solid {border} !important;
            }}

            div[data-testid="stChatInput"] textarea {{
                color: {text_primary} !important;
                background-color: {input_bg} !important;
            }}

            [data-testid="stChatMessage"] {{
                background-color: transparent !important;
            }}
            [data-testid="stChatMessage"] * {{
                color: {text_primary} !important;
            }}

            #MainMenu, footer {{ visibility: hidden; }}
            header[data-testid="stHeader"] {{
                background-color: transparent;
            }}
            div[data-testid="collapsedControl"] {{
                visibility: visible !important;
                display: block !important;
            }}

            .app-header {{
                display: flex;
                align-items: baseline;
                gap: 0.8rem;
                padding: 0.25rem 0 1rem 0;
                border-bottom: 1px solid {border};
                margin-bottom: 1.25rem;
            }}
            .app-header h1 {{
                font-size: 1.4rem;
                font-weight: 600;
                margin: 0;
                color: {text_primary};
            }}
            .app-header span.subtitle {{
                color: {text_secondary};
                font-size: 0.85rem;
            }}

            .stApp .sources-box, .stApp .sources-box b {{
                margin-top: 0.6rem;
                padding-top: 0.5rem;
                border-top: 1px dashed {border};
                font-size: 0.8rem;
                color: {text_secondary} !important;
            }}
            .stApp .sources-box code {{
                background-color: transparent !important;
                color: {accent} !important;
            }}

            .empty-state {{
                text-align: center;
                color: {text_secondary};
                padding: 3rem 1rem;
            }}
            .empty-state h3 {{
                color: {text_primary};
                font-weight: 600;
                margin-bottom: 0.4rem;
            }}

            .status-pill {{
                display: inline-block;
                padding: 0.2rem 0.7rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 600;
            }}
            .stApp .status-pill.ready {{
                background-color: rgba(76, 201, 121, 0.15);
                color: #2fa864 !important;
            }}
            .stApp .status-pill.notready {{
                background-color: rgba(255, 159, 67, 0.15);
                color: #d97e2b !important;
            }}
            .stApp .empty-state, .stApp .empty-state p {{
                color: {text_secondary} !important;
            }}
            .stApp .empty-state h3 {{
                color: {text_primary} !important;
            }}

            div[data-testid="stBottomBlockContainer"],
            div[data-testid="stBottom"] {{
                background-color: {bg} !important;
            }}

            div[data-testid="stChatInput"] {{
                background-color: {bg} !important;
                border-top: 1px solid {border} !important;
            }}

            div[data-testid="stChatInput"] > div {{
                background-color: {input_bg} !important;
                border: 1px solid {border} !important;
                border-radius: 14px !important;
                box-shadow: none !important;
            }}

            div[data-testid="stChatInput"] textarea {{
                background-color: {input_bg} !important;
                color: {text_primary} !important;
                caret-color: {text_primary} !important;
            }}

            div[data-testid="stChatInput"] textarea::placeholder {{
                color: {text_secondary} !important;
                opacity: 1 !important;
            }}

            div[data-testid="stChatInput"] button {{
                background-color: {accent} !important;
                border-radius: 10px !important;
            }}
            div[data-testid="stChatInput"] button svg {{
                fill: #ffffff !important;
            }}

            .stButton > button {{
                border-radius: 8px;
                border: 1px solid {border};
                font-weight: 500;
            }}
            .stButton > button:hover {{
                border-color: {accent};
                color: {accent};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ----------------------------------------------------------------------
# Repo helpers
# ----------------------------------------------------------------------
def normalize_repo_url(url):
    """Return a cleaned, lowercase GitHub repo URL, or None if it isn't valid."""
    url = (url or "").strip().rstrip("/")
    url = re.sub(r"\.git$", "", url)
    if not REPO_URL_PATTERN.match(url):
        return None
    return url.lower()


def repo_label(repo_url):
    """'https://github.com/owner/name' -> 'owner/name'"""
    return "/".join(repo_url.split("/")[-2:])


def get_clone_dir(repo_url):
    owner, name = repo_url.split("/")[-2:]
    return f"cloned_repos/{owner}__{name}"


# ----------------------------------------------------------------------
# Milvus / Zilliz helpers
#   One shared collection holds every repo's chunks. Each chunk carries a
#   "repo" value (stored as a dynamic field) so searches can be limited to
#   the repo whose chat is open.
# ----------------------------------------------------------------------
@st.cache_resource
def get_client():
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


def ensure_collection(client):
    """Create the collection if it doesn't exist yet (never drops anything)."""
    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


# --- Indexing order and batch size ------------------------------------
BATCH_SIZE = 250  # chunks embedded per click (keeps within free API limits)
MAX_WORKERS = 4   # embed_repo.py throttles the real request rate

SOURCE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".cs", ".rb", ".php", ".kt", ".swift",
}
LOW_PRIORITY_DIRS = {
    "test", "tests", "__tests__", "spec", "docs", "doc", "examples", "example",
    "benchmarks", "fixtures", "samples", "demo", "demos",
}
TEST_SUFFIXES = (
    "_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts",
    "test.java", "tests.java",
)


def file_priority(filepath, clone_dir):
    """0 = main source code, 1 = tests/examples, 2 = docs/config/data."""
    rel = filepath.replace("\\", "/")
    prefix = clone_dir.replace("\\", "/") + "/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    parts = rel.lower().split("/")
    name = parts[-1]
    ext = os.path.splitext(name)[1]
    in_low_dir = any(p in LOW_PRIORITY_DIRS for p in parts[:-1])
    is_test_file = name.startswith("test_") or name.endswith(TEST_SUFFIXES)
    if ext not in SOURCE_EXTS:
        return 2
    if in_low_dir or is_test_file:
        return 1
    return 0


def repo_status(client, repo):
    """How much of this repo is indexed? Returns None if nothing is, else
    {"indexed": rows stored, "next_offset": where the next batch starts, "total": chunks in repo}."""
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'repo == "{repo}"',
        output_fields=["chunk_order", "repo_total"],
        limit=16384,
    )
    if not rows:
        return None
    orders = [r["chunk_order"] for r in rows if "chunk_order" in r]
    total = next((r["repo_total"] for r in rows if "repo_total" in r), len(rows))
    next_offset = (max(orders) + 1) if orders else len(rows)
    return {"indexed": len(rows), "next_offset": next_offset, "total": total}


# Remembers the most recent embedding failure so the UI can show *why*
# chunks were skipped (useful when debugging a deployment).
LAST_EMBED_ERROR = {}


def safe_get_embedding(chunk_text, max_retries=1):
    """Try to embed a chunk. If the API rejects it for being too long,
    truncate it and try once more. Returns None if it still fails,
    so the caller can skip this chunk instead of crashing."""
    text_to_send = chunk_text
    for attempt in range(max_retries + 1):
        try:
            return get_embedding(text_to_send)
        except requests.exceptions.HTTPError as e:
            body = e.response.text if e.response is not None else ""
            status = e.response.status_code if e.response is not None else "?"
            LAST_EMBED_ERROR["msg"] = f"HTTP {status}: {body[:300]}"
            if "context length" in body.lower() and attempt < max_retries:
                text_to_send = text_to_send[: len(text_to_send) // 2]
                continue
            return None
        except requests.exceptions.RequestException as e:
            LAST_EMBED_ERROR["msg"] = f"{type(e).__name__}: {e}"
            return None
    return None


def process_repo(repo, status_container, force=False, extend=False):
    """Index a repo (first BATCH_SIZE chunks, main source code first).
    extend=True adds the next batch to what is already indexed.
    force=True rebuilds the repo's index from scratch."""
    client = get_client()
    ensure_collection(client)

    status = repo_status(client, repo)

    # Already indexed? Open it instantly, no embedding quota used.
    if status and not force and not extend:
        status_container.success(
            f"Already indexed ({status['indexed']} of {status['total']} chunks). Opening the chat."
        )
        return True

    clone_dir = get_clone_dir(repo)

    status_container.write("Cloning repo...")
    clone_repo(repo, clone_dir)

    status_container.write("Chunking files...")
    file_chunks = walk_and_chunk(clone_dir)

    # Flatten, then order: main source code first, tests/examples next, docs/config last.
    all_items = [
        (filepath, i, chunk_text)
        for filepath, chunks in file_chunks.items()
        for i, chunk_text in enumerate(chunks)
    ]
    all_items.sort(key=lambda item: (file_priority(item[0], clone_dir), item[0], item[1]))
    total = len(all_items)
    status_container.write(f"Found {total} chunk(s) across {len(file_chunks)} file(s).")

    start = status["next_offset"] if (extend and status) else 0
    if start >= total:
        status_container.success("Every chunk of this repo is already indexed.")
        return True

    batch = [
        (order, filepath, i, chunk_text)
        for order, (filepath, i, chunk_text) in enumerate(all_items[start:start + BATCH_SIZE], start=start)
    ]
    if start + len(batch) < total:
        status_container.info(
            f"Indexing chunks {start + 1} to {start + len(batch)} of {total} (free API limits)."
        )

    status_container.write("Embedding chunks...")
    progress_bar = status_container.progress(0.0)
    records = []
    skipped = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(safe_get_embedding, chunk_text): (order, filepath, i, chunk_text)
            for order, filepath, i, chunk_text in batch
        }

        for future in as_completed(future_to_item):
            order, filepath, i, chunk_text = future_to_item[future]
            embedding = future.result()
            if embedding is None:
                skipped.append(f"{filepath} (chunk {i})")
            else:
                records.append({
                    "file_path": filepath,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                    "repo": repo,
                    "chunk_order": order,
                    "repo_total": total,
                })
            done += 1
            progress_bar.progress(done / len(batch), text=f"Processed {done}/{len(batch)} chunks")

    if skipped:
        status_container.warning(f"Skipped {len(skipped)} chunk(s) that couldn't be embedded.")
        status_container.code(LAST_EMBED_ERROR.get("msg", "unknown error"))

    if not records:
        status_container.error("No chunks were successfully embedded. Nothing to index.")
        return False

    status_container.write("Saving to the vector database...")
    if not extend:
        # Fresh index or forced rebuild: replace this repo's old rows, leave other repos alone
        client.delete(collection_name=COLLECTION_NAME, filter=f'repo == "{repo}"')
    client.insert(collection_name=COLLECTION_NAME, data=records)
    client.flush(COLLECTION_NAME)

    indexed = (status["indexed"] if (extend and status) else 0) + len(records)
    status_container.success(
        f"Indexed {len(records)} new chunk(s): {indexed} of {total} total. Ready to answer questions."
    )
    return True


def open_chat(repo):
    """Make sure a chat exists for this repo, refresh its coverage, and open it."""
    if repo not in st.session_state.chats:
        st.session_state.chats[repo] = {"messages": []}
    st.session_state.chats[repo]["coverage"] = repo_status(get_client(), repo)
    st.session_state.active_repo = repo


# ----------------------------------------------------------------------
# Sidebar: new repo + previous chats
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Codebase Dev Assistant")
    st.caption("RAG over any public GitHub repo")

    st.divider()

    st.markdown("#### New repository")
    repo_input = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo")
    force_reindex = st.checkbox("Re-index even if already indexed", value=False)
    process_clicked = st.button("Process Repo", use_container_width=True, type="primary")

    status_box = st.container()

    if process_clicked:
        repo = normalize_repo_url(repo_input)
        if not repo:
            with status_box:
                st.error("Enter a valid GitHub URL, like https://github.com/user/repo")
        else:
            with status_box:
                with st.spinner("Working..."):
                    success = process_repo(repo, status_box, force_reindex)
            if success:
                open_chat(repo)

    st.divider()

    st.markdown("#### Previous chats")
    if not st.session_state.chats:
        st.caption("No chats yet. Process a repo to start one.")
    for repo_key in reversed(list(st.session_state.chats.keys())):
        is_active = repo_key == st.session_state.active_repo
        label = ("● " if is_active else "💬 ") + repo_label(repo_key)
        if st.button(label, key=f"chat_{repo_key}", use_container_width=True):
            st.session_state.active_repo = repo_key
            st.rerun()

    st.divider()

    active = st.session_state.active_repo
    if active:
        st.markdown(
            '<span class="status-pill ready">● Repo indexed</span>',
            unsafe_allow_html=True,
        )
        st.caption(active)

        coverage = st.session_state.chats[active].get("coverage")
        extend_clicked = False
        if coverage and coverage["indexed"] < coverage["total"]:
            extend_clicked = st.button(
                f"Index next {BATCH_SIZE} chunks", use_container_width=True
            )
        if extend_clicked:
            with status_box:
                with st.spinner("Working..."):
                    extended_ok = process_repo(active, status_box, extend=True)
            if extended_ok:
                open_chat(active)
                coverage = st.session_state.chats[active].get("coverage")
        if coverage:
            st.caption(f"Indexed {coverage['indexed']} of {coverage['total']} chunks")

        if st.session_state.chats[active]["messages"]:
            if st.button("Clear this chat", use_container_width=True):
                st.session_state.chats[active]["messages"] = []
                st.rerun()
    else:
        st.markdown(
            '<span class="status-pill notready">● No repo selected</span>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------
# Main area: the open chat
# ----------------------------------------------------------------------
active = st.session_state.active_repo
messages = st.session_state.chats[active]["messages"] if active else []

subtitle = f'<span class="subtitle">{repo_label(active)}</span>' if active else ""
st.markdown(
    f"""
    <div class="app-header">
        <h1>Ask your codebase</h1>
        {subtitle}
    </div>
    """,
    unsafe_allow_html=True,
)

# Be upfront when only part of a big repo is searchable
active_coverage = st.session_state.chats[active].get("coverage") if active else None
if active_coverage and active_coverage["indexed"] < active_coverage["total"]:
    st.caption(
        f"⚠️ Answers cover {active_coverage['indexed']} of {active_coverage['total']} chunks "
        f"(main source files are indexed first). Use the sidebar button to index more."
    )

if not active:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No chat open</h3>
            <p>Process a repo from the sidebar, or open one from Previous chats.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not messages:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No questions yet</h3>
            <p>Ask anything about this repo below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                sources_html = "<br>".join(f"<code>{s}</code>" for s in msg["sources"])
                st.markdown(
                    f'<div class="sources-box"><b>Sources</b><br>{sources_html}</div>',
                    unsafe_allow_html=True,
                )

question = st.chat_input("Ask something about the codebase...")

if question:
    if not active:
        st.warning("Process a repo or open a previous chat from the sidebar first.")
    else:
        messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Thinking..."):
                answer, cited_files = ask_question(question, repo=active)
            messages.append({
                "role": "assistant",
                "content": answer,
                "sources": cited_files,
            })
        except Exception as e:
            messages.append({
                "role": "assistant",
                "content": f"Something went wrong answering the question: {e}",
            })
        st.rerun()