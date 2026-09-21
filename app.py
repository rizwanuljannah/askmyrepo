import re
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import MilvusClient, DataType

from chunk_repo import clone_repo, walk_and_chunk
from embed_repo import get_embedding
from query_assistant import ask_question

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "code_chunks"
EMBEDDING_DIM = 768

st.set_page_config(
    page_title="Codebase Dev Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
if "repo_processed" not in st.session_state:
    st.session_state.repo_processed = False
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "messages" not in st.session_state:
    # each item: {"role": "user"/"assistant", "content": str, "sources": list[str] (optional)}
    st.session_state.messages = []
if "current_repo" not in st.session_state:
    st.session_state.current_repo = None


# ----------------------------------------------------------------------
# Theme / CSS
# ----------------------------------------------------------------------
def inject_css(theme: str):
    if theme == "dark":
        bg = "#0f1115"
        bg_secondary = "#171a21"
        bg_chat_user = "#2b2f3a"
        bg_chat_assistant = "#1b1e26"
        text_primary = "#e8e9ec"
        text_secondary = "#9aa0ac"
        border = "#2a2e37"
        accent = "#6c8cff"
        input_bg = "#171a21"
    else:
        bg = "#ffffff"
        bg_secondary = "#f7f7f8"
        bg_chat_user = "#eef1ff"
        bg_chat_assistant = "#f7f7f8"
        text_primary = "#1a1a1a"
        text_secondary = "#6b7280"
        border = "#e5e7eb"
        accent = "#4f6bed"
        input_bg = "#ffffff"

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
                align-items: center;
                gap: 0.6rem;
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

            .chat-bubble {{
                padding: 0.85rem 1.1rem;
                border-radius: 14px;
                margin-bottom: 0.9rem;
                line-height: 1.55;
                font-size: 0.95rem;
                max-width: 100%;
            }}
            .chat-bubble.user {{
                background-color: {bg_chat_user};
                border: 1px solid {border};
            }}
            .chat-bubble.assistant {{
                background-color: {bg_chat_assistant};
                border: 1px solid {border};
            }}
            .chat-role {{
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.03em;
                color: {text_secondary};
                margin-bottom: 0.35rem;
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


inject_css(st.session_state.theme)


# ----------------------------------------------------------------------
# Backend helpers (unchanged logic from the original app)
# ----------------------------------------------------------------------
def get_clone_dir(repo_url):
    name = repo_url.rstrip("/").split("/")[-1]
    name = re.sub(r"\.git$", "", name)
    return f"cloned_repos/{name}"


def setup_collection():
    client = MilvusClient(uri=MILVUS_URI)
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_path", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    return client


def safe_get_embedding(chunk_text, max_retries=1):
    """Try to embed a chunk. If Ollama rejects it for being too long,
    truncate it and try once more. Returns None if it still fails,
    so the caller can skip this chunk instead of crashing."""
    text_to_send = chunk_text
    for attempt in range(max_retries + 1):
        try:
            return get_embedding(text_to_send)
        except requests.exceptions.HTTPError as e:
            body = e.response.text if e.response is not None else ""
            if "context length" in body.lower() and attempt < max_retries:
                text_to_send = text_to_send[: len(text_to_send) // 2]
                continue
            return None
        except requests.exceptions.RequestException:
            return None
    return None


def process_repo(repo_url, status_container):
    clone_dir = get_clone_dir(repo_url)

    status_container.write("Cloning repo...")
    clone_repo(repo_url, clone_dir)

    status_container.write("Chunking files...")
    file_chunks = walk_and_chunk(clone_dir)
    total_chunks = sum(len(c) for c in file_chunks.values())
    status_container.write(f"Found {total_chunks} chunk(s) across {len(file_chunks)} file(s).")

    status_container.write("Embedding chunks (parallelized)...")
    progress_bar = status_container.progress(0.0)
    records = []
    skipped = []
    done = 0

    # Flatten to a single list of (filepath, chunk_index, chunk_text) so we
    # can fire many embedding requests at Ollama concurrently instead of
    # waiting for each one to finish before starting the next.
    work_items = [
        (filepath, i, chunk_text)
        for filepath, chunks in file_chunks.items()
        for i, chunk_text in enumerate(chunks)
    ]

    MAX_WORKERS = 8  # concurrent requests to Ollama; lower this if it overwhelms your machine

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(safe_get_embedding, chunk_text): (filepath, i, chunk_text)
            for filepath, i, chunk_text in work_items
        }

        for future in as_completed(future_to_item):
            filepath, i, chunk_text = future_to_item[future]
            embedding = future.result()
            if embedding is None:
                skipped.append(f"{filepath} (chunk {i})")
            else:
                records.append({
                    "file_path": filepath,
                    "chunk_index": i,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                })
            done += 1
            if total_chunks:
                progress_bar.progress(done / total_chunks, text=f"Processed {done}/{total_chunks} chunks")

    if skipped:
        status_container.warning(f"Skipped {len(skipped)} chunk(s) that couldn't be embedded.")

    if not records:
        status_container.error("No chunks were successfully embedded. Nothing to index.")
        return False

    status_container.write("Setting up Milvus collection...")
    client = setup_collection()

    status_container.write("Inserting into Milvus...")
    client.insert(collection_name=COLLECTION_NAME, data=records)
    client.flush(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)

    status_container.success(f"Indexed {stats['row_count']} chunk(s). Ready to answer questions.")
    return True


# ----------------------------------------------------------------------
# Sidebar — repo input + theme toggle
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Codebase Dev Assistant")
    st.caption("RAG over any public GitHub repo")

    st.divider()

    theme_choice = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
    )
    new_theme = theme_choice.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()

    st.markdown("#### Repository")
    repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/user/repo")
    process_clicked = st.button("Process Repo", use_container_width=True, type="primary")

    status_box = st.container()

    if process_clicked and repo_url:
        st.session_state.messages = []
        with status_box:
            with st.spinner("Working..."):
                success = process_repo(repo_url, status_box)
        if success:
            st.session_state.repo_processed = True
            st.session_state.current_repo = repo_url

    st.divider()

    if st.session_state.repo_processed:
        st.markdown(
            f'<span class="status-pill ready">● Repo indexed</span>',
            unsafe_allow_html=True,
        )
        if st.session_state.current_repo:
            st.caption(st.session_state.current_repo)
    else:
        st.markdown(
            '<span class="status-pill notready">● No repo indexed yet</span>',
            unsafe_allow_html=True,
        )

    if st.session_state.messages:
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ----------------------------------------------------------------------
# Main area — ChatGPT-style chat
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Ask your codebase</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No questions yet</h3>
            <p>Process a repo from the sidebar, then ask anything about it below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
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
    st.session_state.messages.append({"role": "user", "content": question})

    if not st.session_state.repo_processed:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please process a repository first (see the sidebar).",
        })
    else:
        try:
            with st.spinner("Thinking..."):
                answer, cited_files = ask_question(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": cited_files,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Something went wrong answering the question: {e}",
            })

    st.rerun()