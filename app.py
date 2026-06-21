"""
GitRAG Frontend - Streamlit UI
Talks to the FastAPI backend (main.py) running on localhost:8000.

"""
import os
import streamlit as st
import requests
import time

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="GitRAG",
    page_icon="⌗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM - injected CSS
# Manuscript/marginalia aesthetic: aged paper + ink + brass accents
# Fraunces (display serif) + JetBrains Mono (everything else)
# ============================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --ink: #0F1419;
    --paper: #F7F3E9;
    --paper-dim: #EDE7D8;
    --brass: #D4A04C;
    --brass-dim: #B8893E;
    --oxblood: #8B3A3A;
    --sage: #5C7A5C;
    --graphite: #3D4147;
    --graphite-soft: #6B6F76;
}

/* kill streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
}

.stApp {
    background: var(--ink);
    background-image:
        radial-gradient(circle at 20% 0%, rgba(212, 160, 76, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 80% 100%, rgba(92, 122, 92, 0.05) 0%, transparent 45%);
}

.block-container {
    max-width: 720px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ---------- HEADER ---------- */
.gitrag-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid rgba(247, 243, 233, 0.12);
    padding-bottom: 1.1rem;
    margin-bottom: 2.2rem;
}
.gitrag-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-style: italic;
    font-size: 2.1rem;
    color: var(--paper);
    letter-spacing: -0.01em;
}
.gitrag-title .mark { color: var(--brass); font-style: normal; }
.gitrag-status {
    font-size: 0.72rem;
    color: var(--graphite-soft);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--graphite-soft);
    display: inline-block;
}
.dot.live { background: var(--sage); box-shadow: 0 0 8px var(--sage); animation: pulse 2s ease-in-out infinite; }
.dot.busy { background: var(--brass); box-shadow: 0 0 8px var(--brass); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ---------- SECTION LABEL (marginalia style) ---------- */
.margin-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--brass);
    margin-bottom: 0.6rem;
    padding-left: 1.1rem;
    position: relative;
}
.margin-label::before {
    content: '—';
    position: absolute;
    left: 0;
    color: var(--brass-dim);
}

/* ---------- INPUT STYLING ---------- */
.stTextInput input, .stTextArea textarea {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.92rem !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 2px 0 rgba(0,0,0,0.25), inset 0 0 0 1px rgba(15,20,25,0.06) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    box-shadow: 0 2px 0 rgba(0,0,0,0.25), 0 0 0 2px var(--brass) !important;
}
.stTextInput input::placeholder { color: #9A9486 !important; }

/* buttons */
.stButton button {
    background: var(--brass) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em;
    padding: 0.7rem 1.3rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 0 rgba(0,0,0,0.3) !important;
}
.stButton button:hover {
    background: var(--brass-dim) !important;
    transform: translateY(1px);
    box-shadow: 0 1px 0 rgba(0,0,0,0.3) !important;
}
.stButton button:disabled {
    background: var(--graphite) !important;
    color: var(--graphite-soft) !important;
}

/* ---------- INDEX CARD (repo status) ---------- */
.index-card {
    background: var(--paper);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 3px 0 rgba(0,0,0,0.3), 2px 5px 14px rgba(0,0,0,0.35);
    position: relative;
    transform: rotate(-0.3deg);
}
.index-card::before {
    content: '';
    position: absolute;
    top: -6px; left: 24px;
    width: 28px; height: 12px;
    background: rgba(212, 160, 76, 0.65);
    border-radius: 1px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.index-card-repo {
    font-size: 0.8rem;
    color: var(--ink);
    font-weight: 600;
    word-break: break-all;
}
.index-card-meta {
    font-size: 0.72rem;
    color: var(--graphite-soft);
    margin-top: 0.4rem;
    display: flex;
    gap: 1.2rem;
}
.index-card-meta b { color: var(--sage); }

/* ---------- ANSWER BLOCK ---------- */
.answer-block {
    background: var(--paper);
    border-radius: 4px;
    padding: 1.6rem 1.6rem 1.2rem 1.6rem;
    margin-top: 1.2rem;
    box-shadow: 0 3px 0 rgba(0,0,0,0.3), 2px 6px 18px rgba(0,0,0,0.4);
    color: var(--ink);
    font-size: 0.88rem;
    line-height: 1.65;
    animation: riseIn 0.4s ease;
}
@keyframes riseIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.answer-block .q {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 1.15rem;
    color: var(--ink);
    margin-bottom: 0.9rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px dashed rgba(15,20,25,0.18);
}
.answer-block .a { color: var(--ink); white-space: pre-wrap; }

/* sources as pinned index tabs */
.sources-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--graphite-soft);
    margin-top: 1.3rem;
    margin-bottom: 0.5rem;
}
.source-tab {
    display: inline-block;
    background: rgba(212, 160, 76, 0.16);
    border: 1px solid rgba(212, 160, 76, 0.4);
    color: var(--brass-dim);
    font-size: 0.68rem;
    padding: 0.3rem 0.55rem;
    border-radius: 2px;
    margin: 0.2rem 0.3rem 0.2rem 0;
    font-family: 'JetBrains Mono', monospace;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--graphite-soft);
}
.empty-state .glyph {
    font-size: 2.2rem;
    color: var(--brass-dim);
    margin-bottom: 0.8rem;
}
.empty-state .msg {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--paper);
    opacity: 0.55;
}

/* progress text */
.progress-line {
    font-size: 0.78rem;
    color: var(--brass);
    margin-top: 0.6rem;
    padding-left: 1.1rem;
}

hr { border-color: rgba(247, 243, 233, 0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "chunks_created" not in st.session_state:
    st.session_state.chunks_created = None
if "files_processed" not in st.session_state:
    st.session_state.files_processed = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {q, a, sources}
if "indexing" not in st.session_state:
    st.session_state.indexing = False

# ============================================================
# HEADER
# ============================================================
status_html = '<span class="dot live"></span>Repo indexed' if st.session_state.indexed_repo else '<span class="dot"></span>No repo indexed'
st.markdown(f"""
<div class="gitrag-header">
    <div class="gitrag-title">git<span class="mark">rag</span></div>
    <div class="gitrag-status">{status_html}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INGEST SECTION
# ============================================================
st.markdown('<div class="margin-label">point it at a repository</div>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    repo_url = st.text_input(
        "repo_url", placeholder="https://github.com/owner/repo",
        label_visibility="collapsed", key="repo_input"
    )
with col2:
    ingest_clicked = st.button("Read it", use_container_width=True)

if ingest_clicked and repo_url:
    st.session_state.indexing = True
    progress_placeholder = st.empty()
    progress_placeholder.markdown(
        '<div class="progress-line">⌗ cloning, parsing, embedding — this takes a minute...</div>',
        unsafe_allow_html=True
    )
    try:
        resp = requests.post(f"{API_BASE}/ingest", json={"repo_url": repo_url}, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.indexed_repo = data["repo_url"]
            st.session_state.chunks_created = data["chunks_created"]
            st.session_state.files_processed = data["files_processed"]
            st.session_state.history = []
            progress_placeholder.empty()
            st.rerun()
        else:
            progress_placeholder.empty()
            detail = resp.json().get("detail", "Unknown error")
            st.error(f"Couldn't read this repo — {detail}")
    except requests.exceptions.ConnectionError:
        progress_placeholder.empty()
        st.error("Backend isn't reachable. Is `uvicorn main:app --reload` running on port 8000?")
    except requests.exceptions.Timeout:
        progress_placeholder.empty()
        st.error("Took too long — repo may be too large, or the backend hung.")
    st.session_state.indexing = False

# index card if a repo is loaded
if st.session_state.indexed_repo:
    st.markdown(f"""
    <div class="index-card">
        <div class="index-card-repo">{st.session_state.indexed_repo}</div>
        <div class="index-card-meta">
            <span><b>{st.session_state.files_processed}</b> files read</span>
            <span><b>{st.session_state.chunks_created}</b> chunks indexed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ASK SECTION
# ============================================================
st.markdown('<div class="margin-label">ask it something</div>', unsafe_allow_html=True)

with st.form(key="ask_form", clear_on_submit=True):
    qcol1, qcol2 = st.columns([4, 1])
    with qcol1:
        question = st.text_input(
            "question", placeholder="how does dependency injection work?",
            label_visibility="collapsed"
        )
    with qcol2:
        ask_clicked = st.form_submit_button("Ask", use_container_width=True)

if ask_clicked and question:
    if not st.session_state.indexed_repo:
        st.warning("Point it at a repo first — there's nothing indexed yet.")
    else:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            '<div class="progress-line">⌗ searching the index, drafting an answer...</div>',
            unsafe_allow_html=True
        )
        try:
            resp = requests.post(
                f"{API_BASE}/ask",
                json={"question": question, "n_results": 5},
                timeout=60
            )
            thinking_placeholder.empty()
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.history.insert(0, {
                    "q": question, "a": data["answer"], "sources": data["sources"]
                })
            else:
                detail = resp.json().get("detail", "Unknown error")
                st.error(f"Couldn't answer that — {detail}")
        except requests.exceptions.ConnectionError:
            thinking_placeholder.empty()
            st.error("Backend isn't reachable. Is `uvicorn main:app --reload` running on port 8000?")
        except requests.exceptions.Timeout:
            thinking_placeholder.empty()
            st.error("The LLM call took too long. Try again.")

# ============================================================
# HISTORY / ANSWERS
# ============================================================
if not st.session_state.history:
    st.markdown("""
    <div class="empty-state">
        <div class="glyph">⌗</div>
        <div class="msg">no questions asked yet — the margins are still blank</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for entry in st.session_state.history:
        sources_html = "".join(
            f'<span class="source-tab">{s.split("::")[-1].strip() if "::" in s else s}</span>'
            for s in entry["sources"]
        )
        st.markdown(f"""
        <div class="answer-block">
            <div class="q">{entry['q']}</div>
            <div class="a">{entry['a']}</div>
            <div class="sources-label">found in</div>
            {sources_html}
        </div>
        """, unsafe_allow_html=True)