"""Ask Bhavin — Streamlit chat UI."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from src.config import DATA_DIR, INDEX_DIR
from src.ingest import build_index
from src.rag import ask

load_dotenv()

st.set_page_config(
    page_title="Ask Bhavin",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    h1 { color: #f8fafc !important; }
    .subtitle { color: #94a3b8; margin-bottom: 1.5rem; }

    /* Assistant chat — light text on dark background */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
        color: #e2e8f0 !important;
    }

    /* User chat bubble — dark text on light background */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }

    /* Sources expander & source snippets */
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary span {
        color: #94a3b8 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
        color: #cbd5e1 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stText"],
    [data-testid="stChatMessage"] [data-testid="stText"] pre {
        color: #e2e8f0 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #475569 !important;
    }

    /* Sidebar legibility on dark theme */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background-color: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        color: #94a3b8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ask Bhavin")
st.markdown(
    '<p class="subtitle">Ask anything about Bhavin Sen — skills, experience, projects, and reviews. '
    "Answers are grounded in a curated knowledge base built from public profiles.</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def ensure_index():
    meta = INDEX_DIR / "chunks.json"
    if not meta.exists():
        build_index()
        return "built"
    return "ready"


with st.sidebar:
    st.header("About this app")
    st.write("RAG demo: TF-IDF retrieval + Sarvam AI for answers.")
    st.write(f"**Sources:** `{DATA_DIR.name}/` markdown files")
    if os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY"):
        st.success("Sarvam API key detected — synthesized answers enabled.")
    else:
        st.warning("No Sarvam API key — retrieval-only mode.")

    if st.button("Rebuild knowledge index"):
        build_index()
        st.cache_resource.clear()
        st.success("Index rebuilt from data/*.md")

    with st.expander("Example questions"):
        st.markdown(
            """
            - What is Bhavin's experience with RAG and LangChain?
            - Where is he based and what is his hourly rate?
            - What industries has he worked in?
            - Summarize his Freelancer reputation.
            - What does a negative review say, and how should I interpret it?
            """
        )

ensure_index()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm Ask Bhavin. Ask me about Bhavin's skills, career, projects, or client feedback.",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']} (score {s['score']:.3f})")
                    st.text(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))

if prompt := st.chat_input("Ask about Bhavin…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                result = ask(prompt)
                st.markdown(result.answer)
                sources = [
                    {"source": c.source, "text": c.text, "score": c.score}
                    for c in result.chunks
                ]
                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            st.caption(f"📄 {s['source']} (score {s['score']:.3f})")
                            st.text(
                                s["text"][:400]
                                + ("…" if len(s["text"]) > 400 else "")
                            )
            except Exception as e:
                st.error(str(e))
                result = None
                sources = []

    entry = {
        "role": "assistant",
        "content": result.answer if result else str(e),
    }
    if result:
        entry["sources"] = sources
    st.session_state.messages.append(entry)
