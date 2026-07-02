"""
QuickNotes: Offline Meeting Notetaker
"""

import streamlit as st
import os
import sys
import time
import base64
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from _recorder import audio_recorder

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="QuickNotes",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Lazy import flags
def check_module_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None


WHISPER_AVAILABLE = check_module_available("whisper")
OLLAMA_AVAILABLE = check_module_available("ollama")
FAISS_AVAILABLE = check_module_available("faiss")
SENTENCE_TRANSFORMERS_AVAILABLE = check_module_available("sentence_transformers")

# Global background indexing state
indexing_status = {"running": False, "progress": 0, "total": 0}
rag_lock = threading.Lock()

def background_index_worker(meetings_to_index):
    """Run indexing in a background thread."""
    global indexing_status
    indexing_status["running"] = True
    indexing_status["total"] = len(meetings_to_index)
    indexing_status["progress"] = 0

    try:
        rag = get_rag_engine()

        for i, m in enumerate(meetings_to_index):
            transcript = m.get('transcript') or ''
            summary = m.get('summary') or ''
            if transcript or summary:
                text = f"Title: {m['title']}\nDate: {m['date']}\n\n{transcript}\n\n{summary}"
                meta = {"title": m['title'], "date": m['date']}
                with rag_lock:
                    rag.add_text(text, source=f"meeting_{m['id']}", metadata=meta)
            indexing_status["progress"] = i + 1

    except Exception as e:
        print(f"Background indexing error: {e}")
    finally:
        indexing_status["running"] = False

# Apple/Notion-style Minimalist CSS (Adaptive)
st.markdown("""
<style>
    /* Global Typography */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Clean Cards (Adaptive) */
    .css-1d391kg, .css-12oz5g7, div[data-testid="stExpander"] {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        box-shadow: none;
    }

    /* Buttons - Apple Pills */
    .stButton button {
        border-radius: 999px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s ease;
    }

    /* Primary Action Buttons */
    .stButton button[kind="primary"] {
        background-color: #007AFF !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.2);
    }

    /* Secondary/Outline Buttons */
    .stButton button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        color: inherit !important;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: rgba(128, 128, 128, 0.1) !important;
        border-color: rgba(128, 128, 128, 0.5) !important;
    }

    /* Tabs - Clean Minimalist */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border: none;
        color: inherit;
        font-weight: 500;
        opacity: 0.7;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #007AFF !important;
        font-weight: 600;
        opacity: 1;
        border-bottom: 2px solid #007AFF;
    }

    /* Inputs */
    .stTextInput input, .stSelectbox [data-baseweb="select"] {
        background-color: rgba(128, 128, 128, 0.05) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
    }

    /* Status Colors */
    .status-available {
        color: #34C759;
        font-weight: 600;
    }
    .status-unavailable {
        color: #FF3B30;
        font-weight: 600;
    }

    /* Hide Streamlit Header/Toolbar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'page' not in st.session_state:
        st.session_state['page'] = 'New Meeting'

    defaults = {
        'recording': False,
        'recording_start_time': None,
        'current_transcript': None,
        'current_summary': None,
        'current_actions': [],
        'current_quotes': [],
        'audio_file': None,
        'audio_source': None,
        'detected_language': 'en',
        'processing': False,
        'rag_results': None,
        'services_loaded': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Import src modules (with caching)
from src.database import Database

@st.cache_resource
def get_database():
    return Database()


@st.cache_resource
def get_transcription_service(model_name: str = "base"):
    """Get transcription service (loads Whisper - HEAVY)."""
    from src.transcription import get_transcription_service as _get_service
    return _get_service(model_name)

@st.cache_resource
def get_summarization_service():
    """Get summarization service (loads Ollama client)."""
    from src.summarizer import get_summarization_service as _get_service
    return _get_service()

def get_ollama_url():
    """Get current Ollama URL from session state."""
    return st.session_state.get('ollama_url', 'http://localhost:11434')

@st.cache_resource
def get_action_extractor():
    """Get action extractor (lightweight)."""
    from src.action_extractor import get_action_extractor as _get_extractor
    return _get_extractor()

@st.cache_resource
def get_rag_engine():
    """Get RAG engine (loads SentenceTransformers + FAISS - HEAVY)."""
    from src.rag_engine import get_rag_engine as _get_engine
    return _get_engine()

@st.cache_resource
def get_email_service():
    """Get email service (lightweight)."""
    from src.email_service import get_email_service as _get_service
    return _get_service()

@st.cache_resource
def get_export_service():
    """Get export service (lightweight)."""
    from src.export_utils import get_export_service as _get_service
    return _get_service()

# Get lightweight services immediately
db = get_database()


# ============== Sidebar ==============
def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.title("QuickNotes")
        st.caption("Offline Meeting Notetaker")

        st.markdown("---")

        nav_options = ["New Meeting", "Meeting History", "Search", "Settings", "How It Works"]

        for option in nav_options:
            is_active = (st.session_state.get('page') == option)
            btn_type = "primary" if is_active else "secondary"

            if st.button(option, key=f"nav_{option}", use_container_width=True, type=btn_type):
                st.session_state.page = option
                st.rerun()

        st.markdown("---")

        # System Status
        st.subheader("System status")

        status_items = [
            ("Whisper (Transcription)", WHISPER_AVAILABLE),
            ("Ollama (LLM)", OLLAMA_AVAILABLE),
            ("FAISS (RAG)", FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE),
        ]

        for name, available in status_items:
            dot_color = "#34C759" if available else "#FF3B30"
            label = "available" if available else "not installed"
            st.markdown(
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:{dot_color};margin-right:6px;vertical-align:middle;"></span>'
                f'<span style="font-size:0.88em;">{name}</span> '
                f'<span style="font-size:0.78em;opacity:0.55;">({label})</span>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Server Configuration
        st.subheader("Server")
        ollama_url = st.text_input(
            "Ollama URL",
            value=st.session_state.get('ollama_url', 'http://localhost:11434'),
            help="Enter your Ollama server URL. Use your local IP for remote access.",
            key="ollama_url_input"
        )
        st.session_state['ollama_url'] = ollama_url

        with st.expander("Remote setup guide"):
            st.markdown("""
**To use from Streamlit Cloud:**

Download ngrok from https://ngrok.com/
Download ollama from https://ollama.com/

**Option 1: ngrok (Recommended)**
```bash
# Terminal 1: Start Ollama
OLLAMA_HOST=0.0.0.0 \\
OLLAMA_ORIGINS="https://quicknotesai.streamlit.app" \\
ollama serve

# Terminal 2: Expose with ngrok
ngrok http 11434
```
Copy the ngrok URL and paste it above.

**Option 2: Direct IP (requires port forwarding)**
```bash
OLLAMA_HOST=0.0.0.0 \\
OLLAMA_ORIGINS="https://quicknotesai.streamlit.app" \\
ollama serve
```
Enter `http://YOUR_PUBLIC_IP:11434`
            """)


# ============== Active Meeting Page ==============
def render_results():
    """Render transcription and summary results with edit capability."""
    col_title, col_save = st.columns([3, 1])
    with col_title:
        st.markdown("### Meeting details")
        if not st.session_state.get('meeting_saved', False):
            st.caption("Not saved yet — review and save below")
        else:
            st.caption("Meeting saved")

    tab_transcript, tab_summary, tab_actions = st.tabs(["Transcript", "Summary", "Actions"])

    # --- TAB 1: TRANSCRIPT (EDITABLE) ---
    with tab_transcript:
        if st.session_state.current_transcript:
            result = st.session_state.current_transcript

            confidence = result.language_probability
            confidence_text = f"{confidence:.0%} confidence" if confidence > 0.01 else "auto-detected"
            st.caption(f"Detected language: {result.language} ({confidence_text})")

            if not st.session_state.get('edited_transcript'):
                transcription_service = get_transcription_service("base")
                st.session_state.edited_transcript = transcription_service.format_transcript_with_speakers(result)

                if not st.session_state.edited_transcript:
                    st.session_state.edited_transcript = result.full_text

            edited_text = st.text_area(
                "Edit Transcript",
                value=st.session_state.edited_transcript,
                height=400,
                label_visibility="collapsed",
                key="transcript_editor"
            )

            if edited_text != st.session_state.edited_transcript:
                st.session_state.edited_transcript = edited_text
                st.session_state.transcript_modified = True

            if st.session_state.get('transcript_modified', False):
                st.warning("Transcript modified. Regenerate summary to update.")
                if st.button("Regenerate summary", type="primary", use_container_width=True):
                    with st.spinner("Regenerating summary..."):
                        try:
                            summarizer = get_summarization_service()
                            summarizer.set_host(get_ollama_url())
                            summary_res = summarizer.summarize(
                                st.session_state.edited_transcript,
                                language=result.language
                            )
                            st.session_state.current_summary = summary_res

                            extractor = get_action_extractor()
                            actions = extractor.extract_from_structured(summary_res.action_items)
                            st.session_state.current_actions = actions

                            st.session_state.transcript_modified = False
                            st.success("Summary regenerated.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to regenerate: {e}")
        else:
            st.info("No transcript available.")

    # --- TAB 2: SUMMARY ---
    with tab_summary:
        if st.session_state.current_summary:
            summary = st.session_state.current_summary

            st.markdown("#### Key points")
            for bullet in summary.summary_bullets:
                st.markdown(f"• {bullet}")

            if summary.key_quotes:
                st.markdown("#### Key quotes")
                for quote in summary.key_quotes:
                    speaker = quote.get('speaker', 'Speaker')
                    text = quote.get('quote', '')
                    st.markdown(f'''
                        <div style="background-color: rgba(128, 128, 128, 0.05); border-left: 3px solid #007AFF; padding: 12px; border-radius: 0 8px 8px 0; margin: 8px 0;">
                            <span style="font-style: italic;">"{text}"</span><br>
                            <span style="font-size: 0.85em; opacity: 0.7;">— {speaker}</span>
                        </div>
                    ''', unsafe_allow_html=True)
        else:
            if st.session_state.processing:
                st.info("Generating summary...")
            else:
                st.info("No summary available yet.")

    # --- TAB 3: ACTIONS ---
    with tab_actions:
        st.markdown("#### Action items")

        if st.session_state.current_actions:
            for i, action in enumerate(st.session_state.current_actions):
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    completed = st.checkbox("Completed", key=f"action_{i}", value=action.completed, label_visibility="collapsed")
                    action.completed = completed
                with col2:
                    details = []
                    if action.assignee: details.append(action.assignee)
                    if action.deadline: details.append(action.deadline)
                    details_str = " · ".join(details)

                    st.markdown(f"**{action.task}**")
                    if details_str:
                        st.caption(details_str)
        else:
            st.info("No action items detected.")

    # --- SAVE SECTION ---
    st.markdown("---")

    st.markdown("#### Tags")
    all_tags = db.get_all_tags() or ["Work", "Personal", "Team", "Project"]
    st.multiselect(
        "Add tags",
        options=list(set(all_tags + ["Work", "Personal", "Urgent"])),
        key="new_meeting_tags",
        placeholder="Select or type tags..."
    )

    st.markdown("---")

    if not st.session_state.get('meeting_saved', False):
        col_save1, col_save2 = st.columns([2, 1])
        with col_save1:
            if st.button("Save meeting", type="primary", use_container_width=True):
                save_current_meeting()
                st.session_state.meeting_saved = True
                st.rerun()
        with col_save2:
            if st.button("Discard", type="secondary", use_container_width=True):
                st.session_state.current_transcript = None
                st.session_state.current_summary = None
                st.session_state.current_actions = []
                st.session_state.edited_transcript = None
                st.session_state.audio_file = None
                st.rerun()
    else:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("Email summary", key="btn_email", use_container_width=True):
                st.info("Please configure email settings first.")
        with col_exp2:
            if st.session_state.current_actions:
                try:
                    export_service = get_export_service()
                    ics_bytes = export_service.get_ics_bytes(
                        [a.to_dict() for a in st.session_state.current_actions],
                        "Meeting Actions"
                    )
                    if ics_bytes:
                        st.download_button(
                            "Export to calendar",
                            data=ics_bytes,
                            file_name="actions.ics",
                            mime="text/calendar",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")


def save_current_meeting():
    """Save the current meeting to database."""
    if not st.session_state.current_transcript:
        st.warning("No meeting to save")
        return

    if st.session_state.get('edited_transcript'):
        transcript_text = st.session_state.edited_transcript
    else:
        transcript_text = st.session_state.current_transcript.full_text

    summary_text = "\n".join(st.session_state.current_summary.summary_bullets) if st.session_state.current_summary else ""
    meeting_title = st.session_state.get('meeting_name', f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    tags = st.session_state.get('new_meeting_tags', [])

    meeting_id = db.create_meeting(
        title=meeting_title,
        transcript=transcript_text,
        summary=summary_text,
        tags=tags,
        language=st.session_state.detected_language,
        audio_path=st.session_state.audio_file or ""
    )

    for action in st.session_state.current_actions:
        db.add_action_item(
            meeting_id=meeting_id,
            task=action.task,
            assignee=action.assignee,
            deadline=action.deadline,
            emoji=action.emoji
        )

    if FAISS_AVAILABLE:
        try:
            rag = get_rag_engine()
            date_str = datetime.now().strftime('%Y-%m-%d')
            meeting_text = f"Title: {meeting_title}\nDate: {date_str}\n\n{transcript_text}\n\n{summary_text}"
            with rag_lock:
                rag.add_text(
                    meeting_text,
                    source=f"meeting_{meeting_id}",
                    metadata={"title": meeting_title, "date": date_str}
                )
            st.info("Meeting indexed for search.")
        except Exception:
            pass

    st.session_state.current_meeting_id = meeting_id
    st.toast("Meeting saved", icon="💾")
    st.success(f"Meeting saved! ID: {meeting_id}")


# ============== Meeting History Page ==============
def render_history_page():
    """Render the meeting history page."""
    st.title("Meeting history")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filter by tag")
        all_tags = db.get_all_tags()
        selected_tag = st.selectbox(
            "Select tag",
            ["All"] + all_tags,
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("Search")
        search_query = st.text_input(
            "Search meetings",
            placeholder="Search by title, content...",
            label_visibility="collapsed"
        )

    st.markdown("---")

    if search_query:
        meetings = db.search_meetings(search_query)
    elif selected_tag != "All":
        meetings = db.get_all_meetings(tag_filter=selected_tag)
    else:
        meetings = db.get_all_meetings()

    if not meetings:
        st.info("No meetings found. Record your first meeting!")
        return

    for meeting in meetings:
        with st.expander(f"{meeting['title']} — {meeting['date'][:10]}", expanded=False):

            h_tab_sum, h_tab_trans, h_tab_act = st.tabs(["Summary", "Transcript", "Actions"])

            with h_tab_sum:
                if meeting['summary']:
                    bullets = [b.strip() for b in meeting['summary'].split('\n') if b.strip()]
                    for b in bullets:
                        st.markdown(f"• {b}")
                else:
                    st.info("No summary available.")

            with h_tab_trans:
                if meeting['transcript']:
                    st.text_area(
                        "Full Transcript",
                        value=meeting['transcript'],
                        height=300,
                        disabled=True,
                        key=f"transcript_hist_{meeting['id']}"
                    )
                else:
                    st.info("No transcript available.")

            with h_tab_act:
                actions = db.get_action_items(meeting['id'])
                if actions:
                    for action in actions:
                        col_act1, col_act2 = st.columns([0.8, 0.2])
                        with col_act1:
                            is_checked = st.checkbox(
                                f"**{action['task']}**",
                                value=bool(action['completed']),
                                key=f"hist_act_{action['id']}"
                            )

                            if is_checked != bool(action['completed']):
                                db.toggle_action_item(action['id'])
                                st.rerun()

                            details = []
                            if action['assignee']: details.append(action['assignee'])
                            if action['deadline']: details.append(action['deadline'])
                            if details:
                                st.caption(" · ".join(details))

                        with col_act2:
                            if st.button("Delete", key=f"del_act_{action['id']}", help="Delete task"):
                                db.delete_action_item(action['id'])
                                st.rerun()
                else:
                    st.info("No action items.")

            st.markdown("---")
            col_del, _ = st.columns([0.2, 0.8])
            with col_del:
                if st.button("Delete meeting", key=f"del_mtg_{meeting['id']}", type="primary"):
                    db.delete_meeting(meeting['id'])
                    try:
                        rag = get_rag_engine()
                        with rag_lock:
                            rag.remove_source(f"meeting_{meeting['id']}")
                    except Exception:
                        pass
                    st.rerun()


# ============== Search Page ==============
def render_rag_page():
    """Render the RAG search page with fallback to text search."""
    st.title("Search")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    rag_available = FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE

    search_mode = st.radio(
        "Search Mode",
        ["Text search", "Semantic search"] if rag_available else ["Text search"],
        horizontal=True,
        help="Text search is fast and reliable. Semantic search finds meaning-based matches."
    )

    use_semantic = "Semantic" in search_mode and rag_available

    st.markdown("---")

    if use_semantic:
        render_semantic_search()
    else:
        render_text_search()


def render_text_search():
    """Render simple text-based search using SQLite."""
    query = st.text_input("Search query", placeholder="Enter keywords...")

    if query:
        results = db.search_meetings(query)
        st.subheader(f"Found {len(results)} matches")

        for meeting in results:
            with st.expander(f"{meeting['title']} ({meeting['date']})"):
                st.markdown(meeting['summary'] or "No summary")
                if st.button("Go to history", key=f"go_{meeting['id']}"):
                    st.session_state.page = "Meeting History"
                    st.info("Navigate to Meeting History to view full details")


def render_semantic_search():
    """Render RAG-based semantic search."""
    rag = get_rag_engine()

    sources = rag.get_indexed_sources()
    all_meetings = db.get_all_meetings()
    indexed_ids = {s.split('_')[-1] for s in sources if s.startswith('meeting_')}
    unindexed = [m for m in all_meetings if str(m['id']) not in indexed_ids]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Indexed chunks", rag.document_count)
    with col2:
        st.metric("Sources", len(sources))

    if unindexed:
        st.info(f"{len(unindexed)} older meetings not yet indexed (new ones auto-index).")

        if indexing_status["running"]:
            progress = indexing_status["progress"] / max(indexing_status["total"], 1)
            st.progress(progress, text=f"Indexing... {indexing_status['progress']}/{indexing_status['total']}")
            st.caption("You can leave this page — indexing will continue.")
            if st.button("Refresh status"):
                st.rerun()
        else:
            if st.button("Start background indexing", key="start_bg_index"):
                thread = threading.Thread(target=background_index_worker, args=(unindexed,))
                thread.start()
                st.rerun()
    else:
        st.success("All meetings indexed.")

    st.markdown("---")

    st.subheader("Upload documents")
    st.caption("Add PDFs or text files to search alongside your meetings")

    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_docs:
        for doc in uploaded_docs:
            os.makedirs("uploads", exist_ok=True)
            doc_path = os.path.join("uploads", doc.name)
            with open(doc_path, "wb") as f:
                f.write(doc.getbuffer())

            try:
                with st.spinner(f"Indexing {doc.name}..."):
                    chunks = rag.add_file(doc_path)
                    st.success(f"{doc.name} indexed ({chunks} chunks).")
            except Exception as e:
                st.error(f"Failed to index {doc.name}: {e}")

    st.markdown("---")

    st.subheader("Ask a question")

    # --- Retrieval controls (hybrid + re-ranking) ---
    from src.rag_engine import BM25_AVAILABLE

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + vectors)",
            value=BM25_AVAILABLE,
            disabled=not BM25_AVAILABLE,
            help="Fuse keyword (BM25) and semantic (vector) retrieval with Reciprocal Rank Fusion."
        )
    with opt_col2:
        reranker_ready = rag.reranker.is_available
        use_reranker = st.toggle(
            "Cross-encoder re-ranking",
            value=reranker_ready,
            disabled=not reranker_ready,
            help="Re-score candidates with a cross-encoder for sharper relevance. Downloads a small model on first use."
        )
    if not reranker_ready:
        st.caption("Re-ranker unavailable — falling back to hybrid retrieval. Ensure sentence-transformers is installed and the model can download.")

    query = st.text_area(
        "Ask about your meetings or uploaded documents",
        placeholder="What did we decide about the marketing budget?",
        label_visibility="collapsed"
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            results = []
            answer = None
            llm_error = None
            method = ""

            with st.status("Searching your knowledge base...", expanded=True) as status:
                try:
                    status.write("Retrieving relevant passages...")
                    results = rag.search(
                        query,
                        top_k=3,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                    )
                except Exception as e:
                    status.update(label="Retrieval failed", state="error")
                    st.error(f"Search failed: {e}")

                if results:
                    method = results[0].method
                    status.write(f"Retrieved {len(results)} passages via **{method}**.")
                    context_text = "\n\n".join([r.document.content for r in results])

                    try:
                        status.write("Generating answer with the LLM...")
                        summarizer = get_summarization_service()
                        summarizer.set_host(get_ollama_url())
                        answer = summarizer.answer_question(query, context_text)
                        status.update(label=f"Done — retrieved via {method}", state="complete", expanded=False)
                    except Exception as e:
                        llm_error = str(e)
                        status.update(label="Passages retrieved, LLM unavailable", state="error")
                elif not results:
                    status.update(label="No matches found", state="error")

            if not results:
                st.warning("No relevant information found. Try uploading more documents or recording meetings.")
            else:
                st.toast(f"Found {len(results)} passages ({method})", icon="✅")

                if answer:
                    st.markdown("### Answer")
                    st.info(answer)
                elif llm_error:
                    st.error(f"LLM services unavailable: {llm_error}")
                    st.caption("On Streamlit Cloud, local Ollama is not available. Run locally or set a remote Ollama URL to generate answers.")

                st.markdown("### Sources")
                st.caption("Scores — **rerank**: cross-encoder relevance · **dense**: vector cosine · **bm25**: keyword score")
                for r in results:
                    label = r.document.source
                    if r.rerank_score is not None:
                        label += f" · rerank {r.rerank_score:.2f}"
                    label += f" · dense {r.dense_score:.2f}"
                    if r.sparse_score:
                        label += f" · bm25 {r.sparse_score:.1f}"
                    with st.expander(label):
                        st.text(r.document.content[:500] + "...")


# ============== Settings Page ==============
def render_settings_page():
    st.title("Settings")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    email_service = get_email_service()

    with st.form("email_settings"):
        st.subheader("Email configuration")
        sender = st.text_input("Gmail address", value=email_service.config.sender_email or "")
        password = st.text_input("App password", type="password", value=email_service.config.sender_password or "")

        if st.form_submit_button("Save settings"):
            if sender and password:
                email_service.configure_from_preset("gmail", sender, password)
                st.success("Credentials saved.")
            else:
                st.error("Please fill all fields")

    st.markdown("---")

    # Danger Zone
    st.subheader("Danger zone")
    st.warning("These actions are irreversible.")

    with st.expander("Reset application data"):
        st.markdown("This will permanently delete:")
        st.markdown("- All saved meetings and transcripts")
        st.markdown("- All indexed documents")
        st.markdown("- All tags and action items")

        confirm = st.checkbox("I understand that this action cannot be undone", key="reset_confirm")

        if st.button("Confirm reset", disabled=not confirm):
            if db.reset_database():
                st.success("Application data reset.")
                st.session_state.current_transcript = None
                st.session_state.current_summary = None
                time.sleep(2)
                st.rerun()
            else:
                st.error("Failed to reset database.")


# ============== How It Works Page ==============
def render_how_it_works_page():
    """Render the How It Works reference page."""
    st.title("How it works")
    st.caption("A plain-English guide to the pipeline and models behind QuickNotes.")

    st.markdown("---")

    # ── Flow diagram ─────────────────────────────────────────────────────────
    st.subheader("End-to-end pipeline")

    st.markdown("""
<style>
.hiw-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 0.75rem 0; }
.hiw-box {
    background: rgba(0, 122, 255, 0.07);
    border: 1px solid rgba(0, 122, 255, 0.2);
    border-radius: 10px;
    padding: 9px 15px;
    font-size: 0.84rem;
    font-weight: 500;
    line-height: 1.3;
}
.hiw-box-neutral {
    background: rgba(128, 128, 128, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 10px;
    padding: 9px 15px;
    font-size: 0.84rem;
    line-height: 1.3;
}
.hiw-arrow { font-size: 1rem; opacity: 0.45; }
</style>

<p style="font-size:0.82rem;opacity:0.6;margin-bottom:0.5rem;">Recording &amp; transcription</p>
<div class="hiw-flow">
  <div class="hiw-box">Record or upload audio</div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box-neutral">Whisper (base model)<br><span style="font-size:0.76rem;opacity:0.55;">speech-to-text, locally</span></div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box-neutral">Transcript + speaker segments</div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box">Ollama (local LLM)<br><span style="font-size:0.76rem;opacity:0.55;">summary, actions, quotes</span></div>
</div>

<p style="font-size:0.82rem;opacity:0.6;margin-bottom:0.5rem;margin-top:1rem;">Indexing &amp; search</p>
<div class="hiw-flow">
  <div class="hiw-box-neutral">Transcript text</div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box-neutral">500-char chunks<br><span style="font-size:0.76rem;opacity:0.55;">50-char overlap</span></div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box-neutral">all-MiniLM-L6-v2<br><span style="font-size:0.76rem;opacity:0.55;">384-dim embeddings</span></div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box-neutral">FAISS vector index<br><span style="font-size:0.76rem;opacity:0.55;">saved to disk</span></div>
  <span class="hiw-arrow">→</span>
  <div class="hiw-box">Semantic search<br><span style="font-size:0.76rem;opacity:0.55;">top-3 chunks → LLM</span></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── RAG explanation ───────────────────────────────────────────────────────
    st.subheader("What is RAG?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Retrieval-Augmented Generation** is how the search page gives accurate answers instead of guesses.

Without RAG, asking an LLM "what was our Q3 budget?" would produce a hallucinated answer — the model has no knowledge of your meetings.

With RAG:
1. Your question is converted to an embedding (a vector of numbers representing meaning)
2. FAISS finds the 3 most similar chunks from your indexed meetings
3. Those chunks are injected into the LLM prompt as context
4. The LLM answers based on *your actual meeting content*

This dramatically reduces hallucinations and makes answers citable.
""")

    with col2:
        st.markdown("""
**Why it matters**

A plain LLM call costs nothing extra and can answer general questions — but it cannot know what was said in your meetings last Tuesday.

RAG bridges that gap: it retrieves the relevant slice of your private knowledge base and hands it to the LLM as a grounded reference.

The model never needs to "remember" your meetings. It reads the relevant chunk fresh each time, making answers more reliable and easier to verify against the source.
""")

    st.markdown("---")

    # ── Semantic vs text search ───────────────────────────────────────────────
    st.subheader("Semantic search vs text search")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
**Text search**

Runs a `LIKE` query against the SQLite database. Fast and always available, but purely keyword-based.

A search for *"Q3 costs"* won't match a meeting that said *"third quarter expenses"* — no shared words.

Use text search when you remember a specific word or phrase that appeared in the meeting.
""")

    with col4:
        st.markdown("""
**Semantic search**

Encodes your query with `all-MiniLM-L6-v2` into a 384-dimensional vector. FAISS finds the nearest stored vectors by cosine distance.

A search for *"Q3 costs"* will match *"third quarter expenses"* because the model understands that these phrases mean the same thing, even without word overlap.

Use semantic search when you remember what was *discussed* but not the exact wording.
""")

    st.markdown("---")

    # ── Source citations ──────────────────────────────────────────────────────
    st.subheader("How source citations work")

    st.markdown("""
Each search result displays a **source ID** (e.g., `meeting_42`) and a **similarity score** (0.0–1.0).

- The source ID maps to a specific meeting in the database. `meeting_42` means the row with `id = 42`.
- The score is cosine similarity between your query embedding and the stored chunk embedding. Scores above 0.65 typically indicate a relevant match; below 0.4 indicates a weak or coincidental match.
- The preview shows the first 500 characters of the matching chunk. The full chunk may span a longer passage in the original transcript.

If you get poor results, try rephrasing your question or indexing meetings that haven't been indexed yet.
""")

    st.markdown("---")

    # ── Privacy note ─────────────────────────────────────────────────────────
    st.subheader("Privacy")

    st.markdown("""
Everything runs on your machine. No data is sent to any external service:

| Component | Runs locally |
|---|---|
| Whisper (transcription) | Yes — model downloaded once to `~/.cache/whisper` |
| Ollama (summarization, Q&A) | Yes — models stored in `~/.ollama` |
| SentenceTransformers (embeddings) | Yes — model cached to `~/.cache/huggingface` |
| FAISS (vector index) | Yes — index saved to `data/vector_store/` |
| SQLite (meeting database) | Yes — `data/meetings.db` |
| Audio files | Yes — stored in `uploads/` |

The only exception is if you run the app on Streamlit Cloud and point Ollama at a remote server via ngrok — in that case, your questions and retrieved context are sent over the network to your server.
""")

    st.markdown("---")

    # ── Tips ─────────────────────────────────────────────────────────────────
    st.subheader("Tips for better results")

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("""
**What works well**

- Specific questions: *"What was decided about the marketing budget?"*
- Named entities: *"What did Sarah say about the timeline?"*
- Action items: *"What tasks were assigned to the engineering team?"*
- Decisions: *"Did we agree to delay the launch?"*
- Index all your meetings before running a broad search
""")

    with col6:
        st.markdown("""
**What works less well**

- Vague questions: *"Summarize everything"* — better to ask `st.caption` specific
- Very short recordings (under ~30 seconds) — Whisper may produce sparse output
- Background noise — consider re-recording or using the `small` or `medium` Whisper model for noisy audio
- Questions that span many unrelated meetings — semantic search returns the top-3 chunks, so coverage is limited

Change the Whisper model in `get_transcription_service()` in `app.py` to trade accuracy for speed.
""")


# ============== Main App Entry ==============
def main():
    init_session_state()

    render_sidebar()
    page = st.session_state.page

    if page == "New Meeting":
        st.title("New meeting")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Browser audio recorder
        st.caption("Click 'Start Recording', speak, then click 'Stop'.")
        rec_result = audio_recorder()
        if rec_result == "__reset__":
            st.session_state.audio_file = None
            st.session_state.audio_source = None
        elif rec_result is not None:
            try:
                audio_bytes = base64.b64decode(rec_result)
                os.makedirs("uploads", exist_ok=True)
                file_path = os.path.join("uploads", "browser_recording.webm")
                with open(file_path, "wb") as f:
                    f.write(audio_bytes)
                st.session_state.audio_file = file_path
                st.session_state.audio_source = "recorded"
                st.success("Audio captured.")
            except Exception:
                pass

        st.markdown("---")
        st.subheader("Upload audio")
        uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'm4a'])

        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file = path
            st.session_state.audio_source = "uploaded"

        st.markdown("---")

        if st.session_state.audio_file and not st.session_state.processing:
            audio_fname = os.path.basename(st.session_state.audio_file)
            source_label = st.session_state.get('audio_source', 'recorded')

            st.markdown(f"""
                <div style="background-color: rgba(0, 122, 255, 0.1); border: 1px solid #007AFF; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <strong>Audio ready:</strong><br>
                    <span style="font-size: 1.1em;">{audio_fname}</span>
                    <span style="opacity: 0.7; margin-left: 10px;">({source_label})</span>
                </div>
            """, unsafe_allow_html=True)

            default_title = f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            meeting_name = st.text_input(
                "Meeting name",
                value=st.session_state.get('meeting_name', ''),
                placeholder=default_title,
                help="Optional: give your meeting a custom name"
            )
            st.session_state.meeting_name = meeting_name if meeting_name.strip() else default_title

            if st.button("Process audio", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.rerun()
        elif not st.session_state.audio_file:
            st.info("Record audio or upload a file to get started.")

        if st.session_state.processing:
            with st.status("Processing meeting...", expanded=True) as status:
                try:
                    status.write("Transcribing audio...")
                    transcriber = get_transcription_service()
                    transcript_res = transcriber.transcribe(st.session_state.audio_file)
                    st.session_state.current_transcript = transcript_res

                    status.write("Generating summary...")
                    summarizer = get_summarization_service()
                    summarizer.set_host(get_ollama_url())
                    summary_res = summarizer.summarize(transcript_res.full_text, language=transcript_res.language)
                    st.session_state.current_summary = summary_res

                    status.write("Extracting actions...")
                    extractor = get_action_extractor()
                    actions = extractor.extract_from_structured(summary_res.action_items)
                    st.session_state.current_actions = actions

                    status.update(label="Complete! Review below.", state="complete", expanded=False)
                    st.toast("Meeting processed", icon="✅")

                    st.session_state.meeting_saved = False
                    st.session_state.edited_transcript = None
                    st.session_state.transcript_modified = False

                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.processing = False
                    st.rerun()

        if st.session_state.current_transcript:
            render_results()

    elif page == "Meeting History":
        render_history_page()

    elif page == "Search":
        render_rag_page()

    elif page == "Settings":
        render_settings_page()

    elif page == "How It Works":
        render_how_it_works_page()

if __name__ == "__main__":
    main()
