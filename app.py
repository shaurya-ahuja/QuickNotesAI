"""
QuickNotes-AI: Offline Meeting Notetaker
100% Local - No Data Leaves Your Device

A powerful meeting assistant that records, transcribes, summarizes,
and extracts action items - all running locally on your machine.
"""

import streamlit as st
import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from st_audiorec import st_audiorec

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="QuickNotes-AI",
    page_icon="📝",
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
        # Use cached instance for performance and sync
        rag = get_rag_engine() 
        
        for i, m in enumerate(meetings_to_index):
            transcript = m.get('transcript') or ''
            summary = m.get('summary') or ''
            if transcript or summary:
                text = f"Title: {m['title']}\nDate: {m['date']}\n\n{transcript}\n\n{summary}"
                
                # metadata for better context
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
        background-color: rgba(128, 128, 128, 0.05); /* Works in Light & Dark */
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
    
    /* Status Badges */
    .recording-indicator {
        background-color: rgba(255, 59, 48, 0.1);
        color: #FF3B30;
        padding: 12px 20px;
        border-radius: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(255, 59, 48, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Status Colors */
    .status-available {
        color: #34C759; /* Apple Green */
        font-weight: 600;
    }
    .status-unavailable {
        color: #FF3B30; /* Apple Red */
        font-weight: 600;
    }
    
    /* Privacy Badge */
    .privacy-badge {
        background-color: rgba(0, 122, 255, 0.1);
        color: #007AFF;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border: 1px solid rgba(0, 122, 255, 0.2);
        margin-bottom: 24px;
    }

    /* Hide Streamlit Header/Toolbar to remove "fireworks"/deploy button */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stApp {
        margin-top: -60px; /* Pull content up since header is gone */
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    # Ensure page key exists with a default
    if 'page' not in st.session_state:
        st.session_state['page'] = '🎙️ New Meeting'
        
    defaults = {
        'recording': False,
        'recording_start_time': None,
        'current_transcript': None,
        'current_summary': None,
        'current_actions': [],
        'current_quotes': [],
        'audio_file': None,
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
        st.markdown('<div class="privacy-badge">🔒 100% Local - No Data Leaves Your Device</div>', unsafe_allow_html=True)
        
        st.title("📝 QuickNotes-AI")
        st.caption("Offline Meeting Notetaker")
        
        st.markdown("---")
        
        # Navigation using Buttons (Guaranteed Visibility & Box Look)
        # Using buttons prevents the CSS visibility issues with radio buttons
        nav_options = ["🎙️ New Meeting", "📚 Meeting History", "🔍 RAG Search", "⚙️ Settings"]
        
        for option in nav_options:
            # Check if this button is active
            is_active = (st.session_state.get('page') == option)
            
            # Use 'primary' type for active button to highlight it
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(option, key=f"nav_{option}", use_container_width=True, type=btn_type):
                st.session_state.page = option
                st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.subheader("System Status")
        
        status_items = [

            ("Whisper (Transcription)", WHISPER_AVAILABLE),
            ("Ollama (LLM)", OLLAMA_AVAILABLE),
            ("FAISS (RAG)", FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE),
        ]
        
        for name, available in status_items:
            icon = "✅" if available else "❌"
            color = "status-available" if available else "status-unavailable"
            st.markdown(f'{icon} <span class="{color}">{name}</span>', unsafe_allow_html=True)


# ============== Active Meeting Page ==============
def render_results():
    """Render transcription and summary results with edit capability."""
    # Header with Save Status
    col_title, col_save = st.columns([3, 1])
    with col_title:
        st.markdown("### 📝 Meeting Details")
        if not st.session_state.get('meeting_saved', False):
            st.caption("⚠️ Not saved yet - review and save below")
        else:
            st.caption("✅ Meeting saved")
    
    # Apple/Notion-style Tabs
    tab_transcript, tab_summary, tab_actions = st.tabs(["📝 Transcript (Editable)", "✨ AI Summary", "✅ Action Items"])
    
    # --- TAB 1: TRANSCRIPT (EDITABLE) ---
    with tab_transcript:
        if st.session_state.current_transcript:
            result = st.session_state.current_transcript
            
            # Language badge
            confidence = result.language_probability
            confidence_text = f"{confidence:.0%} confidence" if confidence > 0.01 else "auto-detected"
            st.caption(f"Detected language: {result.language} ({confidence_text})")
            
            # Get current transcript text - initialize if not set or empty
            if not st.session_state.get('edited_transcript'):
                transcription_service = get_transcription_service("base")
                st.session_state.edited_transcript = transcription_service.format_transcript_with_speakers(result)
                
                # Fallback if formatting failed/empty
                if not st.session_state.edited_transcript:
                    st.session_state.edited_transcript = result.full_text
            
            # Editable text area
            edited_text = st.text_area(
                "Edit Transcript",
                value=st.session_state.edited_transcript,
                height=400,
                label_visibility="collapsed",
                key="transcript_editor"
            )
            
            # Update if changed
            if edited_text != st.session_state.edited_transcript:
                st.session_state.edited_transcript = edited_text
                st.session_state.transcript_modified = True
            
            # Regenerate button
            if st.session_state.get('transcript_modified', False):
                st.warning("Transcript modified. Regenerate summary to update AI analysis.")
                if st.button("🔄 Regenerate Summary from Edited Transcript", type="primary", use_container_width=True):
                    with st.spinner("Regenerating summary..."):
                        try:
                            summarizer = get_summarization_service()
                            summary_res = summarizer.summarize(
                                st.session_state.edited_transcript,
                                language=result.language
                            )
                            st.session_state.current_summary = summary_res
                            
                            extractor = get_action_extractor()
                            actions = extractor.extract_from_structured(summary_res.action_items)
                            st.session_state.current_actions = actions
                            
                            st.session_state.transcript_modified = False
                            st.success("✅ Summary regenerated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to regenerate: {e}")
        else:
            st.info("No transcript available.")
    
    # --- TAB 2: SUMMARY ---
    with tab_summary:
        if st.session_state.current_summary:
            summary = st.session_state.current_summary
            
            # Key Points
            st.markdown("#### Key Points")
            for bullet in summary.summary_bullets:
                st.markdown(f"• {bullet}")
            
            # Key Quotes (styled as inset cards)
            if summary.key_quotes:
                st.markdown("#### 💬 Key Quotes")
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
        st.markdown("#### Action Items")
        
        if st.session_state.current_actions:
            for i, action in enumerate(st.session_state.current_actions):
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    completed = st.checkbox("Completed", key=f"action_{i}", value=action.completed, label_visibility="collapsed")
                    action.completed = completed
                with col2:
                    details = []
                    if action.assignee: details.append(f"👤 {action.assignee}")
                    if action.deadline: details.append(f"📅 {action.deadline}")
                    details_str = " • ".join(details)
                    
                    st.markdown(f"**{action.emoji} {action.task}**")
                    if details_str:
                        st.caption(details_str)
        else:
            st.info("No action items detected.")
    
    # --- SAVE SECTION ---
    st.markdown("---")
    
    # Tags Input
    st.markdown("#### 🏷️ Tags")
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
            if st.button("💾 Save Meeting", type="primary", use_container_width=True):
                save_current_meeting()
                st.session_state.meeting_saved = True
                st.rerun()
        with col_save2:
            if st.button("🗑️ Discard", type="secondary", use_container_width=True):
                # Clear current session
                st.session_state.current_transcript = None
                st.session_state.current_summary = None
                st.session_state.current_actions = []
                st.session_state.edited_transcript = None
                st.session_state.audio_file = None
                st.rerun()
    else:
        # Export options after saving
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📧 Email Summary", key="btn_email", use_container_width=True):
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
                            "📅 Export to Calendar",
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
    
    # Use edited transcript if available, otherwise use original
    if st.session_state.get('edited_transcript'):
        transcript_text = st.session_state.edited_transcript
    else:
        transcript_text = st.session_state.current_transcript.full_text
    
    summary_text = "\n".join(st.session_state.current_summary.summary_bullets) if st.session_state.current_summary else ""
    
    # Create meeting with custom or default name
    meeting_title = st.session_state.get('meeting_name', f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Get tags
    tags = st.session_state.get('new_meeting_tags', [])

    meeting_id = db.create_meeting(
        title=meeting_title,
        transcript=transcript_text,
        summary=summary_text,
        tags=tags,
        language=st.session_state.detected_language,
        audio_path=st.session_state.audio_file or ""
    )
    
    # Add action items
    for action in st.session_state.current_actions:
        db.add_action_item(
            meeting_id=meeting_id,
            task=action.task,
            assignee=action.assignee,
            deadline=action.deadline,
            emoji=action.emoji
        )
        
    # Index for RAG
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
                
            st.info("📚 Meeting indexed for semantic search!")
        except Exception as e:
            pass  # Silently fail if RAG not available
    
    st.session_state.current_meeting_id = meeting_id
    st.success(f"Meeting saved! ID: {meeting_id}")


# ============== Meeting History Page ==============
def render_history_page():
    """Render the meeting history page."""
    st.title("📚 Meeting History")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🏷️ Filter by Tag")
        all_tags = db.get_all_tags()
        selected_tag = st.selectbox(
            "Select tag",
            ["All"] + all_tags,
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("🔎 Search")
        search_query = st.text_input(
            "Search meetings",
            placeholder="Search by title, content...",
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Get meetings
    if search_query:
        meetings = db.search_meetings(search_query)
    elif selected_tag != "All":
        meetings = db.get_all_meetings(tag_filter=selected_tag)
    else:
        meetings = db.get_all_meetings()
    
    if not meetings:
        st.info("No meetings found. Record your first meeting!")
        return
    
    # Display meetings
    for meeting in meetings:
        with st.expander(f"📅 {meeting['title']} - {meeting['date'][:10]}", expanded=False):
            
            # Apple-style Tabs for History
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
                        key=f"transcript_hist_{meeting['id']}" # Fixed Duplicate ID
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
                                f"{action['emoji']} **{action['task']}**", 
                                value=bool(action['completed']),
                                key=f"hist_act_{action['id']}"
                            )
                            
                            if is_checked != bool(action['completed']):
                                db.toggle_action_item(action['id'])
                                st.rerun()
                                
                            details = []
                            if action['assignee']: details.append(f"👤 {action['assignee']}")
                            if action['deadline']: details.append(f"📅 {action['deadline']}")
                            if details:
                                st.caption(" • ".join(details))
                                
                        with col_act2:
                            if st.button("🗑️", key=f"del_act_{action['id']}", help="Delete Task"):
                                db.delete_action_item(action['id'])
                                st.rerun()
                else:
                    st.info("No action items.")

            st.markdown("---")
            col_del, _ = st.columns([0.2, 0.8])
            with col_del:
                if st.button("🗑️ Delete Meeting", key=f"del_mtg_{meeting['id']}", type="primary"):
                    db.delete_meeting(meeting['id'])
                    # Remove from RAG index
                    try:
                        rag = get_rag_engine()
                        with rag_lock:
                            rag.remove_source(f"meeting_{meeting['id']}")
                    except:
                        pass
                    st.rerun()



# ============== RAG Search Page ==============
def render_rag_page():
    """Render the RAG search page with fallback to text search."""
    st.title("🔍 Search Past Meetings")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Check if vector search is available
    rag_available = FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
    
    # Show search mode selector
    search_mode = st.radio(
        "Search Mode",
        ["📝 Text Search (Fast)", "🧠 Semantic Search (AI-powered)"] if rag_available else ["📝 Text Search (Fast)"],
        horizontal=True,
        help="Text search is fast and reliable. Semantic search uses AI but requires more resources."
    )
    
    use_semantic = "Semantic" in search_mode and rag_available
    
    st.markdown("---")
    
    if use_semantic:
        render_semantic_search()
    else:
        render_text_search()


def render_text_search():
    """Render simple text-based search using SQLite."""
    query = st.text_input("Search Query", placeholder="Enter keywords...")
    
    if query:
        results = db.search_meetings(query)
        st.subheader(f"Found {len(results)} matches")
        
        for meeting in results:
            with st.expander(f"{meeting['title']} ({meeting['date']})"):
                st.markdown(meeting['summary'] or "No summary")
                if st.button("Go to History", key=f"go_{meeting['id']}"):
                    st.session_state.page = "Menu" # Simple redirect not fully implemented in sidebar nav
                    st.info("Navigate to History tab to view full details")


def render_semantic_search():
    """Render RAG-based semantic search."""
    rag = get_rag_engine()
    
    # Get metrics data first
    sources = rag.get_indexed_sources()
    all_meetings = db.get_all_meetings()
    indexed_ids = {s.split('_')[-1] for s in sources if s.startswith('meeting_')}
    unindexed = [m for m in all_meetings if str(m['id']) not in indexed_ids]
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Indexed Chunks", rag.document_count)
    with col2:
        st.metric("📁 Sources", len(sources))
    
    # Index missing section (outside columns for clean rendering)
    # Index missing section
    if unindexed:
        st.info(f"ℹ️ {len(unindexed)} older meetings not indexed (new ones auto-index).")
        
        # Check background status
        if indexing_status["running"]:
            progress = indexing_status["progress"] / max(indexing_status["total"], 1)
            st.progress(progress, text=f"🔄 Indexing in background: {indexing_status['progress']}/{indexing_status['total']}")
            st.caption("You can leave this page - indexing will continue.")
            if st.button("🔄 Refresh Status"):
                st.rerun()
        else:
            if st.button("▶️ Start Background Indexing", key="start_bg_index"):
                # Start thread
                thread = threading.Thread(target=background_index_worker, args=(unindexed,))
                thread.start()
                st.rerun()
    else:
        st.success("✅ All meetings indexed")

    
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("📂 Upload Documents")
    st.caption("Add PDFs or text files to search alongside your meetings")
    
    uploaded_docs = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_docs:
        for doc in uploaded_docs:
            # Save temporarily and index
            os.makedirs("uploads", exist_ok=True)
            doc_path = os.path.join("uploads", doc.name)
            with open(doc_path, "wb") as f:
                f.write(doc.getbuffer())
            
            try:
                with st.spinner(f"Indexing {doc.name}..."):
                    chunks = rag.add_file(doc_path)
                    st.success(f"✅ Added {doc.name} ({chunks} chunks)")
            except Exception as e:
                st.error(f"Failed to index {doc.name}: {e}")
    
    st.markdown("---")
    
    # Query Section
    st.subheader("🔍 Ask a Question")
    query = st.text_area(
        "Ask about your meetings or uploaded documents",
        placeholder="What did we decide about the marketing budget?",
        label_visibility="collapsed"
    )
    
    # Button always visible
    if st.button("🧠 Ask AI", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching and thinking..."):
                # Get relevant context
                results = rag.search(query, top_k=3)
                
                if not results:
                    st.warning("No relevant information found. Try uploading more documents or recording meetings.")
                else:
                    context_text = "\n\n".join([r.document.content for r in results])
                    
                    # Generate answer
                    # Generate answer
                    try:
                        summarizer = get_summarization_service()
                        answer = summarizer.answer_question(query, context_text)
                        
                        st.markdown("### 🤖 AI Answer")
                        st.info(answer)
                    except Exception as e:
                        st.error(f"⚠️ AI Services Unavailable: {e}")
                        st.caption("On Streamlit Cloud, local Ollama is not available. Please run locally for full AI features.")
                    
                    st.markdown("### 📚 Sources")
                    for r in results:
                        with st.expander(f"{r.document.source} (Score: {r.score:.2f})"):
                            st.text(r.document.content[:500] + "...")


# ============== Settings Page ==============
def render_settings_page():
    st.title("⚙️ Settings")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    email_service = get_email_service()
    
    with st.form("email_settings"):
        st.subheader("📧 Email Configuration")
        sender = st.text_input("Gmail Address", value=email_service.config.sender_email or "")
        password = st.text_input("App Password", type="password", value=email_service.config.sender_password or "")
        
        if st.form_submit_button("Save Settings"):
            if sender and password:
                email_service.configure_from_preset("gmail", sender, password)
                st.success("Credentials saved!")
            else:
                st.error("Please fill all fields")
    
    st.markdown("---")
    

    
    # Danger Zone
    st.subheader("⚠️ Danger Zone")
    st.warning("These actions are irreversible.")
    
    with st.expander("🗑️ Reset Application Data"):
        st.markdown("This will permanently delete:")
        st.markdown("- All saved meetings and transcripts")
        st.markdown("- All indexed documents")
        st.markdown("- All tags and action items")
        
        confirm = st.checkbox("I understand that this action cannot be undone", key="reset_confirm")
        
        if st.button("🔴 Confirm Reset", disabled=not confirm):
            if db.reset_database():
                st.success("✅ Application data successfully reset.")
                # Clear session state for safety
                st.session_state.current_transcript = None
                st.session_state.current_summary = None
                time.sleep(2)
                st.rerun()
            else:
                st.error("Failed to reset database.")


# ============== Main App Entry ==============
def main():
    init_session_state()
    
    render_sidebar()
    page = st.session_state.page
    
    if page == "🎙️ New Meeting":
        st.title("🎙️ New Meeting")
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Browser Audio Recorder
        st.info("🎙️ Browser Audio: Click 'Start Recording', speak, then click 'Stop'.")
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            os.makedirs("uploads", exist_ok=True)
            # Use a consistent filename for the current session recording
            file_path = os.path.join("uploads", "browser_recording.wav")
            with open(file_path, "wb") as f:
                f.write(wav_audio_data)
            
            st.session_state.audio_file = file_path
            st.session_state.audio_source = "recorded (browser)"
            
            # Show success message
            st.success("✅ Audio captured successfully!")
            
        # Upload Fallback
        st.markdown("---")
        st.subheader("📂 Or Upload Audio")
        uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A)", type=['wav', 'mp3', 'm4a'])
        
        if uploaded_file:
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file = path
            st.session_state.audio_source = "uploaded"  # Track source
        
        # --- AUDIO READY SECTION ---
        st.markdown("---")
        
        if st.session_state.audio_file and not st.session_state.processing:
            audio_fname = os.path.basename(st.session_state.audio_file)
            source_label = st.session_state.get('audio_source', 'recorded')
            
            # Show which audio will be processed
            st.markdown(f"""
                <div style="background-color: rgba(0, 122, 255, 0.1); border: 1px solid #007AFF; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <strong>🎵 Audio Ready for Processing:</strong><br>
                    <span style="font-size: 1.1em;">{audio_fname}</span>
                    <span style="opacity: 0.7; margin-left: 10px;">({source_label})</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Meeting Name Input
            default_title = f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            meeting_name = st.text_input(
                "📝 Meeting Name",
                value=st.session_state.get('meeting_name', ''),
                placeholder=default_title,
                help="Optional: Give your meeting a custom name"
            )
            st.session_state.meeting_name = meeting_name if meeting_name.strip() else default_title
            
            # Process Button
            if st.button("⚡ Process Audio", type="primary", use_container_width=True):
                st.session_state.processing = True
                st.rerun()
        elif not st.session_state.audio_file:
            st.info("👆 Record audio or upload a file to get started.")

        # Processing Logic
        if st.session_state.processing:
            with st.status("Processing Meeting...", expanded=True) as status:
                try:
                    # 1. Transcribe
                    status.write("📝 Transcribing audio...")
                    transcriber = get_transcription_service()
                    transcript_res = transcriber.transcribe(st.session_state.audio_file)
                    st.session_state.current_transcript = transcript_res
                    
                    # 2. Summarize
                    status.write("🤖 Generating summary...")
                    summarizer = get_summarization_service()
                    summary_res = summarizer.summarize(transcript_res.full_text, language=transcript_res.language)
                    st.session_state.current_summary = summary_res
                    
                    # 3. Extract Actions
                    status.write("✅ Extracting actions...")
                    extractor = get_action_extractor()
                    actions = extractor.extract_from_structured(summary_res.action_items)
                    st.session_state.current_actions = actions
                    
                    status.update(label="Complete! Review below.", state="complete", expanded=False)
                    
                    # Reset states for new meeting
                    st.session_state.meeting_saved = False
                    st.session_state.edited_transcript = None
                    st.session_state.transcript_modified = False
                    
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.processing = False
                    st.rerun()

        # Render Results if available
        if st.session_state.current_transcript:
            render_results()

    elif page == "📚 Meeting History":
        render_history_page()
        
    elif page == "🔍 RAG Search":
        render_rag_page()
            
    elif page == "⚙️ Settings":
        render_settings_page()

if __name__ == "__main__":
    main()
