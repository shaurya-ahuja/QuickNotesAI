import streamlit as st
import os
import sys
import time
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- Page Config (Must be first) ---
st.set_page_config(
    page_title="QuickNotes-AI", 
    page_icon="📝", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Imports (Modular) ---
from src.ui.sidebar import render_sidebar
from src.ui.dashboard import render_main_content, render_history_page, render_rag_page, render_settings_page
from src.ui.record import render_record_page
import src.controller as controller

# --- Service Definitions (Lazy Loading) ---
def check_module_available(module_name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module_name) is not None

# Availability Flags
PYAUDIO_AVAILABLE = check_module_available("pyaudio")
WHISPER_AVAILABLE = check_module_available("whisper")
OLLAMA_AVAILABLE = check_module_available("ollama")
FAISS_AVAILABLE = check_module_available("faiss")
SENTENCE_TRANSFORMERS_AVAILABLE = check_module_available("sentence_transformers")

@st.cache_resource
def get_recorder():
    from src.audio_recorder import AudioRecorder
    return AudioRecorder()

@st.cache_resource
def get_transcription_service(model_name="base"):
    from src.transcription import TranscriptionService
    return TranscriptionService(model_name)

@st.cache_resource
def get_summarization_service():
    from src.summarizer import SummarizationService
    return SummarizationService()

@st.cache_resource
def get_action_extractor():
    from src.action_extractor import ActionExtractor
    return ActionExtractor()

@st.cache_resource
def get_rag_engine():
    from src.rag_engine import RAGEngine
    engine = RAGEngine()
    if FAISS_AVAILABLE:
        engine.load_index()
    return engine

@st.cache_resource
def get_database():
    from src.database import Database
    return Database()

@st.cache_resource
def get_email_service():
    from src.email_service import EmailService
    return EmailService()

@st.cache_resource
def get_export_service():
    from src.export_utils import ExportService
    return ExportService()

# --- CSS Loading ---
def load_css():
    css_files = ["assets/style.css", "assets/sidebar.css", "assets/components.css"]
    combined_css = ""
    for file in css_files:
        if os.path.exists(file):
            with open(file) as f:
                combined_css += f.read() + "\n"
    st.markdown(f'<style>{combined_css}</style>', unsafe_allow_html=True)

# --- Session State ---
def init_session_state():
    defaults = {
        "recording": False,
        "recording_start_time": None,
        "audio_file": None,
        "processing": False,
        "current_transcript": None,
        "current_summary": None,
        "current_actions": None,
        "detected_language": "en"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# --- Main App ---
def main():
    init_session_state()
    load_css()
    
    # 1. Sidebar & Navigation
    status_flags = {
        'PYAUDIO_AVAILABLE': PYAUDIO_AVAILABLE,
        'WHISPER_AVAILABLE': WHISPER_AVAILABLE,
        'OLLAMA_AVAILABLE': OLLAMA_AVAILABLE,
        'RAG_AVAILABLE': FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
    }
    
    page_selection = render_sidebar(status_flags)
    
    # 2. Services Initialization
    # We initialize only what's needed or pass factories?
    # For Controller, we need instances.
    db = get_database()
    
    # Callbacks for Controller
    def on_process_audio(path):
        controller.process_audio(
            path,
            get_transcription_service(),
            get_summarization_service(),
            get_action_extractor()
        )
        
    def on_save_meeting(audio_path=None): # audio_path arg to match signature if needed, or use session state
        return controller.save_current_meeting(db, audio_path or st.session_state.audio_file)

    # 3. Routing
    if "Record" in page_selection:
        render_record_page(
            get_recorder() if PYAUDIO_AVAILABLE else None, 
            on_process_audio,
            on_save_meeting
        )
        
        # Also show results below if available?
        # Notion style usually separates, but user might want to see results immediately.
        # Dashboard handles results display.
        if st.session_state.current_transcript:
            st.markdown("---")
            render_main_content(
                get_transcription_service(),
                get_email_service(),
                get_export_service()
            )

    elif "History" in page_selection:
        render_history_page(db)

    elif "RAG" in page_selection:
        render_rag_page(
            get_rag_engine() if FAISS_AVAILABLE else None,
            db,
            get_summarization_service(),
            OLLAMA_AVAILABLE
        )
        
    elif "Settings" in page_selection:
        render_settings_page(get_email_service())
        
if __name__ == "__main__":
    main()
