import streamlit as st

def render_sidebar(status_flags):
    """
    Render the sidebar with navigation and settings.
    
    Args:
        status_flags (dict): Dictionary of availability flags (e.g., {'PYAUDIO_AVAILABLE': True})
    
    Returns:
        str: The selected page name
    """
    with st.sidebar:
        st.markdown('<div class="privacy-badge">🔒 100% Local - No Data Leaves Your Device</div>', unsafe_allow_html=True)
        
        st.title("📝 QuickNotes-AI")
        st.caption("Offline Meeting Notetaker")
        
        st.markdown("---")
        
        # Navigation
        # Uses the custom "Box" styling from assets/sidebar.css
        page = st.radio(
            "Navigation",
            ["🎙️ Record Meeting", "📚 Meeting History", "🔍 RAG Search", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System Status
        st.subheader("System Status")
        
        status_items = [
            ("PyAudio (Recording)", status_flags.get('PYAUDIO_AVAILABLE', False)),
            ("Whisper (Transcription)", status_flags.get('WHISPER_AVAILABLE', False)),
            ("Ollama (LLM)", status_flags.get('OLLAMA_AVAILABLE', False)),
            ("FAISS (RAG)", status_flags.get('RAG_AVAILABLE', False)),
        ]
        
        for name, available in status_items:
            icon = "✅" if available else "❌"
            color = "status-available" if available else "status-unavailable"
            st.markdown(f'{icon} <span class="{color}">{name}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("Made with ❤️ for privacy")
        
        return page
