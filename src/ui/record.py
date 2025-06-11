import streamlit as st
import time
import os
from .components import render_header

def render_record_page(recorder, process_callback, save_callback):
    """
    Render the recording page.
    """
    render_header("Record Meeting", metadata="Live Audio Capture & Processing")
    
    col_live, col_upload = st.columns([1, 1], gap="large")
    
    # --- Live Recording Section ---
    with col_live:
        st.subheader("🎙️ Live Recording")
        st.caption("Capture audio directly from your microphone.")
        
        # State: Resting
        if not st.session_state.recording:
            if st.button("Start Recording", type="primary", use_container_width=True):
                try:
                    recorder.start_recording()
                    st.session_state.recording = True
                    st.session_state.recording_start_time = time.time()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")
        
        # State: Recording
        else:
            st.markdown("""
                <div style="background-color: rgba(255, 59, 48, 0.1); color: #FF3B30; padding: 16px; border-radius: 8px; text-align: center; font-weight: 600; margin-bottom: 16px; border: 1px solid rgba(255, 59, 48, 0.2);">
                    🔴 Recording in Progress...
                </div>
            """, unsafe_allow_html=True)
            
            # Duration
            if st.session_state.recording_start_time:
                duration = int(time.time() - st.session_state.recording_start_time)
                mins, secs = divmod(duration, 60)
                st.metric("Duration", f"{mins:02d}:{secs:02d}")
            
            if st.button("Stop Recording", use_container_width=True):
                path = recorder.stop_recording()
                st.session_state.recording = False
                st.session_state.audio_file = path
                st.rerun()
            
            time.sleep(1)
            st.rerun()

        # Actions after recording
        if st.session_state.audio_file and not st.session_state.recording:
            st.success("Audio captured successfully.")
            if st.button("⚡ Process Audio", type="primary", use_container_width=True):
                process_callback(st.session_state.audio_file)
                st.rerun()

        if st.session_state.current_transcript:
            if st.button("💾 Save Meeting to History", use_container_width=True):
                save_callback(st.session_state.audio_file)


    # --- Upload Section ---
    with col_upload:
        st.subheader("📂 Upload Audio")
        st.caption("Process pre-recorded meetings (WAV, MP3).")
        
        uploaded_file = st.file_uploader("Drop file here", type=['wav', 'mp3', 'm4a'], label_visibility="collapsed")
        
        if uploaded_file:
            # Save temp
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"File loaded: {uploaded_file.name}")
            if st.button("⚡ Process Upload", type="primary", use_container_width=True, key="proc_upload"):
                process_callback(path)
                st.session_state.audio_file = path
                st.rerun()
