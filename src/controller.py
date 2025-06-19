import streamlit as st
import time
from datetime import datetime

def process_audio(audio_path, transcription_service, summarizer_service, action_extractor):
    """
    Process audio file: Transcribe -> Summarize -> Extract Actions.
    Updates st.session_state directly.
    """
    try:
        st.session_state.processing = True
        
        # UI Feedback via placeholders (passed or just generic st)
        # We'll use a progress container if we can, but st.spinner/toast is easier here.
        
        with st.status("🎧 Processing audio...", expanded=True) as status:
            
            # 1. Transcribe
            status.write("Transcribing audio (Whisper)...")
            result = transcription_service.transcribe(audio_path)
            st.session_state.current_transcript = result
            st.session_state.detected_language = result.language
            
            # 2. Summarize
            status.write("Generating summary (Ollama)...")
            # Convert transcript to text
            full_text = transcription_service.format_transcript(result)
            summary_result = summarizer_service.summarize(full_text, language=result.language)
            st.session_state.current_summary = summary_result
            
            # 3. Actions
            status.write("Extracting action items...")
            actions = action_extractor.extract_from_structured(summary_result.action_items)
            st.session_state.current_actions = actions
            
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
            
        return True
        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        return False
    finally:
        st.session_state.processing = False

def save_current_meeting(db, audio_file):
    """Save current session state to DB."""
    if not st.session_state.current_transcript:
        st.warning("No transcript to save.")
        return

    try:
        # Prepare data
        transcript_text = st.session_state.current_transcript.full_text
        summary_text = "\n".join(st.session_state.current_summary.summary_bullets) if st.session_state.current_summary else ""
        
        # Save
        meeting_id = db.create_meeting(
            title=f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            transcript=transcript_text,
            summary=summary_text,
            language=st.session_state.detected_language,
            audio_path=audio_file or ""
        )
        
        # Save actions
        if st.session_state.current_actions:
            for action in st.session_state.current_actions:
                db.add_action_item(
                    meeting_id=meeting_id,
                    task=action.task,
                    assignee=action.assignee,
                    deadline=action.deadline,
                    emoji=action.emoji,
                    completed=action.completed
                )
        
        st.toast(f"Meeting saved! ID: {meeting_id}", icon="💾")
        return meeting_id
        
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None
