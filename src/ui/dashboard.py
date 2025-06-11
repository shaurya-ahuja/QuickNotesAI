import streamlit as st
from .components import render_summary_card, render_quote_block, render_action_item, render_header

def render_main_content(transcription_service, email_service, export_service):
    """
    Render the active meeting dashboard with Tabs.
    """
    # Header
    render_header(
        title="Active Meeting",
        metadata=f"Detected Language: {st.session_state.get('detected_language', 'Auto')}"
    )

    if not st.session_state.current_transcript:
        st.info("Start recording or upload audio to see results.")
        return

    # Tabs (Notion Style)
    tab_summary, tab_transcript, tab_notes = st.tabs(["✨ AI Summary", "📝 Transcript", "✅ Notes"])
    
    # --- TAB 1: SUMMARY ---
    with tab_summary:
        if st.session_state.current_summary:
            summary = st.session_state.current_summary
            
            # Key Points Card
            render_summary_card(summary, title="Key Points")
            
            # Key Quotes
            if summary.key_quotes:
                st.markdown("### 💬 Key Quotes")
                for quote in summary.key_quotes:
                    render_quote_block(quote.get('speaker', 'Unknown'), quote.get('quote', ''))
        else:
            if st.session_state.processing:
                st.info("Generating summary...")
            else:
                st.info("No summary available yet.")

    # --- TAB 2: TRANSCRIPT ---
    with tab_transcript:
        result = st.session_state.current_transcript
        formatted = transcription_service.format_transcript_with_speakers(result)
        
        st.text_area(
            "Full Transcript",
            value=formatted,
            height=500,
            label_visibility="collapsed",
            key="active_transcript_view"
        )

    # --- TAB 3: NOTES / ACTIONS ---
    with tab_notes:
        st.markdown("### ✅ Action Items")
        if st.session_state.current_actions:
            for i, action in enumerate(st.session_state.current_actions):
                render_action_item(action, i, key_prefix="active_action")
            
            st.markdown("---")
            
            # Export Controls
            col1, col2 = st.columns(2)
            with col1:
                # Placeholder for email
                st.button("📧 Email To Team", use_container_width=True, disabled=True, help="Configure SMTP in settings")
            with col2:
                try:
                    ics_bytes = export_service.get_ics_bytes(
                        [a.to_dict() for a in st.session_state.current_actions],
                        "Meeting Actions"
                    )
                    if ics_bytes:
                        st.download_button(
                            "📅 Export ICS",
                            data=ics_bytes,
                            file_name="meeting_actions.ics",
                            mime="text/calendar",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Export error: {e}")
        else:
            st.caption("No action items detected.")

def render_history_page(db):
    """
    Render the history page with the same clean UI.
    """
    render_header("Meeting History", metadata="Search and Manage your past recordings")
    
    # Search/Filter Bar
    col_search, col_filter = st.columns([0.7, 0.3])
    with col_search:
        search_query = st.text_input("Search meetings...", placeholder="Type query...", label_visibility="collapsed")
    with col_filter:
        tag_filter = st.selectbox("Tags", ["All"] + db.get_all_tags(), label_visibility="collapsed")
    
    st.markdown("---")

    # Fetch Data
    if search_query:
        meetings = db.search_meetings(search_query)
    elif tag_filter != "All":
        meetings = db.get_all_meetings(tag_filter=tag_filter)
    else:
        meetings = db.get_all_meetings()

    if not meetings:
        st.info("No meetings found.")
        return

    # Render List
    for meeting in meetings:
        with st.expander(f"📅 {meeting['title']} - {meeting['date'][:10]}"):
            # Inner Tabs
            h_tab_sum, h_tab_trans, h_tab_act = st.tabs(["Summary", "Transcript", "Actions"])
            
            with h_tab_sum:
                if meeting['summary']:
                    st.markdown(f'<div class="summary-card-container">', unsafe_allow_html=True)
                    bullets = [b.strip() for b in meeting['summary'].split('\n') if b.strip()]
                    for b in bullets:
                        st.markdown(f"• {b}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.caption("No summary.")
            
            with h_tab_trans:
                st.text_area("Transcript", value=meeting['transcript'] or "", height=300, disabled=True, key=f"hist_trans_{meeting['id']}")
            
            with h_tab_act:
                actions = db.get_action_items(meeting['id'])
                if actions:
                    for i, action in enumerate(actions):
                        # Simple rendering for history (read-only mostly)
                        icon = "✅" if action['completed'] else "⬜"
                        st.markdown(f"{icon} {action['emoji']} **{action['task']}**")
                else:
                    st.caption("No actions.")
            
            # Delete Action
            if st.button("🗑️ Delete", key=f"del_{meeting['id']}"):
                db.delete_meeting(meeting['id'])
                st.rerun()

# ==========================================
# RAG & SETTINGS
# ==========================================

def render_rag_page(rag, db, summarizer, olama_available):
    """
    Render the RAG search page.
    """
    render_header("Search Past Meetings", metadata="Text & Semantic Search")
    
    # Mode Switcher
    rag_capable = rag is not None
    
    # Radio needs a unique key or label visibility trick if we want box style, 
    # but here standard radio is fine or segmented control if available.
    mode = st.radio("Search Mode", ["Text Search", "Semantic Search (AI)"], horizontal=True)
    
    st.markdown("---")
    
    if mode == "Semantic Search (AI)":
        if not rag_capable:
            st.warning("⚠️ Semantic search unavailable (Missing dependencies).")
            return
            
        render_semantic_search_content(rag, db, summarizer, olama_available)
    else:
        render_text_search_content(db)

def render_text_search_content(db):
    """Simple text search UI."""
    query = st.text_input("Search Query", placeholder="Enter keywords...", key="txt_search")
    
    if query:
        results = db.search_meetings(query)
        if results:
            st.success(f"Found {len(results)} matches.")
            for meeting in results:
                with st.expander(f"📅 {meeting['title']}"):
                    st.markdown(f"**Summary snippet**: ...{meeting.get('summary', '')[:200]}...")
                    # Actions
                    actions = db.get_action_items(meeting['id'])
                    if actions:
                        st.markdown("**Actions:**")
                        for a in actions:
                            st.caption(f"- {a['task']}")
        else:
            st.info("No matches found.")

def render_semantic_search_content(rag, db, summarizer, olama_available):
    """Semantic search UI."""
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        query = st.text_input("Ask a question", placeholder="What did we decide about the budget?")
    with col2:
        if st.button("🔄 Index New Meetings", use_container_width=True):
            # Indexing logic (simplified for UI component - usually this should adhere to Controller pattern)
            # We'll just run it here for simplicity
            with st.spinner("Indexing..."):
                meetings = db.get_all_meetings()
                for m in meetings:
                    if m.get('transcript'):
                        rag.add_text(f"{m['title']}\n{m['transcript']}", f"meeting_{m['id']}")
            st.success("Indexed!")

    if query:
        context = rag.get_context(query)
        if context:
            if olama_available and summarizer:
                with st.spinner("AI Generating answer..."):
                    ans = summarizer.answer_question(query, context)
                st.markdown("### 💡 Answer")
                st.markdown(ans)
                with st.expander("Sources"):
                    st.text(context)
            else:
                st.markdown("### Context Found")
                st.text(context)
                st.warning("Ollama not available for answer generation.")
        else:
            st.info("No relevant context found.")

def render_settings_page(email_service):
    """Render settings page."""
    render_header("Settings", metadata="Configure Email & Preferences")
    
    st.subheader("📧 Email Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", ["Gmail", "Outlook", "Custom SMTP"])
        email = st.text_input("Email")
        password = st.text_input("App Password", type="password")
        
    with col2:
        if provider == "Custom SMTP":
            server = st.text_input("SMTP Server")
            port = st.number_input("Port", value=587)
    
    if st.button("Save Configuration"):
         st.info("Settings saving is a placeholder in this UI refactor.")
