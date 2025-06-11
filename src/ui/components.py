import streamlit as st
from datetime import datetime

def render_summary_card(summary, title="Key Points"):
    """
    Render a clean summary card with bullets.
    """
    st.markdown(f'<div class="summary-card-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-heading">{title}</div>', unsafe_allow_html=True)
    
    if summary and summary.summary_bullets:
        for bullet in summary.summary_bullets:
            st.markdown(f"• {bullet}")
    else:
        st.info("No summary content available.")
        
    st.markdown('</div>', unsafe_allow_html=True)

def render_quote_block(speaker, text):
    """
    Render a styled quote block.
    """
    st.markdown(f"""
        <div class="quote-block">
            "{text}"
            <span class="quote-author">— {speaker}</span>
        </div>
    """, unsafe_allow_html=True)

def render_action_item(action, index, key_prefix="action"):
    """
    Render a single action item row.
    """
    col_check, col_content = st.columns([0.05, 0.95])
    
    with col_check:
        # Checkbox update
        completed = st.checkbox("", value=action.completed, key=f"{key_prefix}_{index}")
        action.completed = completed
        
    with col_content:
        # Using HTML for cleaner control than standard markdown
        assignee_html = f'<span class="action-meta">👤 {action.assignee}</span>' if action.assignee else ""
        deadline_html = f'<span class="action-meta">📅 {action.deadline}</span>' if action.deadline else ""
        
        st.markdown(f"""
            <div class="action-item">
                <div style="flex-grow: 1;">
                    <span class="action-text"><strong>{action.emoji} {action.task}</strong></span>
                    <div>{assignee_html} {deadline_html}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_header(title, subtitle=None, metadata=None):
    """
    Render a clean workspace header.
    """
    # Breadcrumbs
    st.caption("Workspace / QuickNotes / Today")
    
    # Title
    st.markdown(f"# {title}")
    
    # Meta info line
    if metadata:
        st.caption(metadata)
    
    st.markdown("---")
