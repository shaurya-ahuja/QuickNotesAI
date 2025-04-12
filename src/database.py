"""
QuickNotes-AI Database Layer
SQLite storage for meetings, tags, and action items.
100% Local - No Data Leaves Your Device
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


class Database:
    """SQLite database manager for QuickNotes-AI."""
    
    def __init__(self, db_path: str = "data/meetings.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Meetings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                transcript TEXT,
                summary TEXT,
                speaker_quotes TEXT,
                audio_path TEXT,
                language TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        # Meeting-Tags junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meeting_tags (
                meeting_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (meeting_id, tag_id),
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        """)
        
        # Action items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER,
                task TEXT NOT NULL,
                assignee TEXT,
                deadline TEXT,
                emoji TEXT DEFAULT '📋',
                completed INTEGER DEFAULT 0,
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            )
        """)
        
        # RAG documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT,
                embedding_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== Meeting Operations ====================
    
    def create_meeting(
        self,
        title: str,
        transcript: str = "",
        summary: str = "",
        speaker_quotes: str = "",
        audio_path: str = "",
        language: str = "en",
        tags: List[str] = None
    ) -> int:
        """Create a new meeting record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO meetings (title, date, transcript, summary, speaker_quotes, audio_path, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, datetime.now().isoformat(), transcript, summary, speaker_quotes, audio_path, language))
        
        meeting_id = cursor.lastrowid
        
        # Add tags if provided
        if tags:
            for tag_name in tags:
                tag_id = self._get_or_create_tag(cursor, tag_name)
                cursor.execute(
                    "INSERT OR IGNORE INTO meeting_tags (meeting_id, tag_id) VALUES (?, ?)",
                    (meeting_id, tag_id)
                )
        
        conn.commit()
        conn.close()
        return meeting_id
    
    def update_meeting(self, meeting_id: int, **kwargs) -> bool:
        """Update meeting fields."""
        allowed_fields = ['title', 'transcript', 'summary', 'speaker_quotes', 'audio_path', 'language']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [meeting_id]
        
        cursor.execute(f"UPDATE meetings SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    
    def get_meeting(self, meeting_id: int) -> Optional[Dict[str, Any]]:
        """Get a single meeting by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,))
        row = cursor.fetchone()
        
        if row:
            meeting = dict(row)
            meeting['tags'] = self._get_meeting_tags(cursor, meeting_id)
            meeting['action_items'] = self._get_meeting_actions(cursor, meeting_id)
        else:
            meeting = None
        
        conn.close()
        return meeting
    
    def get_all_meetings(self, tag_filter: str = None) -> List[Dict[str, Any]]:
        """Get all meetings, optionally filtered by tag."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if tag_filter:
            cursor.execute("""
                SELECT DISTINCT m.* FROM meetings m
                JOIN meeting_tags mt ON m.id = mt.meeting_id
                JOIN tags t ON mt.tag_id = t.id
                WHERE t.name = ?
                ORDER BY m.date DESC
            """, (tag_filter,))
        else:
            cursor.execute("SELECT * FROM meetings ORDER BY date DESC")
        
        meetings = []
        for row in cursor.fetchall():
            meeting = dict(row)
            meeting['tags'] = self._get_meeting_tags(cursor, meeting['id'])
            meetings.append(meeting)
        
        conn.close()
        return meetings
    
    def delete_meeting(self, meeting_id: int) -> bool:
        """Delete a meeting and its related data."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    def search_meetings(self, query: str) -> List[Dict[str, Any]]:
        """Search meetings by title, transcript, or summary."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT * FROM meetings 
            WHERE title LIKE ? OR transcript LIKE ? OR summary LIKE ?
            ORDER BY date DESC
        """, (search_term, search_term, search_term))
        
        meetings = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return meetings
    
    # ==================== Tag Operations ====================
    
    def _get_or_create_tag(self, cursor, tag_name: str) -> int:
        """Get or create a tag, return its ID."""
        cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
        row = cursor.fetchone()
        if row:
            return row['id']
        cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        return cursor.lastrowid
    
    def _get_meeting_tags(self, cursor, meeting_id: int) -> List[str]:
        """Get all tags for a meeting."""
        cursor.execute("""
            SELECT t.name FROM tags t
            JOIN meeting_tags mt ON t.id = mt.tag_id
            WHERE mt.meeting_id = ?
        """, (meeting_id,))
        return [row['name'] for row in cursor.fetchall()]
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM tags ORDER BY name")
        tags = [row['name'] for row in cursor.fetchall()]
        conn.close()
        return tags
    
    def add_tag_to_meeting(self, meeting_id: int, tag_name: str) -> bool:
        """Add a tag to a meeting."""
        conn = self._get_connection()
        cursor = conn.cursor()
        tag_id = self._get_or_create_tag(cursor, tag_name)
        try:
            cursor.execute(
                "INSERT INTO meeting_tags (meeting_id, tag_id) VALUES (?, ?)",
                (meeting_id, tag_id)
            )
            conn.commit()
            success = True
        except sqlite3.IntegrityError:
            success = False
        conn.close()
        return success
    
    # ==================== Action Item Operations ====================
    
    def add_action_item(
        self,
        meeting_id: int,
        task: str,
        assignee: str = None,
        deadline: str = None,
        emoji: str = "📋"
    ) -> int:
        """Add an action item to a meeting."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO action_items (meeting_id, task, assignee, deadline, emoji)
            VALUES (?, ?, ?, ?, ?)
        """, (meeting_id, task, assignee, deadline, emoji))
        
        action_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return action_id
    
    def _get_meeting_actions(self, cursor, meeting_id: int) -> List[Dict[str, Any]]:
        """Get all action items for a meeting."""
        cursor.execute(
            "SELECT * FROM action_items WHERE meeting_id = ? ORDER BY id",
            (meeting_id,)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_action_items(self, meeting_id: int) -> List[Dict[str, Any]]:
        """Get action items for a meeting."""
        conn = self._get_connection()
        cursor = conn.cursor()
        actions = self._get_meeting_actions(cursor, meeting_id)
        conn.close()
        return actions
    
    def toggle_action_item(self, action_id: int) -> bool:
        """Toggle action item completion status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE action_items SET completed = NOT completed WHERE id = ?",
            (action_id,)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    def delete_action_item(self, action_id: int) -> bool:
        """Delete an action item."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM action_items WHERE id = ?", (action_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    # ==================== Document Operations (RAG) ====================
    
    def add_document(self, filename: str, content: str, embedding_id: str = None) -> int:
        """Add a document for RAG."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (filename, content, embedding_id) VALUES (?, ?, ?)",
            (filename, content, embedding_id)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        docs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return docs
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
