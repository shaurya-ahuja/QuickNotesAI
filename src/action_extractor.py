"""
QuickNotes-AI Action Item Extractor
Parse and manage action items from LLM output.
100% Local - No Data Leaves Your Device
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class ActionItem:
    """Structured action item."""
    id: Optional[int] = None
    task: str = ""
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    emoji: str = "📋"
    completed: bool = False
    priority: str = "medium"
    meeting_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_display_string(self) -> str:
        """Format for UI display."""
        parts = [f"{self.emoji} {self.task}"]
        
        if self.assignee:
            parts.append(f"[Assignee: {self.assignee}]")
        
        if self.deadline:
            parts.append(f"[Due: {self.deadline}]")
        
        return " ".join(parts)


class ActionExtractor:
    """
    Extract and parse action items from text or LLM output.
    """
    
    # Emoji mappings for different task types
    TASK_EMOJIS = {
        "email": "📧",
        "call": "📞",
        "meeting": "📅",
        "document": "📝",
        "review": "🔍",
        "code": "💻",
        "design": "🎨",
        "deadline": "⏰",
        "follow up": "🔔",
        "send": "📤",
        "schedule": "🗓️",
        "prepare": "📊",
        "complete": "✅",
        "urgent": "🚨",
        "idea": "💡",
        "question": "❓",
        "decision": "🎯",
        "default": "📋"
    }
    
    # Common deadline patterns
    DEADLINE_PATTERNS = [
        (r'by\s+(\w+day)', 'weekday'),  # by Monday, by Friday
        (r'by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', 'date'),  # by 1/15, by 01-15-2024
        (r'by\s+end\s+of\s+(week|day|month)', 'relative'),  # by end of week
        (r'due\s+(\w+day)', 'weekday'),  # due Monday
        (r'before\s+(\w+)', 'before'),  # before Tuesday
        (r'next\s+(\w+day)', 'next_weekday'),  # next Monday
        (r'tomorrow', 'tomorrow'),
        (r'today', 'today'),
        (r'asap|immediately|urgent', 'urgent'),
    ]
    
    # Action keywords
    ACTION_KEYWORDS = [
        "need to", "should", "must", "will", "have to",
        "action:", "todo:", "task:", "follow up",
        "responsible for", "assigned to", "please",
        "make sure", "don't forget", "remember to"
    ]
    
    def extract_from_text(self, text: str) -> List[ActionItem]:
        """
        Extract action items from raw text using heuristics.
        
        Args:
            text: Raw transcript or meeting notes.
            
        Returns:
            List of ActionItem objects.
        """
        items = []
        lines = text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains action keywords
            line_lower = line.lower()
            is_action = any(keyword in line_lower for keyword in self.ACTION_KEYWORDS)
            
            if is_action:
                item = self._parse_action_line(line)
                if item:
                    items.append(item)
        
        return items
    
    def extract_from_structured(self, llm_actions: List[Any]) -> List[ActionItem]:
        """
        Convert structured LLM output to ActionItems.
        
        Args:
            llm_actions: List of action items from LLM (dataclass or dict).
            
        Returns:
            List of ActionItem objects.
        """
        items = []
        
        for action in llm_actions:
            if hasattr(action, 'task'):
                # From summarizer ActionItem
                item = ActionItem(
                    task=action.task,
                    assignee=action.assignee,
                    deadline=action.deadline,
                    emoji=action.emoji if hasattr(action, 'emoji') else self._get_emoji_for_task(action.task)
                )
            elif isinstance(action, dict):
                item = ActionItem(
                    task=action.get('task', ''),
                    assignee=action.get('assignee'),
                    deadline=action.get('deadline'),
                    emoji=action.get('emoji', self._get_emoji_for_task(action.get('task', '')))
                )
            else:
                continue
            
            if item.task:
                items.append(item)
        
        return items
    
    def _parse_action_line(self, line: str) -> Optional[ActionItem]:
        """Parse a single line into an action item."""
        # Clean the line
        line = line.strip("•-*· ")
        
        if len(line) < 5:
            return None
        
        # Extract assignee
        assignee = self._extract_assignee(line)
        
        # Extract deadline
        deadline = self._extract_deadline(line)
        
        # Clean task text
        task = self._clean_task_text(line)
        
        # Get appropriate emoji
        emoji = self._get_emoji_for_task(task)
        
        return ActionItem(
            task=task,
            assignee=assignee,
            deadline=deadline,
            emoji=emoji
        )
    
    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract assignee from text."""
        patterns = [
            r'assigned?\s+to\s+(\w+)',
            r'\[assignee:\s*(\w+)\]',
            r'@(\w+)',
            r'(\w+)\s+will\b',
            r'(\w+)\s+should\b',
            r'(\w+)\s+needs?\s+to\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if name.lower() not in ['i', 'we', 'you', 'they', 'someone', 'everyone', 'team']:
                    return name.title()
        
        return None
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline from text."""
        text_lower = text.lower()
        
        for pattern, pattern_type in self.DEADLINE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                if pattern_type == 'tomorrow':
                    return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                elif pattern_type == 'today':
                    return datetime.now().strftime("%Y-%m-%d")
                elif pattern_type == 'urgent':
                    return "ASAP"
                elif match.groups():
                    return match.group(1).title()
        
        return None
    
    def _clean_task_text(self, text: str) -> str:
        """Clean task text by removing metadata."""
        # Remove assignee patterns
        text = re.sub(r'\[assignee:\s*\w+\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'assigned?\s+to\s+\w+', '', text, flags=re.IGNORECASE)
        
        # Remove deadline patterns
        text = re.sub(r'\[due:\s*[^\]]+\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[deadline:\s*[^\]]+\]', '', text, flags=re.IGNORECASE)
        
        # Remove action prefixes
        text = re.sub(r'^(action:|todo:|task:)\s*', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _get_emoji_for_task(self, task: str) -> str:
        """Get appropriate emoji for task based on keywords."""
        task_lower = task.lower()
        
        for keyword, emoji in self.TASK_EMOJIS.items():
            if keyword in task_lower:
                return emoji
        
        return self.TASK_EMOJIS["default"]
    
    def format_checklist(
        self,
        items: List[ActionItem],
        show_completed: bool = True
    ) -> str:
        """
        Format action items as a markdown checklist.
        
        Args:
            items: List of ActionItem objects.
            show_completed: Whether to show completed items.
            
        Returns:
            Markdown formatted checklist.
        """
        lines = []
        
        for item in items:
            if not show_completed and item.completed:
                continue
            
            checkbox = "[x]" if item.completed else "[ ]"
            line = f"- {checkbox} {item.to_display_string()}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def group_by_assignee(
        self,
        items: List[ActionItem]
    ) -> Dict[str, List[ActionItem]]:
        """Group action items by assignee."""
        groups: Dict[str, List[ActionItem]] = {"Unassigned": []}
        
        for item in items:
            key = item.assignee or "Unassigned"
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        return groups
    
    def group_by_deadline(
        self,
        items: List[ActionItem]
    ) -> Dict[str, List[ActionItem]]:
        """Group action items by deadline."""
        groups: Dict[str, List[ActionItem]] = {"No Deadline": []}
        
        for item in items:
            key = item.deadline or "No Deadline"
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        return groups


# Singleton instance
_extractor_instance: Optional[ActionExtractor] = None


def get_action_extractor() -> ActionExtractor:
    """Get action extractor instance."""
    global _extractor_instance
    
    if _extractor_instance is None:
        _extractor_instance = ActionExtractor()
    
    return _extractor_instance
