"""
QuickNotes-AI Summarization Service
Ollama LLM integration for meeting summarization and action extraction.
100% Local - No Data Leaves Your Device
"""

import json
import re
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class ActionItem:
    """Extracted action item from meeting."""
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None
    emoji: str = "📋"
    priority: str = "medium"


@dataclass
class SummaryResult:
    """Complete summary result from LLM."""
    summary_bullets: List[str]
    action_items: List[ActionItem]
    key_quotes: List[Dict[str, str]]  # {"speaker": "...", "quote": "..."}
    raw_response: str


class SummarizationService:
    """
    Ollama-based summarization service for meeting transcripts.
    Extracts summaries, action items, and key quotes.
    """
    
    DEFAULT_MODEL = "llama2"
    FALLBACK_MODELS = ["llama2", "mistral", "gemma:7b", "phi"]
    
    def __init__(self, model_name: str = None):
        """
        Initialize summarization service.
        
        Args:
            model_name: Ollama model to use. If None, will auto-detect available model.
        """
        self.model_name = model_name
        self._available_models = []
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return OLLAMA_AVAILABLE
    
    def _ensure_model(self):
        """Ensure we have a valid model to use."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not installed. Please install with: pip install ollama")
        
        if self.model_name:
            return
        
        # Try to find an available model
        try:
            response = ollama.list()
            # Handle both old and new Ollama API formats
            models_list = response.get('models', [])
            self._available_models = []
            for m in models_list:
                # New API uses .model or .name attribute, old API uses dict
                if hasattr(m, 'model'):
                    self._available_models.append(m.model)
                elif hasattr(m, 'name'):
                    self._available_models.append(m.name)
                elif isinstance(m, dict):
                    self._available_models.append(m.get('name') or m.get('model', ''))
                else:
                    self._available_models.append(str(m))
            
            # Try preferred models first
            for preferred in self.FALLBACK_MODELS:
                for available in self._available_models:
                    if preferred in available.lower():
                        self.model_name = available
                        return
            
            # Use first available model
            if self._available_models:
                self.model_name = self._available_models[0]
            else:
                raise RuntimeError(
                    "No Ollama models found. Please pull a model with: ollama pull llama2"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        if not OLLAMA_AVAILABLE:
            return []
        
        try:
            response = ollama.list()
            models_list = response.get('models', [])
            result = []
            for m in models_list:
                if hasattr(m, 'model'):
                    result.append(m.model)
                elif hasattr(m, 'name'):
                    result.append(m.name)
                elif isinstance(m, dict):
                    result.append(m.get('name') or m.get('model', ''))
                else:
                    result.append(str(m))
            return result
        except:
            return []
    
    def summarize(
        self,
        transcript: str,
        language: str = "en",
        progress_callback: Optional[callable] = None
    ) -> SummaryResult:
        """
        Summarize a meeting transcript.
        
        Args:
            transcript: The meeting transcript text.
            language: Detected language code for prompts.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            SummaryResult with bullets, actions, and quotes.
        """
        self._ensure_model()
        
        if progress_callback:
            progress_callback(0.1, "Preparing prompts...")
        
        # Build comprehensive prompt
        prompt = self._build_summary_prompt(transcript, language)
        
        if progress_callback:
            progress_callback(0.2, f"Generating summary with {self.model_name}...")
        
        # Call Ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Ollama summarization failed: {e}")
        
        if progress_callback:
            progress_callback(0.8, "Parsing results...")
        
        # Parse the structured response
        result = self._parse_response(raw_response)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return result
    
    def summarize_stream(
        self,
        transcript: str,
        language: str = "en"
    ) -> Generator[str, None, None]:
        """
        Stream summarization response for real-time UI updates.
        
        Yields:
            Chunks of the response as they're generated.
        """
        self._ensure_model()
        
        prompt = self._build_summary_prompt(transcript, language)
        
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"\n\n[Error: {e}]"
    
    def answer_question(
        self,
        question: str,
        context: str,
        language: str = "en"
    ) -> str:
        """
        Answer a question using provided context (for RAG).
        
        Args:
            question: User's question.
            context: Retrieved context from vector search.
            language: Language for response.
            
        Returns:
            Answer string.
        """
        self._ensure_model()
        
        prompt = f"""Based on the following meeting notes and context, please answer the question.
Be concise and specific. If the information isn't available in the context, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _build_summary_prompt(self, transcript: str, language: str) -> str:
        """Build the summarization prompt."""
        
        # Language-aware instruction
        lang_instruction = ""
        if language != "en":
            lang_instruction = f"Please respond in the same language as the transcript ({language}). "
        
        prompt = f"""You are a meeting assistant. {lang_instruction}Analyze the following meeting transcript and provide:

1. **SUMMARY** - Key points as bullet points (5-10 bullets)
2. **ACTION ITEMS** - Tasks extracted from the meeting with:
   - Task description
   - Assignee (if mentioned, otherwise "Unassigned")
   - Deadline (if mentioned, otherwise "TBD")
   - Emoji that fits the task type (📅 for deadlines, 📧 for emails, 📞 for calls, 💻 for tech tasks, 📝 for documents, etc.)
3. **KEY QUOTES** - Important statements attributed to speakers

Format your response EXACTLY like this:

## SUMMARY
• [First key point]
• [Second key point]
• [etc.]

## ACTION ITEMS
- Task: [task description] | Assignee: [name] | Deadline: [date] | 📋
- Task: [task description] | Assignee: [name] | Deadline: [date] | 📧

## KEY QUOTES
- Speaker 1: "[Important quote here]"
- Speaker 2: "[Another important quote]"

---

MEETING TRANSCRIPT:
{transcript}

---

Please provide your analysis:"""
        
        return prompt
    
    def _parse_response(self, response: str) -> SummaryResult:
        """Parse LLM response into structured result."""
        
        summary_bullets = []
        action_items = []
        key_quotes = []
        
        # Split into sections
        sections = response.split("##")
        
        for section in sections:
            section = section.strip()
            
            if section.upper().startswith("SUMMARY"):
                # Extract bullet points
                lines = section.split("\n")[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if line.startswith(("•", "-", "*", "·")):
                        bullet = line.lstrip("•-*· ").strip()
                        if bullet:
                            summary_bullets.append(bullet)
            
            elif section.upper().startswith("ACTION"):
                # Extract action items
                lines = section.split("\n")[1:]
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") or "Task:" in line:
                        action = self._parse_action_item(line)
                        if action:
                            action_items.append(action)
            
            elif section.upper().startswith("KEY QUOTES") or section.upper().startswith("QUOTES"):
                # Extract quotes
                lines = section.split("\n")[1:]
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") and ":" in line:
                        quote = self._parse_quote(line)
                        if quote:
                            key_quotes.append(quote)
        
        # If parsing failed, try to extract what we can
        if not summary_bullets:
            # Look for any bullet points
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith(("•", "-", "*")) and len(line) > 5:
                    bullet = line.lstrip("•-*· ").strip()
                    if bullet and "Task:" not in bullet and "Speaker" not in bullet:
                        summary_bullets.append(bullet)
        
        return SummaryResult(
            summary_bullets=summary_bullets,
            action_items=action_items,
            key_quotes=key_quotes,
            raw_response=response
        )
    
    def _parse_action_item(self, line: str) -> Optional[ActionItem]:
        """Parse a single action item line."""
        line = line.lstrip("-•* ").strip()
        
        if not line or "Task:" not in line:
            return None
        
        # Extract components using regex
        task_match = re.search(r"Task:\s*([^|]+)", line)
        assignee_match = re.search(r"Assignee:\s*([^|]+)", line)
        deadline_match = re.search(r"Deadline:\s*([^|]+)", line)
        emoji_match = re.search(r"([📅📧📞💻📝📋🔔✅❌🎯🔍💡📊🗓️]+)", line)
        
        task = task_match.group(1).strip() if task_match else line
        assignee = assignee_match.group(1).strip() if assignee_match else None
        deadline = deadline_match.group(1).strip() if deadline_match else None
        emoji = emoji_match.group(1) if emoji_match else "📋"
        
        # Clean up "TBD" or "Unassigned" values
        if assignee and assignee.lower() in ["tbd", "unassigned", "none", "n/a"]:
            assignee = None
        if deadline and deadline.lower() in ["tbd", "none", "n/a"]:
            deadline = None
        
        return ActionItem(
            task=task,
            assignee=assignee,
            deadline=deadline,
            emoji=emoji
        )
    
    def _parse_quote(self, line: str) -> Optional[Dict[str, str]]:
        """Parse a speaker quote line."""
        line = line.lstrip("-•* ").strip()
        
        # Format: Speaker X: "quote"
        match = re.match(r'(Speaker\s*\d+|[^:]+):\s*["\']?(.+?)["\']?$', line)
        if match:
            return {
                "speaker": match.group(1).strip(),
                "quote": match.group(2).strip().strip('"\'')
            }
        
        return None


# Singleton instance
_service_instance: Optional[SummarizationService] = None


def get_summarization_service(model_name: str = None) -> SummarizationService:
    """Get or create summarization service instance."""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = SummarizationService(model_name)
    elif model_name and _service_instance.model_name != model_name:
        _service_instance = SummarizationService(model_name)
    
    return _service_instance
