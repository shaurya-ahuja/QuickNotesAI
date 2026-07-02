"""
QuickNotes-AI Transcription Service
Local Whisper-based speech-to-text with speaker segmentation.
100% Local - No Data Leaves Your Device
"""

import os
import subprocess
import shutil
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import re

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with speaker and timing info."""
    text: str
    start: float
    end: float
    speaker: str = "Speaker"


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    full_text: str
    segments: List[TranscriptSegment]
    language: str
    language_probability: float


class TranscriptionService:
    """
    Whisper-based transcription service with simple speaker segmentation.
    Uses pause detection to approximate speaker changes.
    """
    
    AVAILABLE_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize transcription service.
        
        Args:
            model_name: Whisper model to use. Options: tiny, base, small, medium, large
                       Larger models are more accurate but slower.
        """
        self.model_name = model_name
        self._model = None
        self._device = None
    
    @property
    def is_available(self) -> bool:
        """Check if Whisper is available."""
        return WHISPER_AVAILABLE
    
    def _load_model(self):
        """Lazy load the Whisper model, always on CPU to avoid MPS segfaults."""
        if self._model is None:
            if not WHISPER_AVAILABLE:
                raise RuntimeError("Whisper is not installed. Please install with: pip install openai-whisper")

            # Force single-thread mode before loading to prevent MPS dispatch conflicts
            # on Apple Silicon with PyTorch 2.x (segfault at inference start).
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass

            print(f"Loading Whisper {self.model_name} model...")
            self._model = whisper.load_model(self.model_name, device="cpu")
            print(f"Model loaded successfully!")

    @staticmethod
    def _to_wav(audio_path: str) -> str:
        """
        Convert audio to 16 kHz mono WAV using ffmpeg.
        Returns path to the WAV file (may be a temp file).
        Falls back to original path if ffmpeg is unavailable or conversion fails.
        """
        ext = os.path.splitext(audio_path)[1].lower()
        if ext == ".wav":
            return audio_path

        if not shutil.which("ffmpeg"):
            return audio_path

        wav_path = os.path.splitext(audio_path)[0] + "_converted.wav"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le",
                    wav_path,
                ],
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0 and os.path.exists(wav_path):
                return wav_path
        except Exception:
            pass
        return audio_path
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        enable_speaker_detection: bool = True,
        progress_callback: Optional[callable] = None
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Optional language code (e.g., 'en', 'es'). Auto-detects if None.
            enable_speaker_detection: Whether to attempt speaker segmentation.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            TranscriptionResult with full text and segments.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        # Ensure single-thread mode is active at call time (survives model reload).
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        # Convert non-WAV formats (e.g. webm from browser recorder) to 16 kHz mono WAV.
        # Whisper can call ffmpeg itself, but doing it explicitly avoids codec edge-cases
        # that produce malformed numpy arrays and trigger a segfault on Apple Silicon.
        prepared_path = self._to_wav(audio_path)

        if progress_callback:
            progress_callback(0.1, "Loading audio...")

        # fp16=False: required on CPU.  word_timestamps omitted (defaults to False):
        # the DTW post-processing pass it enables runs AFTER the main loop in native
        # C extension code — a SIGSEGV there cannot be caught by Python try/except,
        # so it must never be set to True.  QuickNotes only uses segment-level start/end
        # timestamps (from the "segments" list), not word-level timestamps, so this
        # feature is unnecessary.
        transcribe_options: Dict = {
            "verbose": False,
            "fp16": False,
        }

        if language:
            transcribe_options["language"] = language

        if progress_callback:
            progress_callback(0.3, "Transcribing audio...")

        result = self._model.transcribe(prepared_path, **transcribe_options)
        
        if progress_callback:
            progress_callback(0.7, "Processing segments...")
        
        # Extract segments with speaker detection
        segments = self._process_segments(
            result.get("segments", []),
            enable_speaker_detection
        )
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return TranscriptionResult(
            full_text=result.get("text", "").strip(),
            segments=segments,
            language=result.get("language", "en"),
            language_probability=result.get("language_probability", 0.0)
        )
    
    def _process_segments(
        self,
        whisper_segments: List[Dict],
        enable_speaker_detection: bool
    ) -> List[TranscriptSegment]:
        """
        Process Whisper segments and add speaker labels.
        Uses pause-based heuristics for speaker detection.
        """
        if not whisper_segments:
            return []
        
        segments = []
        current_speaker = 1
        last_end = 0.0
        
        # Threshold for pause-based speaker change (seconds)
        PAUSE_THRESHOLD = 1.5
        
        for seg in whisper_segments:
            text = seg.get("text", "").strip()
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            
            if not text:
                continue
            
            # Simple speaker change detection based on pauses
            if enable_speaker_detection:
                pause_duration = start - last_end
                if pause_duration > PAUSE_THRESHOLD and last_end > 0:
                    # Alternate speakers on significant pauses
                    current_speaker = (current_speaker % 3) + 1
            
            speaker_label = f"Speaker {current_speaker}" if enable_speaker_detection else "Speaker"
            
            segments.append(TranscriptSegment(
                text=text,
                start=start,
                end=end,
                speaker=speaker_label
            ))
            
            last_end = end
        
        # Merge consecutive segments from same speaker
        if enable_speaker_detection:
            segments = self._merge_speaker_segments(segments)
        
        return segments
    
    def _merge_speaker_segments(
        self,
        segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """Merge consecutive segments from the same speaker."""
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for seg in segments[1:]:
            if seg.speaker == current.speaker:
                # Merge with current segment
                current = TranscriptSegment(
                    text=current.text + " " + seg.text,
                    start=current.start,
                    end=seg.end,
                    speaker=current.speaker
                )
            else:
                merged.append(current)
                current = seg
        
        merged.append(current)
        return merged
    
    def format_transcript_with_speakers(
        self,
        result: TranscriptionResult,
        include_timestamps: bool = False
    ) -> str:
        """
        Format transcription result with speaker labels.
        
        Returns formatted string like:
        Speaker 1: "Hello, how are you?"
        Speaker 2: "I'm doing well, thanks!"
        """
        lines = []
        
        for seg in result.segments:
            if include_timestamps:
                timestamp = f"[{self._format_time(seg.start)} - {self._format_time(seg.end)}] "
            else:
                timestamp = ""
            
            lines.append(f'{timestamp}{seg.speaker}: "{seg.text}"')
        
        return "\n\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_speaker_quotes(self, result: TranscriptionResult) -> Dict[str, List[str]]:
        """
        Extract quotes organized by speaker.
        
        Returns dict like:
        {
            "Speaker 1": ["Quote 1", "Quote 2"],
            "Speaker 2": ["Quote 3"]
        }
        """
        quotes = {}
        
        for seg in result.segments:
            if seg.speaker not in quotes:
                quotes[seg.speaker] = []
            quotes[seg.speaker].append(seg.text)
        
        return quotes


# Singleton instance for reuse
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(model_name: str = "base") -> TranscriptionService:
    """Get or create transcription service instance."""
    global _service_instance
    
    if _service_instance is None or _service_instance.model_name != model_name:
        _service_instance = TranscriptionService(model_name)
    
    return _service_instance
