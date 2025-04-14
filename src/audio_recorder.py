"""
QuickNotes-AI Audio Recorder
Live microphone recording using PyAudio.
100% Local - No Data Leaves Your Device
"""

import wave
import os
import threading
import time
from datetime import datetime
from typing import Optional, Callable
import struct
import math

# Try to import pyaudio, provide fallback message if not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class AudioRecorder:
    """
    Live audio recorder using PyAudio.
    Records from the default microphone and saves to WAV format.
    """
    
    # Audio settings
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
    CHANNELS = 1
    RATE = 16000  # 16kHz for Whisper compatibility
    
    def __init__(self, output_dir: str = "uploads"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self._recording = False
        self._frames = []
        self._audio = None
        self._stream = None
        self._thread = None
        self._current_file = None
        self._start_time = None
        self._audio_level = 0.0
        self._level_callback: Optional[Callable[[float], None]] = None
    
    @property
    def is_available(self) -> bool:
        """Check if PyAudio is available."""
        return PYAUDIO_AVAILABLE
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if self._start_time and self._recording:
            return time.time() - self._start_time
        return 0.0
    
    @property
    def audio_level(self) -> float:
        """Get current audio level (0.0 to 1.0)."""
        return self._audio_level
    
    def set_level_callback(self, callback: Callable[[float], None]):
        """Set callback for audio level updates."""
        self._level_callback = callback
    
    def _calculate_rms(self, data: bytes) -> float:
        """Calculate RMS (Root Mean Square) of audio data for level visualization."""
        if len(data) < 2:
            return 0.0
        
        # Convert bytes to shorts
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        
        # Calculate RMS
        sum_squares = sum(s * s for s in shorts)
        rms = math.sqrt(sum_squares / count) if count > 0 else 0
        
        # Normalize to 0-1 range (assuming 16-bit audio)
        normalized = min(rms / 32768.0 * 10, 1.0)  # Scale up for visibility
        return normalized
    
    def _record_thread(self):
        """Recording thread function."""
        while self._recording:
            try:
                data = self._stream.read(self.CHUNK, exception_on_overflow=False)
                self._frames.append(data)
                
                # Calculate and report audio level
                self._audio_level = self._calculate_rms(data)
                if self._level_callback:
                    self._level_callback(self._audio_level)
                    
            except Exception as e:
                print(f"Recording error: {e}")
                break
    
    def start_recording(self) -> bool:
        """Start recording from microphone."""
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio is not installed. Please install it with: pip install pyaudio")
        
        if self._recording:
            return False
        
        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self._frames = []
            self._recording = True
            self._start_time = time.time()
            
            # Start recording thread
            self._thread = threading.Thread(target=self._record_thread, daemon=True)
            self._thread.start()
            
            return True
            
        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> Optional[str]:
        """Stop recording and save to file. Returns the file path."""
        if not self._recording:
            return None
        
        self._recording = False
        
        # Wait for recording thread to finish
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save to WAV file
        try:
            self._save_wav(filepath)
            self._current_file = filepath
        finally:
            self._cleanup()
        
        return filepath
    
    def _save_wav(self, filepath: str):
        """Save recorded frames to WAV file."""
        if not self._frames:
            raise RuntimeError("No audio data to save")
        
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self._audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self._frames))
    
    def _cleanup(self):
        """Clean up audio resources."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except:
                pass
            self._stream = None
        
        if self._audio:
            try:
                self._audio.terminate()
            except:
                pass
            self._audio = None
        
        self._frames = []
        self._audio_level = 0.0
    
    def get_last_recording(self) -> Optional[str]:
        """Get path to the last recorded file."""
        return self._current_file
    
    def list_recordings(self) -> list:
        """List all recordings in the output directory."""
        if not os.path.exists(self.output_dir):
            return []
        
        recordings = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(self.output_dir, filename)
                recordings.append({
                    'filename': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath))
                })
        
        return sorted(recordings, key=lambda x: x['modified'], reverse=True)


class AudioRecorderMock:
    """
    Mock audio recorder for systems without PyAudio.
    Allows testing UI without actual recording capability.
    """
    
    def __init__(self, output_dir: str = "uploads"):
        self.output_dir = output_dir
        self._recording = False
        self._start_time = None
    
    @property
    def is_available(self) -> bool:
        return False
    
    @property
    def is_recording(self) -> bool:
        return self._recording
    
    @property
    def duration(self) -> float:
        if self._start_time and self._recording:
            return time.time() - self._start_time
        return 0.0
    
    @property
    def audio_level(self) -> float:
        return 0.0
    
    def set_level_callback(self, callback):
        pass
    
    def start_recording(self) -> bool:
        self._recording = True
        self._start_time = time.time()
        return True
    
    def stop_recording(self) -> Optional[str]:
        self._recording = False
        return None
    
    def get_last_recording(self) -> Optional[str]:
        return None
    
    def list_recordings(self) -> list:
        return []


def get_recorder(output_dir: str = "uploads") -> AudioRecorder:
    """Factory function to get appropriate recorder instance."""
    if PYAUDIO_AVAILABLE:
        return AudioRecorder(output_dir)
    else:
        return AudioRecorderMock(output_dir)
