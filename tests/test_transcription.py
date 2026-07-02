"""
Unit tests for src/transcription.py

Covers:
- Audio conversion (_to_wav)
- Segment processing and speaker detection
- Transcript formatting
- Service initialisation and device/thread safety
- Integration smoke-test with a real Whisper call (marked slow, skipped in CI)
"""

import os
import sys
import struct
import wave
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, call

# Make src importable when running from repo root or tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transcription import (
    TranscriptionService,
    TranscriptionResult,
    TranscriptSegment,
    get_transcription_service,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_wav(path: str, duration_s: float = 0.5, sample_rate: int = 16000) -> str:
    """Write a minimal silent WAV file to *path* and return the path."""
    n_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _make_segments(specs):
    """Build a list of dicts matching Whisper's segment format."""
    return [
        {"text": text, "start": start, "end": end}
        for text, start, end in specs
    ]


# ─── _to_wav ──────────────────────────────────────────────────────────────────

class TestToWav(unittest.TestCase):

    def test_wav_input_returned_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = TranscriptionService._to_wav(path)
            self.assertEqual(result, path)
        finally:
            os.unlink(path)

    def test_non_wav_converted_when_ffmpeg_present(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Create a real WAV and rename it .webm so ffmpeg can decode it
            src = os.path.join(tmpdir, "clip.wav")
            _make_wav(src)
            webm = os.path.join(tmpdir, "clip.webm")
            shutil.copy(src, webm)

            if not shutil.which("ffmpeg"):
                self.skipTest("ffmpeg not installed")

            result = TranscriptionService._to_wav(webm)
            # Should produce a _converted.wav alongside the original
            self.assertTrue(result.endswith("_converted.wav"))
            self.assertTrue(os.path.exists(result))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_returns_original_when_ffmpeg_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            path = f.name
        try:
            with patch("shutil.which", return_value=None):
                result = TranscriptionService._to_wav(path)
            self.assertEqual(result, path)
        finally:
            os.unlink(path)

    def test_returns_original_when_ffmpeg_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            with patch("subprocess.run", side_effect=OSError("no ffmpeg")):
                with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                    result = TranscriptionService._to_wav(path)
            self.assertEqual(result, path)
        finally:
            os.unlink(path)


# ─── _process_segments ────────────────────────────────────────────────────────

class TestProcessSegments(unittest.TestCase):

    def setUp(self):
        self.svc = TranscriptionService.__new__(TranscriptionService)
        self.svc.model_name = "base"
        self.svc._model = None
        self.svc._device = None

    def test_empty_segments_returns_empty(self):
        result = self.svc._process_segments([], enable_speaker_detection=True)
        self.assertEqual(result, [])

    def test_single_segment_no_speaker_detection(self):
        segs = _make_segments([("Hello world.", 0.0, 1.0)])
        result = self.svc._process_segments(segs, enable_speaker_detection=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Hello world.")
        self.assertEqual(result[0].speaker, "Speaker")

    def test_consecutive_segments_same_speaker_merged(self):
        # Two close segments — no long pause → same speaker → should merge
        segs = _make_segments([
            ("First sentence.", 0.0, 1.0),
            ("Second sentence.", 1.2, 2.0),  # gap = 0.2s < 1.5s threshold
        ])
        result = self.svc._process_segments(segs, enable_speaker_detection=True)
        self.assertEqual(len(result), 1)
        self.assertIn("First sentence.", result[0].text)
        self.assertIn("Second sentence.", result[0].text)

    def test_long_pause_triggers_speaker_change(self):
        segs = _make_segments([
            ("Hello.", 0.0, 1.0),
            ("How are you?", 3.0, 4.5),  # gap = 2.0s > 1.5s threshold
        ])
        result = self.svc._process_segments(segs, enable_speaker_detection=True)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0].speaker, result[1].speaker)

    def test_empty_text_segments_skipped(self):
        segs = _make_segments([("", 0.0, 1.0), ("Real text.", 1.5, 2.5)])
        result = self.svc._process_segments(segs, enable_speaker_detection=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Real text.")

    def test_segment_fields_populated(self):
        segs = _make_segments([("Test.", 1.5, 3.0)])
        result = self.svc._process_segments(segs, enable_speaker_detection=False)
        seg = result[0]
        self.assertEqual(seg.start, 1.5)
        self.assertEqual(seg.end, 3.0)
        self.assertIsInstance(seg.text, str)


# ─── format_transcript_with_speakers ─────────────────────────────────────────

class TestFormatTranscript(unittest.TestCase):

    def _make_result(self, speaker_texts):
        segs = [
            TranscriptSegment(text=t, start=i * 2.0, end=i * 2.0 + 1.5, speaker=s)
            for i, (s, t) in enumerate(speaker_texts)
        ]
        return TranscriptionResult(
            full_text=" ".join(t for _, t in speaker_texts),
            segments=segs,
            language="en",
            language_probability=0.99,
        )

    def test_basic_formatting(self):
        svc = TranscriptionService.__new__(TranscriptionService)
        result = self._make_result([
            ("Speaker 1", "Hello there."),
            ("Speaker 2", "Hi, how are you?"),
        ])
        fmt = svc.format_transcript_with_speakers(result)
        self.assertIn('Speaker 1: "Hello there."', fmt)
        self.assertIn('Speaker 2: "Hi, how are you?"', fmt)

    def test_empty_segments_returns_empty_string(self):
        svc = TranscriptionService.__new__(TranscriptionService)
        result = TranscriptionResult(
            full_text="", segments=[], language="en", language_probability=0.0
        )
        fmt = svc.format_transcript_with_speakers(result)
        self.assertEqual(fmt, "")

    def test_timestamps_included_when_requested(self):
        svc = TranscriptionService.__new__(TranscriptionService)
        result = self._make_result([("Speaker 1", "With timestamps.")])
        fmt = svc.format_transcript_with_speakers(result, include_timestamps=True)
        self.assertIn("[00:00", fmt)


# ─── device / thread safety ───────────────────────────────────────────────────

class TestLoadModel(unittest.TestCase):

    @patch("src.transcription.whisper")
    def test_model_loaded_on_cpu(self, mock_whisper):
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        svc = TranscriptionService("base")
        svc._load_model()

        mock_whisper.load_model.assert_called_once_with("base", device="cpu")

    @patch("src.transcription.whisper")
    def test_set_num_threads_called(self, mock_whisper):
        mock_whisper.load_model.return_value = MagicMock()

        import torch
        with patch.object(torch, "set_num_threads") as mock_threads:
            svc = TranscriptionService("base")
            svc._load_model()
            mock_threads.assert_called_with(1)

    @patch("src.transcription.whisper")
    def test_model_not_reloaded_on_second_call(self, mock_whisper):
        mock_whisper.load_model.return_value = MagicMock()
        svc = TranscriptionService("base")
        svc._load_model()
        svc._load_model()
        self.assertEqual(mock_whisper.load_model.call_count, 1)


# ─── transcribe() — unit (mocked Whisper) ─────────────────────────────────────

class TestTranscribeMocked(unittest.TestCase):

    def _make_svc(self, mock_whisper_module):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello world.",
            "segments": [{"text": "Hello world.", "start": 0.0, "end": 1.5}],
            "language": "en",
            "language_probability": 0.98,
        }
        mock_whisper_module.load_model.return_value = mock_model
        svc = TranscriptionService("base")
        return svc, mock_model

    @patch("src.transcription.whisper")
    def test_transcribe_returns_result_object(self, mock_whisper):
        svc, _ = self._make_svc(mock_whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name)
        try:
            result = svc.transcribe(path)
            self.assertIsInstance(result, TranscriptionResult)
            self.assertEqual(result.language, "en")
            self.assertIn("Hello", result.full_text)
        finally:
            os.unlink(path)

    @patch("src.transcription.whisper")
    def test_transcribe_fp16_false(self, mock_whisper):
        svc, mock_model = self._make_svc(mock_whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name)
        try:
            svc.transcribe(path)
            call_kwargs = mock_model.transcribe.call_args[1]
            self.assertFalse(call_kwargs.get("fp16", True))
        finally:
            os.unlink(path)

    @patch("src.transcription.whisper")
    def test_word_timestamps_never_set(self, mock_whisper):
        """word_timestamps must never be passed to avoid DTW segfault on macOS/Apple Silicon.
        The DTW post-processing runs in native C code after the Python loop and cannot
        be caught by try/except — it must simply not be invoked."""
        svc, mock_model = self._make_svc(mock_whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name)
        try:
            svc.transcribe(path)
            call_kwargs = mock_model.transcribe.call_args[1]
            self.assertNotIn("word_timestamps", call_kwargs,
                             "word_timestamps must not be passed — DTW segfaults on Apple Silicon")
        finally:
            os.unlink(path)

    @patch("src.transcription.whisper")
    def test_transcribe_called_exactly_once(self, mock_whisper):
        """With word_timestamps removed there is no retry path — model.transcribe called once."""
        svc, mock_model = self._make_svc(mock_whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name)
        try:
            svc.transcribe(path)
            self.assertEqual(mock_model.transcribe.call_count, 1)
        finally:
            os.unlink(path)

    @patch("src.transcription.whisper")
    def test_missing_file_raises(self, _mock_whisper):
        svc = TranscriptionService("base")
        with self.assertRaises(FileNotFoundError):
            svc.transcribe("/nonexistent/path/audio.wav")

    @patch("src.transcription.whisper")
    def test_progress_callback_called(self, mock_whisper):
        svc, _ = self._make_svc(mock_whisper)
        calls = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name)
        try:
            svc.transcribe(path, progress_callback=lambda p, m: calls.append(p))
            self.assertTrue(len(calls) >= 2)
            self.assertAlmostEqual(calls[-1], 1.0)
        finally:
            os.unlink(path)


# ─── get_transcription_service singleton ──────────────────────────────────────

class TestGetTranscriptionService(unittest.TestCase):

    def test_returns_transcription_service(self):
        svc = get_transcription_service("base")
        self.assertIsInstance(svc, TranscriptionService)

    def test_singleton_same_model(self):
        a = get_transcription_service("base")
        b = get_transcription_service("base")
        self.assertIs(a, b)

    def test_new_instance_on_model_change(self):
        a = get_transcription_service("base")
        b = get_transcription_service("tiny")
        self.assertIsNot(a, b)
        self.assertEqual(b.model_name, "tiny")


# ─── integration smoke-test (skipped unless QUICKNOTES_SLOW_TESTS=1) ──────────

@unittest.skipUnless(
    os.environ.get("QUICKNOTES_SLOW_TESTS") == "1",
    "Set QUICKNOTES_SLOW_TESTS=1 to run integration tests (loads Whisper model)",
)
class TestTranscribeIntegration(unittest.TestCase):

    def test_transcribe_silent_wav(self):
        """Whisper should complete without segfaulting on a short silent WAV."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = _make_wav(f.name, duration_s=1.0)
        try:
            svc = TranscriptionService("base")
            result = svc.transcribe(path)
            self.assertIsInstance(result, TranscriptionResult)
            self.assertIsInstance(result.full_text, str)
        finally:
            os.unlink(path)

    def test_webm_converted_before_transcription(self):
        """A WAV renamed to .webm should be converted and transcribed cleanly."""
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            path = f.name
        try:
            _make_wav(path.replace(".webm", ".wav"), duration_s=0.5)
            shutil.copy(path.replace(".webm", ".wav"), path)
            if not shutil.which("ffmpeg"):
                self.skipTest("ffmpeg not installed")
            svc = TranscriptionService("base")
            result = svc.transcribe(path)
            self.assertIsInstance(result, TranscriptionResult)
        finally:
            for p in [path, path.replace(".webm", ".wav"), path.replace(".webm", "_converted.wav")]:
                if os.path.exists(p):
                    os.unlink(p)


if __name__ == "__main__":
    unittest.main()
