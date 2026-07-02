"""
Custom audio recorder component — replaces st_audiorec.

Renders a styled light-background recorder using the browser MediaRecorder API.
Returns a base64-encoded webm string when recording stops, or "__reset__" on reset.
Uses declare_component with an HTML file written to a temp directory so no
external package or dark iframe is involved.
"""

import os
import tempfile
import streamlit.components.v1 as components

# ── HTML written once to disk, served by Streamlit's component host ────────────
_COMPONENT_DIR = os.path.join(tempfile.gettempdir(), "qn_audio_recorder_v2")
os.makedirs(_COMPONENT_DIR, exist_ok=True)

_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  background: #F7F6F3;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  height: 44px;
  overflow: hidden;
}

.row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  height: 44px;
}

button {
  font-family: inherit;
  font-size: 12px;
  font-weight: 500;
  color: #37352F;
  background: transparent;
  border: 1px solid #E9E9E7;
  border-radius: 5px;
  padding: 5px 11px;
  cursor: pointer;
  transition: background 0.1s ease, border-color 0.1s ease;
  white-space: nowrap;
  line-height: 1;
}
button:hover:not(:disabled) {
  background: #F1F1EF;
  border-color: #D5D2CA;
}
button:disabled {
  color: #C0BDB8;
  border-color: #F0EDE8;
  cursor: default;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #E57373;
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.2s;
}
.dot.active {
  opacity: 1;
  animation: blink 1.2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

#status {
  font-size: 11px;
  color: #9B9B9B;
  flex: 1;
  white-space: nowrap;
}
</style>
</head>
<body>
<div class="row">
  <button id="btnStart" onclick="startRec()">Start Recording</button>
  <button id="btnStop"  onclick="stopRec()"  disabled>Stop</button>
  <button id="btnReset" onclick="resetRec()">Reset</button>
  <div   class="dot"   id="dot"></div>
  <span  id="status">Ready</span>
</div>

<script>
var mediaRecorder = null;
var chunks = [];
var stream = null;

/* ── Streamlit message protocol ── */
function send(type, extra) {
  var msg = Object.assign({ isStreamlitMessage: true, type: type }, extra || {});
  window.parent.postMessage(msg, '*');
}

function setValue(val) {
  send('streamlit:setComponentValue', { value: val, dataType: 'json' });
}

/* ── UI helpers ── */
function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function setRecording(active) {
  document.getElementById('btnStart').disabled = active;
  document.getElementById('btnStop').disabled  = !active;
  document.getElementById('dot').className = active ? 'dot active' : 'dot';
  setStatus(active ? 'Recording…' : 'Ready');
}

/* ── Recorder logic ── */
function startRec() {
  navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    .then(function(s) {
      stream = s;
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = function(e) {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      mediaRecorder.onstop = onStop;
      mediaRecorder.start(100);
      setRecording(true);
    })
    .catch(function(err) {
      setStatus('Mic error: ' + err.message);
    });
}

function stopRec() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
  document.getElementById('btnStop').disabled = true;
  document.getElementById('dot').className = 'dot';
  setStatus('Processing…');
}

function resetRec() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    if (stream) stream.getTracks().forEach(function(t) { t.stop(); });
  }
  chunks = [];
  setRecording(false);
  setValue('__reset__');
}

function onStop() {
  var mimeType = (chunks[0] && chunks[0].type) ? chunks[0].type : 'audio/webm';
  var blob = new Blob(chunks, { type: mimeType });
  var reader = new FileReader();
  reader.onload = function() {
    var b64 = reader.result.split(',')[1];
    setValue(b64);
    setStatus('Captured ✓');
    document.getElementById('btnStart').disabled = false;
  };
  reader.onerror = function() {
    setStatus('Read error');
  };
  reader.readAsDataURL(blob);
}

/* ── Init ── */
window.addEventListener('load', function() {
  send('streamlit:componentReady', { apiVersion: 1 });
  send('streamlit:setFrameHeight', { height: 44 });
});
</script>
</body>
</html>
"""

# Write the HTML once (overwrite on each import to pick up changes)
with open(os.path.join(_COMPONENT_DIR, "index.html"), "w", encoding="utf-8") as _f:
    _f.write(_INDEX_HTML)

_component_func = components.declare_component("qn_audio_recorder", path=_COMPONENT_DIR)


def audio_recorder(key: str = "qn_audio_recorder"):
    """
    Render the custom audio recorder.

    Returns:
      - None      : nothing recorded yet
      - "__reset__": user clicked Reset (caller should clear audio_file)
      - str       : base64-encoded webm audio bytes
    """
    return _component_func(key=key, default=None)
