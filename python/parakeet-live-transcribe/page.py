"""The single-page live display. No build step, no external assets."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Transcription</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0f1115; color: #e7e9ee;
         font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px;
           padding: 16px 20px; background: #161922; border-bottom: 1px solid #262b38; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  #status { margin-left: auto; font-size: 13px; display: flex; align-items: center; gap: 7px; }
  #dot { width: 9px; height: 9px; border-radius: 50%; background: #e0b341; }
  #dot.up { background: #35c26a; } #dot.down { background: #e5484d; }
  main { max-width: 820px; margin: 0 auto; padding: 20px; }
  .hint { color: #8b94a7; font-size: 14px; margin: 8px 2px 18px; }
  .card { background: #161922; border: 1px solid #262b38; border-left: 3px solid #4c8dff;
          border-radius: 12px; padding: 14px 16px; margin: 0 0 12px; animation: pop .18s ease; }
  @keyframes pop { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .text { font-size: 19px; }
  .meta { margin-top: 8px; font-size: 12px; color: #8b94a7; display: flex; gap: 14px; flex-wrap: wrap; }
  .meta b { color: #b9c1d1; font-weight: 600; }
  #empty { color: #6b7385; text-align: center; padding: 60px 0; }
</style>
</head>
<body>
<header>
  <h1>Live Transcription</h1>
  <div id="status"><span id="dot"></span><span id="statusText">connecting...</span></div>
</header>
<main>
  <div class="hint">Speech is transcribed on the device by NVIDIA Parakeet. Newest first.</div>
  <div id="feed"><div id="empty">Waiting for speech...</div></div>
</main>
<script>
  const feed = document.getElementById('feed');
  const dot = document.getElementById('dot');
  const statusText = document.getElementById('statusText');
  let empty = document.getElementById('empty');

  function setStatus(up) {
    dot.className = up ? 'up' : 'down';
    statusText.textContent = up ? 'connected' : 'disconnected';
  }

  function addCard(msg) {
    if (empty) { empty.remove(); empty = null; }
    const card = document.createElement('div');
    card.className = 'card';
    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = msg.text;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = '<span>' + new Date().toLocaleTimeString() + '</span>' +
      '<span><b>' + (msg.audio_ms / 1000).toFixed(1) + 's</b> audio</span>' +
      '<span><b>' + msg.input_dbfs + '</b> dBFS</span>';
    card.appendChild(text); card.appendChild(meta);
    feed.insertBefore(card, feed.firstChild);
    while (feed.children.length > 100) feed.removeChild(feed.lastChild);
  }

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onopen = () => setStatus(true);
    ws.onclose = () => { setStatus(false); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => { try { addCard(JSON.parse(e.data)); } catch (_) {} };
  }
  connect();
</script>
</body>
</html>
"""
