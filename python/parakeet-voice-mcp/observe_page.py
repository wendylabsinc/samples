"""Minimal, action-free microphone and transcript diagnostic page."""

OBSERVE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Woof Mic Read Test</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1016; color: #f2f4f8;
         font: 16px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  main { width: min(680px, 100%); margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 5px; font-size: 24px; }
  .sub { margin: 0 0 22px; color: #9ba5b7; }
  .panel { padding: 18px; border: 1px solid #293143; border-radius: 14px; background: #161b25; }
  .row { display: flex; align-items: center; gap: 9px; margin-bottom: 15px; }
  #dot { width: 10px; height: 10px; border-radius: 50%; background: #d04b55; }
  #dot.up { background: #38c875; }
  .meter { height: 18px; overflow: hidden; border-radius: 9px; background: #252c3a; }
  #bar { width: 0; height: 100%; background: #48a2ff; transition: width .12s linear; }
  #level { margin-top: 6px; color: #9ba5b7; font-size: 13px; }
  h2 { margin: 24px 0 10px; font-size: 15px; color: #aeb7c8; }
  #latest { min-height: 88px; padding: 17px; border-radius: 12px; background: #10141c;
            border: 1px solid #293143; font-size: 24px; }
  #latest.empty { color: #6f798c; font-size: 16px; }
  #history { margin: 12px 0 0; padding: 0; list-style: none; }
  #history li { padding: 10px 12px; margin-top: 7px; border-radius: 9px; background: #161b25; }
  small { color: #7f899b; }
</style>
</head>
<body><main>
  <h1>Woof Mic Read Test</h1>
  <p class="sub">No wake word. No commands. No robot actions.</p>
  <div class="panel">
    <div class="row"><span id="dot"></span><span id="status">connecting...</span></div>
    <div class="meter"><div id="bar"></div></div>
    <div id="level">waiting for microphone level...</div>
  </div>
  <h2>Latest transcript</h2>
  <div id="latest" class="empty">Speak normally, then pause for one second.</div>
  <ul id="history"></ul>
</main>
<script>
  const dot = document.getElementById('dot');
  const status = document.getElementById('status');
  const bar = document.getElementById('bar');
  const level = document.getElementById('level');
  const latest = document.getElementById('latest');
  const history = document.getElementById('history');
  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onopen = () => { dot.className = 'up'; status.textContent = 'connected — listening'; };
    ws.onclose = () => { dot.className = ''; status.textContent = 'disconnected'; setTimeout(connect, 1200); };
    ws.onerror = () => ws.close();
    ws.onmessage = event => {
      let message; try { message = JSON.parse(event.data); } catch (_) { return; }
      if (message.kind === 'level') {
        const db = Number(message.input_dbfs);
        bar.style.width = Math.max(0, Math.min(100, (db + 60) / 60 * 100)) + '%';
        level.textContent = db.toFixed(1) + ' dBFS';
      } else if (message.kind === 'transcript') {
        latest.className = '';
        latest.textContent = message.text || '(empty transcript)';
        const item = document.createElement('li');
        item.textContent = message.text;
        const meta = document.createElement('small');
        meta.textContent = '  ·  ' + (message.audio_ms / 1000).toFixed(1) + 's audio';
        item.appendChild(meta);
        history.prepend(item);
        while (history.children.length > 12) history.lastChild.remove();
      }
    };
  }
  connect();
</script>
</body></html>"""
