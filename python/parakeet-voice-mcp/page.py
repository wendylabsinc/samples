"""The single-page live display. No build step, no external assets."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Commands</title>
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
  #wake { display: none; align-items: center; gap: 10px; margin: 0 0 16px; padding: 12px 16px;
          border-radius: 12px; background: #17243a; border: 1px solid #274b7a; color: #cfe0ff; }
  #wake.show { display: flex; }
  #wake .pulse { width: 10px; height: 10px; border-radius: 50%; background: #4c8dff;
                 animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
  .actions { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
  .action { font-size: 14px; padding: 8px 10px; border-radius: 8px; background: #10231a;
            border: 1px solid #1f4a35; color: #b7f0cf; }
  .action code { color: #eafff2; }
  .action.err { background: #2a1618; border-color: #5a2a2e; color: #f3b7bc; }
  .pending { color: #8b94a7; font-style: italic; }
</style>
</head>
<body>
<header>
  <h1>Voice Commands</h1>
  <div id="status"><span id="dot"></span><span id="statusText">connecting...</span></div>
</header>
<main>
  <div class="hint">Say <b>&ldquo;{{WAKE}}&rdquo;</b>, then a command. Everything runs on the device.</div>
  <div id="wake"><span class="pulse"></span><span>Heard &ldquo;{{WAKE}}&rdquo; &mdash; listening...</span></div>
  <div id="feed"><div id="empty">Waiting for &ldquo;{{WAKE}}&rdquo;...</div></div>
</main>
<script>
  const feed = document.getElementById('feed');
  const dot = document.getElementById('dot');
  const statusText = document.getElementById('statusText');
  const wake = document.getElementById('wake');
  let empty = document.getElementById('empty');
  const cards = {};
  let wakeTimer = null;

  function showWake() {
    wake.classList.add('show');
    if (wakeTimer) clearTimeout(wakeTimer);
    wakeTimer = setTimeout(() => wake.classList.remove('show'), 8000);
  }

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
    const actions = document.createElement('div');
    actions.className = 'actions';
    actions.innerHTML = '<div class="pending">running on device...</div>';
    card.appendChild(text); card.appendChild(meta); card.appendChild(actions);
    if (msg.id) cards[msg.id] = actions;
    wake.classList.remove('show');
    feed.insertBefore(card, feed.firstChild);
    while (feed.children.length > 100) feed.removeChild(feed.lastChild);
  }

  function addAction(msg) {
    const target = msg.id && cards[msg.id];
    if (!target) return;
    target.innerHTML = '';
    if (msg.error || !msg.calls || !msg.calls.length) {
      const d = document.createElement('div');
      d.className = 'action err';
      d.textContent = msg.error ? ('error: ' + msg.error) : 'no matching tool';
      target.appendChild(d);
      return;
    }
    for (const c of msg.calls) {
      const d = document.createElement('div');
      d.className = 'action';
      const args = Object.entries(c.args || {}).map(([k, v]) => k + '=' + JSON.stringify(v)).join(', ');
      d.innerHTML = '<code>' + c.tool + '(' + args + ')</code> &rarr; ' + (c.result || 'done');
      target.appendChild(d);
    }
  }

  function connect() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onopen = () => setStatus(true);
    ws.onclose = () => { setStatus(false); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      let msg; try { msg = JSON.parse(e.data); } catch (_) { return; }
      if (msg.kind === 'armed') showWake();
      else if (msg.kind === 'command') addCard(msg);
      else if (msg.kind === 'action') addAction(msg);
    };
  }
  connect();
</script>
</body>
</html>
"""
