# ruff: noqa: E501
"""The chat surface: talk to a space, see what it remembered and why.

A chat box in front of a vector store is a demo anyone can build in an
afternoon. What is worth showing is the part underneath — so every answer here
renders its sources, marks which ones the model actually cited versus merely
saw, and flags when a claim rests on a memory nobody verified. The interesting
column is the right-hand one.

The API key lives in `localStorage` and is sent as a bearer token straight to
`/v1/...`. That is deliberate: this page is a client of the public API and
nothing else, so what it can do is exactly what any customer's own client can
do. There is no privileged back channel — if something works here, the
documented endpoint works.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>Chat — mapi</title>
<link rel="icon" type="image/png" href="/static/mapi-icon.png">
<link rel="apple-touch-icon" href="/static/mapi-icon.png">
<style>
  :root {
    color-scheme: light;
    --bg:#fbfbfa; --panel:#fff; --line:#e6e4e0; --hair:#f0eeea;
    --text:#1b1b19; --dim:#75726c; --accent:#1b1b19;
    --warn:#b8860b; --ok:#2f9e5e;
    --shadow:0 1px 2px rgba(20,18,14,.05), 0 8px 28px rgba(20,18,14,.06);
  }
  * { box-sizing:border-box }
  html, body { height:100% }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
         -webkit-font-smoothing:antialiased }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace }
  a { color:inherit }

  header { border-bottom:1px solid var(--line); background:var(--panel) }
  header .in { max-width:1180px; margin:0 auto; padding:0 24px; height:84px;
               display:flex; align-items:center; justify-content:space-between; gap:16px }
  header img { height:44px; width:auto; display:block }
  header nav a { margin-left:20px; color:var(--dim); text-decoration:none; font-size:14px }
  header nav a:hover { color:var(--text) }

  .shell { max-width:1180px; margin:0 auto; padding:22px 24px 40px;
           display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:22px;
           align-items:start }

  .bar { display:flex; gap:10px; align-items:center; margin-bottom:16px; flex-wrap:wrap }
  input, select, button, textarea { font:inherit }
  input, select {
    padding:9px 12px; border:1px solid var(--line); border-radius:9px;
    background:var(--panel); color:var(--text) }
  input:focus, select:focus, textarea:focus { outline:2px solid var(--accent); outline-offset:-1px }
  #key { flex:1; min-width:240px; font-family:ui-monospace,Menlo,monospace; font-size:13px }
  .btn { padding:9px 16px; border-radius:9px; border:1px solid var(--line);
         background:var(--panel); font-weight:600; font-size:14px; cursor:pointer;
         box-shadow:var(--shadow) }
  .btn:hover { border-color:#d6d3cd }
  .btn.pri { background:var(--accent); border-color:var(--accent); color:#fff }
  .btn.pri:disabled { opacity:.45; cursor:default }

  #thread { min-height:340px; display:flex; flex-direction:column; gap:16px; margin-bottom:18px }
  .msg { max-width:82%; padding:13px 16px; border-radius:14px; white-space:pre-wrap;
         overflow-wrap:anywhere }
  .msg.you { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:5px }
  .msg.mapi { align-self:flex-start; background:var(--panel); border:1px solid var(--line);
              border-bottom-left-radius:5px; box-shadow:var(--shadow) }
  .msg .cite { color:var(--dim); font-weight:600 }
  .meta { align-self:flex-start; font-size:12px; color:var(--dim); margin-top:-8px }
  .flag { display:inline-block; margin-top:9px; padding:4px 9px; border-radius:7px;
          font-size:12px; background:#fdf8ec; color:var(--warn); border:1px solid #f0e2c0 }

  form { display:flex; gap:10px }
  textarea { flex:1; padding:12px 14px; border:1px solid var(--line); border-radius:11px;
             background:var(--panel); resize:none; min-height:52px; max-height:180px }

  aside { position:sticky; top:22px }
  aside h2 { font-size:12px; letter-spacing:.11em; text-transform:uppercase; color:var(--dim);
             font-weight:600; margin:0 0 12px }
  .src { border:1px solid var(--line); background:var(--panel); border-radius:11px;
         padding:12px 13px; margin-bottom:10px; font-size:13.5px; line-height:1.55 }
  .src.used { border-color:#cfcbc3; box-shadow:var(--shadow) }
  .src.unused { opacity:.55 }
  .src .top { display:flex; align-items:center; gap:8px; margin-bottom:6px }
  .n { flex:none; width:21px; height:21px; border-radius:6px; background:var(--hair);
       color:var(--dim); font-size:11px; font-weight:700; display:grid; place-items:center }
  .src.used .n { background:var(--accent); color:#fff }
  .when { color:var(--dim); font-size:12px }
  .tag { display:inline-block; margin:7px 5px 0 0; padding:1px 8px; border-radius:20px;
         font-size:11px; border:1px solid var(--line); color:var(--dim) }
  .unv { color:var(--warn); font-size:11px; font-weight:700; letter-spacing:.04em }
  .empty { color:var(--dim); font-size:13.5px }
  .err { border-left:3px solid #c2352a; padding:10px 14px; background:#fdf3f2;
         border-radius:0 8px 8px 0; margin-bottom:14px; font-size:14px }
  .dots::after { content:'…'; animation:d 1.2s steps(4,end) infinite; }
  @keyframes d { 0%{content:''} 25%{content:'·'} 50%{content:'··'} 75%{content:'···'} }

  @media (max-width:900px) {
    .shell { grid-template-columns:minmax(0,1fr) }
    aside { position:static }
    header .in { height:68px } header img { height:34px }
  }
</style>

<header><div class="in">
  <a href="/"><img src="/static/mapi-wordmark.png" alt="mapi" width="720" height="255"></a>
  <nav><a href="/docs">Docs</a><a href="/reference">Reference</a><a href="/orgs">Dashboard</a></nav>
</div></header>

<div class="shell">
  <main>
    <div class="bar">
      <input id="key" type="password" placeholder="API key — sm_..." autocomplete="off" spellcheck="false">
      <select id="space"><option value="">connect first</option></select>
      <button class="btn" id="connect">Connect</button>
    </div>
    <div id="error" class="err" hidden></div>

    <div id="thread">
      <div class="msg mapi empty">Ask anything that lives in this space. Every answer
cites the memories it used, and says so when it has nothing.</div>
    </div>

    <form id="form">
      <textarea id="input" rows="1" placeholder="Ask a question…" autocomplete="off"></textarea>
      <button class="btn pri" id="send" type="submit" disabled>Ask</button>
    </form>
  </main>

  <aside>
    <h2>Memories used</h2>
    <div id="sources"><div class="empty">Sources for the last answer appear here.</div></div>
  </aside>
</div>

<script>
const $ = (id) => document.getElementById(id);
const KEY = 'mapi.chat.key', SPACE = 'mapi.chat.space';
let history = [];

function fail(message) {
  const box = $('error');
  box.textContent = message;
  box.hidden = !message;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: {
      'Authorization': 'Bearer ' + $('key').value.trim(),
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // The API speaks problem+json, so prefer its own words over a status code.
    throw new Error(body.detail || body.title || ('request failed: ' + res.status));
  }
  return body;
}

async function connect() {
  fail('');
  const key = $('key').value.trim();
  if (!key) return fail('Paste an API key first.');
  try {
    const data = await api('/v1/spaces');
    const spaces = data.items || [];
    if (!spaces.length) return fail('This key has no spaces yet. Create one first.');
    const select = $('space');
    select.innerHTML = '';
    for (const s of spaces) {
      const option = document.createElement('option');
      option.value = s.id;
      option.textContent = s.name + '  (' + s.memory_count + ')';
      select.appendChild(option);
    }
    const remembered = localStorage.getItem(SPACE);
    if (remembered && spaces.some(s => s.id === remembered)) select.value = remembered;
    localStorage.setItem(KEY, key);
    localStorage.setItem(SPACE, select.value);
    $('send').disabled = false;
    $('input').focus();
  } catch (e) { fail(e.message); }
}

function bubble(cls, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  $('thread').appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

function renderReply(el, answer) {
  // Citation markers get emphasised rather than stripped: [2] is the thread
  // back to the memory, and the right-hand column numbers match.
  el.innerHTML = '';
  const parts = answer.reply.split(/(\\[\\d{1,2}\\])/g);
  for (const part of parts) {
    if (/^\\[\\d{1,2}\\]$/.test(part)) {
      const span = document.createElement('span');
      span.className = 'cite';
      span.textContent = part;
      el.appendChild(span);
    } else {
      el.appendChild(document.createTextNode(part));
    }
  }
  if (answer.used_unverified) {
    const flag = document.createElement('div');
    flag.className = 'flag';
    flag.textContent = 'This answer uses a memory marked unverified.';
    el.appendChild(flag);
  }
}

function renderSources(citations) {
  const box = $('sources');
  box.innerHTML = '';
  if (!citations.length) {
    box.innerHTML = '<div class="empty">Nothing in this space matched.</div>';
    return;
  }
  citations.forEach((c, i) => {
    const card = document.createElement('div');
    card.className = 'src ' + (c.cited ? 'used' : 'unused');
    const when = c.occurred_at
      ? new Date(c.occurred_at).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })
      : '';
    const head = document.createElement('div');
    head.className = 'top';
    head.innerHTML = '<span class="n">' + (i + 1) + '</span>'
      + '<span class="when">' + when + ' · ' + c.score.toFixed(3) + '</span>'
      + (c.confidence === 'unverified' ? '<span class="unv">UNVERIFIED</span>' : '');
    card.appendChild(head);
    const body = document.createElement('div');
    body.textContent = c.content;
    card.appendChild(body);
    for (const tag of (c.tags || []).slice(0, 4)) {
      const chip = document.createElement('span');
      chip.className = 'tag';
      chip.textContent = tag;
      card.appendChild(chip);
    }
    box.appendChild(card);
  });
}

async function ask(event) {
  event.preventDefault();
  const message = $('input').value.trim();
  const space = $('space').value;
  if (!message || !space) return;
  fail('');
  $('input').value = '';
  $('send').disabled = true;
  bubble('you', message);
  const pending = bubble('mapi', 'searching memory');
  pending.classList.add('dots');

  try {
    const answer = await api('/v1/spaces/' + space + '/chat', {
      method: 'POST',
      body: JSON.stringify({ message, history, k: 8 }),
    });
    pending.classList.remove('dots');
    renderReply(pending, answer);
    renderSources(answer.citations || []);
    history.push({ role: 'user', content: message });
    history.push({ role: 'assistant', content: answer.reply });
    history = history.slice(-16);
  } catch (e) {
    pending.classList.remove('dots');
    pending.textContent = '—';
    fail(e.message);
  } finally {
    $('send').disabled = false;
    $('input').focus();
  }
}

$('connect').addEventListener('click', connect);
$('form').addEventListener('submit', ask);
$('space').addEventListener('change', () => {
  localStorage.setItem(SPACE, $('space').value);
  history = [];
});
$('input').addEventListener('input', (e) => {
  e.target.style.height = 'auto';
  e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px';
});
$('input').addEventListener('keydown', (e) => {
  // Enter sends, shift+enter breaks the line -- the convention every chat
  // client uses, so muscle memory works.
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('form').requestSubmit(); }
});

const saved = localStorage.getItem(KEY);
if (saved) { $('key').value = saved; connect(); }
</script>
"""

__all__ = ["PAGE"]
