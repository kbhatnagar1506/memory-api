# ruff: noqa: E501
# The module body is one embedded HTML document. Wrapping markup mid-attribute
# to satisfy a Python line-length rule makes the template harder to read and
# easier to break; the exemption is scoped to this file only.
"""A read-only memory-graph viewer, served by the app itself.

Deliberately one self-contained page with no build step, no CDN and no
framework: it is a debugging surface for the memory graph, and a viewer that
needs a toolchain is a viewer nobody runs. Force layout is ~30 lines of
Verlet integration, which is enough for the few hundred nodes a space holds.

What it shows that a graph built from write-time extraction cannot: edges
BETWEEN memories. An extraction graph links a document to the chunks pulled
out of it, so every node has exactly one parent and the picture is a forest
of disconnected stars -- it can only answer "where did this text come from".
These edges answer "what does the system believe, and why": what replaced
what (supersedes), what disagrees with what (contradicts), and what was
computed from what (derived_from).

Node colour is STATUS, because status is the part that is invisible in every
competing graph: a superseded fact still exists and is still reachable, and a
STALE derivation is a computed fact whose evidence moved out from under it.
Sessions are the other axis and get their own colouring, because a memory
store fed by a chat product is fed one session at a time and "which
conversation did this come from" is the first question anyone asks.

One space is one person. The picker lists every space with its memory count,
so a multi-tenant store can be walked without editing the URL.
"""

from __future__ import annotations

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mapi · memory graph</title>
<style>
  :root {
    --bg:#fbfbfa; --panel:#fff; --line:#e6e4e0; --text:#1b1b19; --dim:#75726c;
    --hair:#f0eeea; --shadow:0 1px 2px rgba(20,18,14,.06),0 8px 24px rgba(20,18,14,.06);
    --supersedes:#b8860b; --contradicts:#c2352a; --derived:#2b6ea8; --references:#3f7a5f;
    --active:#2f9e5e; --stale:#c2352a; --superseded:#a8a49c; --archived:#c9c5bd;
    --accent:#2b6ea8;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text); overflow:hidden;
         font:13px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
         -webkit-font-smoothing:antialiased }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace }
  #wrap { display:flex; height:100vh }
  /* left rail */
  #rail { width:210px; flex:none; border-right:1px solid var(--line); background:var(--panel);
          padding:18px 12px; display:flex; flex-direction:column; gap:4px }
  .brand { font:600 15px/1 ui-monospace,Menlo,monospace; letter-spacing:.02em;
           padding:4px 10px 16px; color:var(--text) }
  .nav { display:flex; align-items:center; gap:10px; width:100%; padding:9px 10px;
         border:0; border-radius:8px; background:transparent; color:var(--dim);
         font:inherit; font-weight:500; cursor:pointer; text-align:left }
  .nav:hover { background:var(--hair); color:var(--text) }
  .nav.on { background:var(--hair); color:var(--text); font-weight:600 }
  .ico { font-size:14px; width:16px; text-align:center }
  .railfoot { margin-top:auto }
  .sub { color:var(--dim); margin:0 0 14px }
  /* tags */
  #tags { display:flex; flex-wrap:wrap; gap:6px }
  .tag { padding:3px 9px; border:1px solid var(--line); border-radius:20px; cursor:pointer;
         font-size:12px; color:var(--dim); background:var(--panel);
         font-family:ui-monospace,Menlo,monospace }
  .tag:hover { border-color:#cdc9c2; color:var(--text) }
  .tag.on { background:var(--accent); border-color:var(--accent); color:#fff }
  /* replay scrubber */
  #scrub { position:absolute; left:24px; right:24px; bottom:44px; background:var(--panel);
           border:1px solid var(--line); border-radius:11px; padding:13px 16px;
           box-shadow:var(--shadow) }
  .scrubhead { display:flex; justify-content:space-between; align-items:baseline;
               margin-bottom:8px }
  .scrubhead b { font-size:15px; font-family:ui-monospace,Menlo,monospace }
  .scrubhead span { color:var(--dim) }
  #time { width:100%; accent-color:var(--accent) }
  .scrubfoot { display:flex; justify-content:space-between; color:var(--dim);
               font-size:11px; margin-top:4px }
  .view[hidden] { display:none }
  .ev { padding:8px 0; border-bottom:1px solid var(--hair); cursor:pointer; font-size:12.5px }
  .ev:hover { color:var(--accent) }
  .ev small { color:var(--dim); font-family:ui-monospace,Menlo,monospace }
  #stage { flex:1; position:relative; min-width:0 }
  svg { width:100%; height:100%; cursor:grab; display:block }
  svg:active { cursor:grabbing }
  #side { width:370px; flex:none; border-left:1px solid var(--line); background:var(--panel);
          padding:20px; overflow-y:auto }
  h1 { font-size:11px; letter-spacing:.12em; text-transform:uppercase;
       color:var(--dim); margin:0 0 12px; font-weight:600 }
  h1.sec { margin-top:24px }
  .stat { display:flex; justify-content:space-between; padding:3px 0; color:var(--dim) }
  .stat b { color:var(--text); font-weight:600; font-variant-numeric:tabular-nums }
  .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:8px }
  .ln { display:inline-block; width:15px; height:2px; margin-right:8px; vertical-align:middle }

  /* -- space picker ---------------------------------------------------- */
  #picker { width:100%; display:flex; align-items:center; justify-content:space-between;
            gap:8px; padding:9px 11px; border:1px solid var(--line); border-radius:7px;
            background:var(--panel); color:var(--text); cursor:pointer; font:inherit;
            box-shadow:var(--shadow) }
  #picker:hover { border-color:#d6d3cd }
  #picker .who { font-family:ui-monospace,Menlo,monospace; font-size:12px; overflow:hidden;
                 text-overflow:ellipsis; white-space:nowrap }
  #picker .caret { color:var(--dim); flex:none }
  #modal { position:fixed; inset:0; background:rgba(24,22,18,.28); display:none;
           align-items:center; justify-content:center; z-index:10 }
  #modal.on { display:flex }
  #sheet { width:min(560px,92vw); max-height:78vh; display:flex; flex-direction:column;
           background:var(--panel); border:1px solid var(--line); border-radius:14px;
           box-shadow:0 24px 60px rgba(20,18,14,.22); padding:24px }
  #sheet h2 { margin:0 0 4px; font-size:19px; font-weight:650 }
  #sheet p.sub { margin:0 0 16px; color:var(--dim) }
  #q { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px;
       background:var(--bg); color:var(--text); font:inherit; margin-bottom:12px }
  #q:focus { outline:2px solid var(--accent); outline-offset:-1px; border-color:transparent }
  #list { overflow-y:auto; margin:0 -8px; flex:1 }
  .row { display:flex; align-items:center; gap:11px; padding:9px 12px; border-radius:8px;
         cursor:pointer }
  .row:hover { background:var(--hair) }
  .row.on { background:var(--hair) }
  .row .dot { width:15px; height:15px; border:1.5px solid #cdc9c2; border-radius:4px; flex:none }
  .row.on .dot { border-color:var(--accent); background:var(--accent);
                 box-shadow:inset 0 0 0 3px var(--panel) }
  .row .nm { flex:1; font-family:ui-monospace,Menlo,monospace; font-size:12.5px;
             overflow:hidden; text-overflow:ellipsis; white-space:nowrap }
  .row .ct { color:var(--dim); font-variant-numeric:tabular-nums }
  #sheetfoot { display:flex; align-items:center; justify-content:space-between;
               margin-top:16px; padding-top:14px; border-top:1px solid var(--line) }
  #shown { color:var(--dim) }
  .btns { display:flex; gap:9px }
  button.b { padding:9px 17px; border-radius:8px; font:inherit; font-weight:600; cursor:pointer;
             letter-spacing:.03em; border:1px solid var(--line); background:var(--panel);
             color:var(--text) }
  button.b:hover { background:var(--hair) }
  button.b.pri { background:var(--accent); border-color:var(--accent); color:#fff }
  button.b.pri:hover { filter:brightness(1.07) }

  /* -- question card --------------------------------------------------- */
  #ask { margin-top:16px; padding:13px; border:1px solid var(--line); border-radius:9px;
         background:var(--bg) }
  #ask .qt { display:inline-block; font-size:11px; letter-spacing:.06em; color:var(--dim);
             text-transform:uppercase; margin-bottom:7px; font-weight:600 }
  #ask .qq { font-weight:550 }
  #ask .qa { margin-top:7px; color:var(--dim) }
  #ask .qa b { color:var(--active); font-weight:600 }

  /* -- colour-by toggle ------------------------------------------------ */
  .seg { display:flex; border:1px solid var(--line); border-radius:8px; overflow:hidden;
         margin-bottom:12px }
  .seg button { flex:1; padding:7px 4px; font-size:12px; border:0; background:var(--panel); color:var(--dim);
                font:inherit; font-weight:600; cursor:pointer }
  .seg button.on { background:var(--hair); color:var(--text) }

  /* -- detail ---------------------------------------------------------- */
  #detail { margin-top:22px; padding-top:18px; border-top:1px solid var(--line); display:none }
  #detail.on { display:block }
  #content { background:var(--bg); border:1px solid var(--line); border-radius:8px;
             padding:12px; margin:11px 0; white-space:pre-wrap; word-break:break-word;
             max-height:240px; overflow-y:auto }
  .rel { padding:9px 0; border-bottom:1px solid var(--hair); cursor:pointer }
  .rel:hover { color:var(--accent) }
  .rel small { color:var(--dim) }
  .empty { color:var(--dim); font-style:italic }
  .pill { display:inline-block; padding:2px 8px; border-radius:20px; font-size:11px;
          border:1px solid var(--line); margin:0 5px 5px 0; color:var(--dim);
          font-family:ui-monospace,Menlo,monospace }
  circle { cursor:pointer }
  circle.sel { stroke:#1b1b19; stroke-width:2.5 }
  #hint { position:absolute; left:18px; bottom:16px; color:var(--dim); font-size:12px }
</style>
<div id="wrap">
  <nav id="rail">
    <div class="brand">mapi</div>
    <button class="nav on" data-view="graph"><span class="ico">◍</span>Memory graph</button>
    <button class="nav" data-view="replay"><span class="ico">◷</span>Memory replay</button>
    <button class="nav" data-view="keys"><span class="ico">⌁</span>API keys</button>
    <div class="railfoot"><button id="picker"><span class="who">loading…</span><span class="caret">▾</span></button></div>
  </nav>

  <div id="stage">
    <svg id="svg"></svg>
    <div id="hint">drag to pan · scroll to zoom · click a memory</div>
    <div id="scrub" hidden>
      <div class="scrubhead"><b id="scrubdate">—</b><span id="scrubcount"></span></div>
      <input id="time" type="range" min="0" max="100" value="100">
      <div class="scrubfoot"><span id="t0"></span><span>event time — when it happened, not when we learned it</span><span id="t1"></span></div>
    </div>
  </div>

  <div id="side">
    <div id="ask" hidden></div>

    <div class="view" data-view="graph">
      <h1>Colour by</h1>
      <div class="seg">
        <button id="by-status" class="on">status</button>
        <button id="by-session">session</button>
      </div>
      <div id="legend"></div>
      <h1 class="sec">Show</h1>
      <div class="seg">
        <button class="kind on" data-kind="all">all</button>
        <button class="kind" data-kind="episodic">what happened</button>
        <button class="kind" data-kind="derived">what's true</button>
      </div>
      <div id="tags"></div>
      <h1 class="sec">Counts</h1>
      <div id="stats"></div>
    </div>

    <div class="view" data-view="keys" hidden>
      <h1>API keys</h1>
      <p class="sub">A key authorises requests for this organization. It does not
      identify a person &mdash; that is what your Google account is for.</p>
      <div id="keylist"></div>
      <form id="keyform" style="display:flex;gap:8px;margin-top:16px">
        <input id="keyname" placeholder="Key name" required maxlength="120"
               style="flex:1;padding:9px 11px;border:1px solid var(--line);border-radius:8px;font:inherit;background:var(--bg)">
        <button class="b pri" type="submit">Create</button>
      </form>
      <div id="newkey" hidden></div>
    </div>

    <div class="view" data-view="replay" hidden>
      <h1>What entered memory</h1>
      <p class="sub">Scrub the timeline to see the graph as of a moment. Newest first.</p>
      <div id="feed"></div>
    </div>

    <div id="detail">
      <h1>Memory</h1>
      <div id="meta"></div>
      <div id="content"></div>
      <div id="rels"></div>
    </div>
  </div>
</div>
<div id="modal">
  <div id="sheet">
    <h2>Filter by space</h2>
    <p class="sub">One space is one person. Counts are ACTIVE memories, so they read lower than the graph's total by however many are superseded.</p>
    <input id="q" placeholder="Search spaces…" autocomplete="off">
    <div id="list"></div>
    <div id="sheetfoot">
      <span id="shown"></span>
      <div class="btns">
        <button class="b" id="cancel">CANCEL</button>
        <button class="b pri" id="apply">APPLY</button>
      </div>
    </div>
  </div>
</div>
<script>
const qs = new URLSearchParams(location.search);
// The dashboard is reached as /orgs/{id}, so provenance comes from the path
// and authentication from the session cookie. No key in the URL: a key in a
// query string ends up in history, referrers and logs.
const ORG = location.pathname.split('/')[2] || '';
const BASE = `/orgs/${ORG}/api`;
let SPACE = qs.get('space');

const EDGE_COLOR = { supersedes:'#b8860b', contradicts:'#c2352a',
                     derived_from:'#2b6ea8', references:'#3f7a5f' };
const NODE_COLOR = { active:'#2f9e5e', superseded:'#a8a49c',
                     stale:'#c2352a', archived:'#c9c5bd' };
// Distinct hues for sessions. Twelve, then it wraps -- a legend nobody can
// tell apart is worse than an honest repeat.
const SESSION_HUES = [204,28,140,320,52,264,176,4,96,236,320,68];

const svg = document.getElementById('svg');
let nodes = [], edges = [], byId = {}, sel = null, spaces = [], pending = null;
let colourBy = 'status', sessions = [];
let allNodes = [], allEdges = [];          // unfiltered, as loaded
let activeTags = new Set(), cutoff = null; // tag filter and replay cutoff
let kindFilter = 'all';                    // episodic | derived | all
let view = 'graph';
let camera = { x:0, y:0, k:1 };

const esc = s => (s||'').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function api(path, options) {
  const r = await fetch(path, { credentials:'same-origin', ...(options || {}) });
  if (!r.ok) throw new Error(path + ' -> ' + r.status);
  return r.json();
}

const sessionOf = n =>
  (n.tags || []).filter(t => t.startsWith('session:')).map(t => t.slice(8))[0] || '';

function nodeColor(n) {
  if (colourBy === 'status') return NODE_COLOR[n.status] || '#888';
  const i = sessions.indexOf(sessionOf(n));
  if (i < 0) return '#c9c5bd';
  return `hsl(${SESSION_HUES[i % SESSION_HUES.length]} 62% 48%)`;
}

// -- layout ------------------------------------------------------------
function layout() {
  const W = svg.clientWidth, H = svg.clientHeight;
  // Seeded by index so the picture is stable across reloads -- a graph that
  // rearranges every refresh cannot be reasoned about.
  nodes.forEach((n, i) => {
    const a = i * 2.399963;                    // golden angle
    const r = 26 * Math.sqrt(i);
    n.x = W/2 + r * Math.cos(a); n.y = H/2 + r * Math.sin(a);
    n.vx = 0; n.vy = 0;
  });
  for (let step = 0; step < 220; step++) {
    for (const e of edges) {                    // springs pull related memories together
      const a = byId[e.source], b = byId[e.target];
      if (!a || !b) continue;
      let dx = b.x-a.x, dy = b.y-a.y;
      const d = Math.hypot(dx,dy) || 0.01, f = (d-92) * 0.012;
      dx /= d; dy /= d;
      a.vx += dx*f; a.vy += dy*f; b.vx -= dx*f; b.vy -= dy*f;
    }
    for (let i = 0; i < nodes.length; i++) {    // everything repels everything
      for (let j = i+1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = b.x-a.x, dy = b.y-a.y;
        const d2 = dx*dx+dy*dy+0.01, d = Math.sqrt(d2);
        if (d > 340) continue;
        const f = 2400/d2;
        dx /= d; dy /= d;
        a.vx -= dx*f; a.vy -= dy*f; b.vx += dx*f; b.vy += dy*f;
      }
    }
    for (const n of nodes) {
      n.vx += (W/2-n.x)*0.0016; n.vy += (H/2-n.y)*0.0016;   // gentle centring
      n.x += n.vx; n.y += n.vy; n.vx *= 0.86; n.vy *= 0.86;
    }
  }
}

function fit() {
  // Frame whatever the layout produced. Without this the picture depends on
  // the canvas size at load: a narrow pane pushes nodes off the left edge,
  // and a graph you have to hunt for is a graph nobody reads.
  if (!nodes.length) return;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = svg.clientWidth, H = svg.clientHeight, pad = 60;
  const k = Math.min((W - pad*2) / (x1-x0 || 1), (H - pad*2) / (y1-y0 || 1), 1.6);
  camera = { k, x: W/2 - k*(x0+x1)/2, y: H/2 - k*(y0+y1)/2 };
}

function draw() {
  const g = [`<g transform="translate(${camera.x},${camera.y}) scale(${camera.k})">`];
  for (const e of edges) {
    const a = byId[e.source], b = byId[e.target];
    if (!a || !b) continue;
    const dash = e.type === 'contradicts' ? ' stroke-dasharray="5 3"' : '';
    g.push(`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${
      EDGE_COLOR[e.type]||'#bbb'}" stroke-width="1.4" opacity=".7"${dash}/>`);
  }
  for (const n of nodes) {
    const r = 5 + Math.min(n.degree, 8) * 1.5;
    g.push(`<circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${nodeColor(n)}"
      stroke="#fbfbfa" stroke-width="1.5" data-id="${n.id}" class="${sel===n.id?'sel':''}"><title>${
      esc(n.content.slice(0,110))}</title></circle>`);
  }
  g.push('</g>');
  svg.innerHTML = g.join('');
  svg.querySelectorAll('circle').forEach(c =>
    c.onclick = ev => { ev.stopPropagation(); show(c.dataset.id); });
}

function legend() {
  const el = document.getElementById('legend');
  if (colourBy === 'status') {
    el.innerHTML = [
      ['active', 'current'],
      ['superseded', 'replaced, still reachable'],
      ['stale', 'its evidence moved'],
    ].map(([k, note]) =>
      `<div class="stat"><span><i class="sw" style="background:var(--${k})"></i>${k}</span>` +
      `<span>${note}</span></div>`).join('');
  } else {
    el.innerHTML = sessions.map((s, i) => {
      const n = nodes.filter(x => sessionOf(x) === s).length;
      return `<div class="stat"><span><i class="sw" style="background:hsl(${
        SESSION_HUES[i % SESSION_HUES.length]} 62% 48%)"></i><span class="mono">${
        esc(s.slice(0,22))}</span></span><b>${n}</b></div>`;
    }).join('') || '<p class="empty">No session tags on these memories.</p>';
  }
}

// -- filtering ---------------------------------------------------------
// One place decides what is on screen, so the tag filter and the replay
// scrubber cannot disagree about it. Edges survive only when BOTH endpoints
// do -- an edge to a hidden memory is a line to nowhere, and in a graph whose
// whole claim is "these edges are the point", a dangling one is a lie.
function applyFilters() {
  const keep = allNodes.filter(n => {
    if (cutoff && new Date(n.occurred_at) > cutoff) return false;
    if (kindFilter !== 'all' && n.kind !== kindFilter) return false;
    if (activeTags.size && !(n.tags || []).some(t => activeTags.has(t))) return false;
    return true;
  });
  const live = new Set(keep.map(n => n.id));
  nodes = keep;
  edges = allEdges.filter(e => live.has(e.source) && live.has(e.target));
  byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  if (sel && !live.has(sel)) { sel = null; document.getElementById('detail').classList.remove('on'); }
  layout(); fit(); draw();
}

function renderTags() {
  // Session tags are the session colouring's job; showing 12 of them here
  // would bury the tags a person actually chose.
  const counts = {};
  for (const n of allNodes)
    for (const t of n.tags || [])
      if (!t.startsWith('session:')) counts[t] = (counts[t] || 0) + 1;
  const names = Object.keys(counts).sort((a, b) => counts[b] - counts[a]).slice(0, 24);
  const el = document.getElementById('tags');
  // Rendered only when tags a PERSON chose exist. Session ids are already the
  // session colouring, and an always-empty panel is worse than no panel.
  el.innerHTML = names.length
    ? '<h1 class="sec">Tags</h1>' + names.map(t =>
        `<span class="tag ${activeTags.has(t) ? 'on' : ''}" data-tag="${esc(t)}">${
          esc(t)} ${counts[t]}</span>`).join('')
    : '';
  el.querySelectorAll('[data-tag]').forEach(chip => chip.onclick = () => {
    const t = chip.dataset.tag;
    activeTags.has(t) ? activeTags.delete(t) : activeTags.add(t);
    renderTags(); applyFilters();
  });
}

document.querySelectorAll('.kind').forEach(btn => btn.onclick = () => {
  kindFilter = btn.dataset.kind;
  document.querySelectorAll('.kind').forEach(b => b.classList.toggle('on', b === btn));
  applyFilters();
});

// -- replay ------------------------------------------------------------
function replayBounds() {
  const times = allNodes.map(n => new Date(n.occurred_at)).sort((a, b) => a - b);
  return times.length ? [times[0], times[times.length - 1]] : [null, null];
}

function renderReplay(fraction) {
  const [lo, hi] = replayBounds();
  if (!lo) return;
  const at = new Date(lo.getTime() + (hi - lo) * fraction);
  cutoff = at;
  document.getElementById('scrubdate').textContent = at.toISOString().slice(0, 10);
  document.getElementById('t0').textContent = lo.toISOString().slice(0, 10);
  document.getElementById('t1').textContent = hi.toISOString().slice(0, 10);
  applyFilters();
  document.getElementById('scrubcount').textContent =
    `${nodes.length} of ${allNodes.length} memories · ${edges.length} relations`;

  const recent = [...nodes].sort(
    (a, b) => new Date(b.occurred_at) - new Date(a.occurred_at)).slice(0, 40);
  document.getElementById('feed').innerHTML = recent.map(n =>
    `<div class="ev" data-go="${n.id}">${esc(n.content.slice(0, 110))}<br>
     <small>${n.occurred_at.slice(0, 10)} · ${n.kind} · ${n.status}</small></div>`).join('')
    || '<p class="empty">Nothing had happened yet.</p>';
  document.getElementById('feed').querySelectorAll('[data-go]').forEach(
    el => el.onclick = () => show(el.dataset.go));
}

document.getElementById('time').oninput = e => renderReplay(e.target.value / 100);

// -- api keys ----------------------------------------------------------
async function renderKeys() {
  const el = document.getElementById('keylist');
  try {
    const data = await api(`${BASE}/keys`);
    el.innerHTML = (data.items || []).map(k =>
      `<div class="stat" style="padding:9px 0;border-bottom:1px solid var(--hair)">
         <span>${esc(k.name)}</span>
         <span class="mono" style="font-size:11px">${esc(k.created_at.slice(0,10))}</span>
       </div>`).join('') || '<p class="empty">No keys yet.</p>';
  } catch (err) {
    el.innerHTML = `<p class="empty">${esc(err.message)}</p>`;
  }
}

document.getElementById('keyform').onsubmit = async e => {
  e.preventDefault();
  const name = document.getElementById('keyname').value.trim();
  if (!name) return;
  const r = await fetch(`/orgs/${ORG}/keys.json`, {
    method:'POST', credentials:'same-origin',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({name}),
  });
  const data = await r.json();
  const box = document.getElementById('newkey');
  box.hidden = false;
  // Shown once. Only the hash is stored, so a reload cannot recover it.
  box.innerHTML = `<h1 class="sec">Copy this now</h1>
    <div id="content" class="mono" style="word-break:break-all">${esc(data.key || data.detail || '')}</div>
    <p class="empty">It is stored hashed and will not be shown again.</p>`;
  document.getElementById('keyname').value = '';
  renderKeys();
};

// -- view switching ----------------------------------------------------
document.querySelectorAll('.nav').forEach(btn => btn.onclick = () => {
  view = btn.dataset.view;
  document.querySelectorAll('.nav').forEach(b => b.classList.toggle('on', b === btn));
  document.querySelectorAll('.view').forEach(v => v.hidden = v.dataset.view !== view);
  document.getElementById('scrub').hidden = view !== 'replay';
  document.getElementById('hint').hidden = view === 'replay';
  if (view === 'replay') {
    renderReplay(document.getElementById('time').value / 100);
  } else if (view === 'keys') {
    renderKeys();
  } else {
    cutoff = null; applyFilters();
  }
  document.getElementById('stage').hidden = view === 'keys';
});

// -- detail ------------------------------------------------------------
async function show(id) {
  sel = id; draw();
  const n = byId[id];
  document.getElementById('detail').classList.add('on');
  const ses = sessionOf(n);
  document.getElementById('meta').innerHTML =
    `<span class="pill" style="border-color:${NODE_COLOR[n.status]}">${n.status}</span>` +
    `<span class="pill">${n.kind}</span>` +
    `<span class="pill">${n.occurred_at.slice(0,10)}</span>` +
    (ses ? `<span class="pill">session ${esc(ses.slice(0,18))}</span>` : '');
  document.getElementById('content').textContent = 'loading…';
  const rels = document.getElementById('rels');
  rels.innerHTML = '';
  try {
    // The context endpoint resolves the whole neighbourhood in one call --
    // currency, what it replaced, provenance, derivatives, contradictions.
    const ctx = await api(`${BASE}/context/${id}?space=${SPACE}`);
    document.getElementById('content').textContent = ctx.memory.content;
    const groups = [
      // "current version", not "stale" -- stale is a distinct status in this
      // system (a derivation whose evidence moved) and reusing the word here
      // would contradict the legend directly above.
      ['current version', ctx.current_head],
      ['replaced', ctx.replaced],
      ['derived from', ctx.derived_from],
      ['derivatives', ctx.derivatives],
      ['contradicts', ctx.contradicts],
      ['references', ctx.references],
    ];
    let any = false;
    for (const [label, items] of groups) {
      if (!items || !items.length) continue;
      any = true;
      rels.insertAdjacentHTML('beforeend', `<h1 class="sec">${label}</h1>`);
      for (const m of items)
        rels.insertAdjacentHTML('beforeend',
          `<div class="rel" data-go="${m.id}">${esc(m.content.slice(0,140))}<br>
           <small>${m.status} · ${m.occurred_at.slice(0,10)}</small></div>`);
    }
    if (!any) rels.innerHTML = '<p class="empty">No relations. This memory stands alone.</p>';
    rels.querySelectorAll('[data-go]').forEach(el =>
      el.onclick = () => show(el.dataset.go));
  } catch (err) {
    document.getElementById('content').textContent = 'failed to load: ' + err.message;
  }
}

// -- space picker ------------------------------------------------------
function renderList() {
  const term = document.getElementById('q').value.trim().toLowerCase();
  const hits = spaces.filter(s => !term || s.name.toLowerCase().includes(term)
                                        || s.slug.toLowerCase().includes(term));
  document.getElementById('list').innerHTML = hits.map(s =>
    `<div class="row ${pending === s.id ? 'on' : ''}" data-id="${s.id}">
       <span class="dot"></span><span class="nm">${esc(s.name)}</span>
       <span class="ct">${s.memory_count ?? ''}</span>
     </div>`).join('') || '<p class="empty" style="padding:12px">No space matches.</p>';
  document.getElementById('shown').textContent =
    term ? `${hits.length} of ${spaces.length}` : 'Showing all';
  document.querySelectorAll('.row').forEach(r =>
    r.onclick = () => { pending = r.dataset.id; renderList(); });
}

function openPicker() {
  pending = SPACE;
  document.getElementById('modal').classList.add('on');
  document.getElementById('q').value = '';
  renderList();
  document.getElementById('q').focus();
}
const closePicker = () => document.getElementById('modal').classList.remove('on');

document.getElementById('picker').onclick = openPicker;
document.getElementById('cancel').onclick = closePicker;
document.getElementById('apply').onclick = () => {
  closePicker();
  if (pending && pending !== SPACE) { SPACE = pending; loadSpace(); }
};
document.getElementById('q').oninput = renderList;
document.getElementById('modal').onclick = e => {
  if (e.target.id === 'modal') closePicker();
};
addEventListener('keydown', e => { if (e.key === 'Escape') closePicker(); });

for (const [id, mode] of [['by-status','status'], ['by-session','session']]) {
  document.getElementById(id).onclick = () => {
    colourBy = mode;
    document.getElementById('by-status').classList.toggle('on', mode === 'status');
    document.getElementById('by-session').classList.toggle('on', mode === 'session');
    legend(); draw();
  };
}

// -- load --------------------------------------------------------------
async function loadSpace() {
  const space = spaces.find(s => s.id === SPACE);
  document.querySelector('#picker .who').textContent = space ? space.name : SPACE;

  const ask = document.getElementById('ask');
  const meta = space && space.metadata || {};
  if (meta.question) {
    ask.hidden = false;
    ask.innerHTML =
      `<span class="qt">${esc(meta.question_type || 'question')}</span>` +
      `<div class="qq">${esc(meta.question)}</div>` +
      `<div class="qa">answer: <b>${esc(meta.answer || '')}</b></div>`;
  } else ask.hidden = true;

  const g = await api(`${BASE}/graph?space=${SPACE}&limit=400`);
  allNodes = g.nodes; allEdges = g.edges; sel = null;
  nodes = allNodes; edges = allEdges;
  byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  sessions = [...new Set(nodes.map(sessionOf).filter(Boolean))].sort();
  activeTags.clear(); cutoff = null;
  renderTags();
  document.getElementById('detail').classList.remove('on');

  const c = g.counts;
  document.getElementById('stats').innerHTML = [
    ['memories', c.memories], ['sessions', sessions.length], ['relations', c.edges],
    ['isolated', c.isolated + ' / ' + c.memories],
    ['superseded', c.by_status.superseded], ['stale', c.by_status.stale],
    ['contradictions', c.by_edge.contradicts], ['derived facts', c.by_kind.derived],
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join('');

  legend(); layout(); fit(); draw();
  if (view === 'replay') renderReplay(document.getElementById('time').value / 100);
}

(async function () {
  try {
    const list = await api(`${BASE}/spaces`);
    spaces = list.items || [];
  } catch (err) {
    document.getElementById('stats').innerHTML =
      `<p class="empty">Could not load this organization: ${esc(err.message)}</p>`;
    return;
  }
  if (!spaces.length) {
    document.querySelector('#picker .who').textContent = 'no spaces';
    return;
  }
  if (!SPACE || !spaces.some(s => s.id === SPACE)) SPACE = spaces[0].id;
  await loadSpace();
})();

addEventListener('resize', () => { if (nodes.length) { fit(); draw(); } });

// -- pan and zoom ------------------------------------------------------
let drag = null;
svg.addEventListener('mousedown', e => {
  // Never start a pan on a node. Otherwise the few pixels of jitter in an
  // ordinary click drag the graph out from under the cursor, and the click
  // lands on the background instead of the memory you aimed at.
  if (e.target.tagName === 'circle') return;
  drag = { x:e.clientX, y:e.clientY, xv:camera.x, yv:camera.y, live:false };
});
addEventListener('mouseup', () => drag = null);
addEventListener('mousemove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  // A 4px threshold, so a shaky click on empty canvas is still a click.
  if (!drag.live && Math.hypot(dx, dy) < 4) return;
  drag.live = true;
  camera = { ...camera, x: drag.xv + dx, y: drag.yv + dy };
  draw();
});
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.1 : 0.9;
  camera = { x: e.offsetX - (e.offsetX-camera.x)*f, y: e.offsetY - (e.offsetY-camera.y)*f,
             k: camera.k*f };
  draw();
}, { passive:false });
</script>
"""

__all__ = ["PAGE"]
