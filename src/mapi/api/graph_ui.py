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
"""

from __future__ import annotations

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>mapi · memory graph</title>
<style>
  :root {
    --bg:#0b0d0e; --panel:#131617; --line:#242829; --text:#e6e8e9; --dim:#8b9498;
    --supersedes:#c9a227; --contradicts:#e0533d; --derived:#4fa3c7; --references:#3f6b57;
    --active:#4ade80; --stale:#e0533d; --superseded:#6b7280; --archived:#3f4448;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text);
         font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; overflow:hidden }
  #wrap { display:flex; height:100vh }
  #stage { flex:1; position:relative }
  svg { width:100%; height:100%; cursor:grab }
  svg:active { cursor:grabbing }
  #side { width:380px; border-left:1px solid var(--line); background:var(--panel);
          padding:18px; overflow-y:auto }
  h1 { font-size:13px; letter-spacing:.14em; text-transform:uppercase;
       color:var(--dim); margin:0 0 14px; font-weight:500 }
  .stat { display:flex; justify-content:space-between; padding:3px 0; color:var(--dim) }
  .stat b { color:var(--text); font-weight:500 }
  .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:7px }
  .ln { display:inline-block; width:14px; height:2px; margin-right:7px; vertical-align:middle }
  #detail { margin-top:18px; padding-top:16px; border-top:1px solid var(--line); display:none }
  #detail.on { display:block }
  #content { background:#0b0d0e; border:1px solid var(--line); border-radius:5px;
             padding:11px; margin:10px 0; white-space:pre-wrap; word-break:break-word;
             max-height:230px; overflow-y:auto; line-height:1.55 }
  .rel { padding:7px 0; border-bottom:1px solid var(--line); cursor:pointer }
  .rel:hover { color:var(--active) }
  .rel small { color:var(--dim) }
  .empty { color:var(--dim); font-style:italic }
  .pill { display:inline-block; padding:1px 7px; border-radius:3px; font-size:11px;
          border:1px solid var(--line); margin-right:5px }
  circle { cursor:pointer }
  circle.sel { stroke:#fff; stroke-width:2.5 }
</style>
<div id="wrap">
  <div id="stage"><svg id="svg"></svg></div>
  <div id="side">
    <h1>Memory graph</h1>
    <div id="stats"></div>
    <h1 style="margin-top:20px">Status</h1>
    <div class="stat"><span><i class="sw" style="background:var(--active)"></i>active</span></div>
    <div class="stat"><span><i class="sw" style="background:var(--superseded)"></i>superseded &mdash; replaced, still reachable</span></div>
    <div class="stat"><span><i class="sw" style="background:var(--stale)"></i>stale &mdash; its evidence moved</span></div>
    <h1 style="margin-top:20px">Relations</h1>
    <div class="stat"><span><i class="ln" style="background:var(--supersedes)"></i>supersedes</span></div>
    <div class="stat"><span><i class="ln" style="background:var(--contradicts)"></i>contradicts</span></div>
    <div class="stat"><span><i class="ln" style="background:var(--derived)"></i>derived from</span></div>
    <div class="stat"><span><i class="ln" style="background:var(--references)"></i>references</span></div>
    <div id="detail">
      <h1>Memory</h1>
      <div id="meta"></div>
      <div id="content"></div>
      <div id="rels"></div>
    </div>
  </div>
</div>
<script>
const qs = new URLSearchParams(location.search);
const SPACE = qs.get('space'), KEY = qs.get('key');
const EDGE_COLOR = { supersedes:'#c9a227', contradicts:'#e0533d',
                     derived_from:'#4fa3c7', references:'#3f6b57' };
const NODE_COLOR = { active:'#4ade80', superseded:'#6b7280',
                     stale:'#e0533d', archived:'#3f4448' };
const svg = document.getElementById('svg');
let nodes = [], edges = [], byId = {}, sel = null;
let view = { x:0, y:0, k:1 };

async function api(path) {
  const r = await fetch(path, { headers: KEY ? { Authorization:'Bearer '+KEY } : {} });
  if (!r.ok) throw new Error(path + ' -> ' + r.status);
  return r.json();
}

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
  view = { k, x: W/2 - k*(x0+x1)/2, y: H/2 - k*(y0+y1)/2 };
}

function draw() {
  const g = [`<g transform="translate(${view.x},${view.y}) scale(${view.k})">`];
  for (const e of edges) {
    const a = byId[e.source], b = byId[e.target];
    if (!a || !b) continue;
    const dash = e.type === 'contradicts' ? ' stroke-dasharray="5 3"' : '';
    g.push(`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${
      EDGE_COLOR[e.type]||'#333'}" stroke-width="1.4" opacity=".65"${dash}/>`);
  }
  for (const n of nodes) {
    const r = 5 + Math.min(n.degree, 8) * 1.5;
    g.push(`<circle cx="${n.x}" cy="${n.y}" r="${r}" fill="${
      NODE_COLOR[n.status]||'#888'}" stroke="#0b0d0e" stroke-width="1.5"
      data-id="${n.id}" class="${sel===n.id?'sel':''}"><title>${
      esc(n.content.slice(0,110))}</title></circle>`);
  }
  g.push('</g>');
  svg.innerHTML = g.join('');
  svg.querySelectorAll('circle').forEach(c =>
    c.onclick = ev => { ev.stopPropagation(); show(c.dataset.id); });
}

const esc = s => (s||'').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function show(id) {
  sel = id; draw();
  const n = byId[id];
  document.getElementById('detail').classList.add('on');
  document.getElementById('meta').innerHTML =
    `<span class="pill" style="border-color:${NODE_COLOR[n.status]}">${n.status}</span>` +
    `<span class="pill">${n.kind}</span>` +
    `<span class="pill">${n.occurred_at.slice(0,10)}</span>`;
  document.getElementById('content').textContent = 'loading…';
  const rels = document.getElementById('rels');
  rels.innerHTML = '';
  try {
    // The context endpoint resolves the whole neighbourhood in one call --
    // currency, what it replaced, provenance, derivatives, contradictions.
    const ctx = await api(`/v1/spaces/${SPACE}/memories/${id}/context`);
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
      rels.insertAdjacentHTML('beforeend', `<h1 style="margin-top:16px">${label}</h1>`);
      for (const m of items)
        rels.insertAdjacentHTML('beforeend',
          `<div class="rel" data-go="${m.id}">${esc(m.content.slice(0,120))}<br>
           <small>${m.status} · ${m.occurred_at.slice(0,10)}</small></div>`);
    }
    if (!any) rels.innerHTML = '<p class="empty">No relations. This memory stands alone.</p>';
    rels.querySelectorAll('[data-go]').forEach(el =>
      el.onclick = () => show(el.dataset.go));
  } catch (err) {
    document.getElementById('content').textContent = 'failed to load: ' + err.message;
  }
}

(async function () {
  if (!SPACE) {
    document.getElementById('stats').innerHTML =
      '<p class="empty">Add ?space=spc_…&key=sm_… to the URL.</p>';
    return;
  }
  const g = await api(`/v1/spaces/${SPACE}/graph?limit=400`);
  nodes = g.nodes; edges = g.edges;
  byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const c = g.counts;
  document.getElementById('stats').innerHTML = [
    ['memories', c.memories], ['relations', c.edges],
    ['isolated', c.isolated + ' / ' + c.memories],
    ['derived facts', c.by_kind.derived], ['stale', c.by_status.stale],
    ['superseded', c.by_status.superseded], ['contradictions', c.by_edge.contradicts],
  ].map(([k, v]) => `<div class="stat"><span>${k}</span><b>${v}</b></div>`).join('');
  layout(); fit(); draw();
})();

addEventListener('resize', () => { if (nodes.length) { fit(); draw(); } });

// pan and zoom
let drag = null;
svg.addEventListener('mousedown', e => {
  // Never start a pan on a node. Otherwise the few pixels of jitter in an
  // ordinary click drag the graph out from under the cursor, and the click
  // lands on the background instead of the memory you aimed at.
  if (e.target.tagName === 'circle') return;
  drag = { x:e.clientX, y:e.clientY, xv:view.x, yv:view.y, live:false };
});
addEventListener('mouseup', () => drag = null);
addEventListener('mousemove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  // A 4px threshold, so a shaky click on empty canvas is still a click.
  if (!drag.live && Math.hypot(dx, dy) < 4) return;
  drag.live = true;
  view = { ...view, x: drag.xv + dx, y: drag.yv + dy };
  draw();
});
svg.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.1 : 0.9;
  view = { x: e.offsetX - (e.offsetX-view.x)*f, y: e.offsetY - (e.offsetY-view.y)*f,
           k: view.k*f };
  draw();
}, { passive:false });
</script>
"""

__all__ = ["PAGE"]
