# ruff: noqa: E501
# One embedded HTML document per page. Wrapping markup mid-attribute to satisfy
# a Python line-length rule makes the templates harder to read and easier to
# break; the exemption is scoped to this file.
"""The browser-facing pages: landing, organizations, and the dashboard shell.

No build step, no framework, no CDN. The whole product surface is three
documents sharing one stylesheet, which is a deliberate constraint: a
dashboard that needs a toolchain is a dashboard that rots the first time
someone tries to run it a year later.

Light theme throughout, matching the wordmark.
"""

from __future__ import annotations

from html import escape

#: One place for the visual language, so the three pages cannot drift.
STYLE = """
  :root {
    --bg:#fbfbfa; --panel:#fff; --line:#e6e4e0; --hair:#f0eeea;
    --text:#1b1b19; --dim:#75726c; --accent:#1b1b19;
    --ok:#2f9e5e; --warn:#b8860b;
    --shadow:0 1px 2px rgba(20,18,14,.05), 0 8px 28px rgba(20,18,14,.06);
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
         -webkit-font-smoothing:antialiased }
  a { color:inherit }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace }
  .wrap { max-width:980px; margin:0 auto; padding:0 28px }
  header { border-bottom:1px solid var(--line); background:var(--panel) }
  header .wrap { display:flex; align-items:center; justify-content:space-between; height:66px }
  .logo { height:26px; display:block }
  nav a { margin-left:22px; color:var(--dim); text-decoration:none; font-size:14px }
  nav a:hover { color:var(--text) }
  .btn { display:inline-flex; align-items:center; gap:10px; padding:11px 20px;
         border-radius:9px; border:1px solid var(--line); background:var(--panel);
         color:var(--text); text-decoration:none; font-weight:600; font-size:14px;
         cursor:pointer; box-shadow:var(--shadow) }
  .btn:hover { border-color:#d6d3cd }
  .btn.pri { background:var(--accent); border-color:var(--accent); color:#fff }
  .btn.pri:hover { filter:brightness(1.25) }
  h1 { font-size:44px; line-height:1.12; letter-spacing:-.02em; margin:0 0 18px; font-weight:650 }
  h2 { font-size:15px; letter-spacing:.1em; text-transform:uppercase; color:var(--dim);
       font-weight:600; margin:0 0 16px }
  p.lead { font-size:19px; color:var(--dim); margin:0 0 32px; max-width:640px }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:13px;
          padding:22px; box-shadow:var(--shadow) }
  .grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)) }
  .muted { color:var(--dim) }
  .pill { display:inline-block; padding:2px 9px; border-radius:20px; font-size:12px;
          border:1px solid var(--line); color:var(--dim) }
  input[type=text] { width:100%; padding:11px 13px; border:1px solid var(--line);
                     border-radius:9px; font:inherit; background:var(--bg) }
  input[type=text]:focus { outline:2px solid var(--accent); outline-offset:-1px }
  footer { color:var(--dim); font-size:13px; padding:56px 0 40px }
  .err { border-left:3px solid #c2352a; padding:10px 14px; background:#fdf3f2;
         border-radius:0 8px 8px 0; margin-bottom:20px }
"""

_HEAD = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="icon" href="/static/mapi-logo.png">
<style>{style}</style>
"""


def _shell(title: str, body: str, *, nav: str = "") -> str:
    return (
        _HEAD.format(title=escape(title), style=STYLE)
        + f"""
<header><div class="wrap">
  <a href="/"><img class="logo" src="/static/mapi-logo.png" alt="mapi"></a>
  <nav>{nav}</nav>
</div></header>
{body}
<footer><div class="wrap">mapi — memory for agents. <a href="/docs">API reference</a></div></footer>
"""
    )


def landing(signed_in: bool, sign_in_available: bool) -> str:
    """The public page. States what the thing is without overclaiming."""
    cta = (
        '<a class="btn pri" href="/orgs">Open dashboard</a>'
        if signed_in
        else (
            '<a class="btn pri" href="/auth/google/login">Continue with Google</a>'
            if sign_in_available
            else '<span class="pill">sign-in not configured on this deployment</span>'
        )
    )
    return _shell(
        "mapi — memory for agents",
        f"""
<div class="wrap" style="padding-top:84px">
  <h1>Memory that knows what<br>it believes, and why.</h1>
  <p class="lead">A memory API for agents. Hybrid retrieval over episodes and extracted
  claims, with a graph of what replaced what, what disagrees with what, and what was
  computed from what.</p>
  <div style="display:flex;gap:12px;align-items:center">{cta}
    <a class="btn" href="/docs">API reference</a></div>

  <div class="grid" style="margin-top:72px">
    <div class="card">
      <h2>Belief revision</h2>
      <p class="muted" style="margin:0">A superseded memory is not deleted. It stays
      reachable, marked, and pointed at whatever replaced it — so "what did we think in
      March" is answerable.</p>
    </div>
    <div class="card">
      <h2>Edges between memories</h2>
      <p class="muted" style="margin:0">Not a document-to-chunk tree. Supersedes,
      contradicts, derived-from: claims about how two memories relate.</p>
    </div>
    <div class="card">
      <h2>Erasure that propagates</h2>
      <p class="muted" style="margin:0">Delete a source and everything computed from it
      goes stale, because a derivation must never outlive its evidence.</p>
    </div>
  </div>
</div>
""",
        nav='<a href="/docs">Docs</a><a href="/orgs">Dashboard</a>',
    )


def organizations(
    user_email: str, orgs: list[dict[str, object]], *, cap: int, error: str = ""
) -> str:
    """The org list: the first thing someone sees after signing in."""
    if orgs:
        cards = "".join(
            f"""<a class="card" href="/orgs/{escape(str(o["id"]))}" style="text-decoration:none;display:block">
                  <div style="display:flex;justify-content:space-between;align-items:start">
                    <strong style="font-size:17px">{escape(str(o["name"]))}</strong>
                    <span class="pill">{escape(str(o["role"]))}</span></div>
                  <div class="muted" style="margin-top:10px;font-size:13px">
                    {o["spaces"]} space{"" if o["spaces"] == 1 else "s"}</div>
                  <div class="mono muted" style="margin-top:4px;font-size:11px">{escape(str(o["id"]))}</div>
                </a>"""
            for o in orgs
        )
    else:
        cards = '<p class="muted">No organizations yet. Create one to get started.</p>'

    at_cap = len(orgs) >= cap
    form = (
        f'<p class="muted">You are at the limit of {cap} organizations.</p>'
        if at_cap
        else """
        <form method="post" action="/orgs" style="display:flex;gap:10px;max-width:460px">
          <input type="text" name="name" placeholder="Organization name" required maxlength="200">
          <button class="btn pri" type="submit">Create</button>
        </form>"""
    )
    err = f'<div class="err">{escape(error)}</div>' if error else ""

    return _shell(
        "Organizations — mapi",
        f"""
<div class="wrap" style="padding-top:48px">
  {err}
  <h2>Organizations</h2>
  <div class="grid">{cards}</div>
  <div style="margin-top:40px">
    <h2>New organization</h2>
    {form}
  </div>
</div>
""",
        nav=f'<span class="muted mono" style="font-size:13px">{escape(user_email)}</span>'
        '<a href="/auth/logout">Sign out</a>',
    )


__all__ = ["STYLE", "landing", "organizations"]
