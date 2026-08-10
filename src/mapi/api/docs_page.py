# ruff: noqa: E501
# One embedded HTML document. Wrapping markup mid-attribute to satisfy a
# line-length rule makes it harder to read and easier to break.
"""The documentation. Distinct from the reference, on purpose.

A reference answers "what are the parameters of this endpoint" and is
generated from the schema, so it is always right and never explains
anything. Documentation answers "what is this for, and what will happen if I
do it" -- which the schema cannot express and which is the part people
actually need. Both exist here; `/reference` is the former, this is the
latter.

Written to a rule: describe BEHAVIOUR and CONTRACTS, never mechanism. What a
caller can rely on is public. How retrieval ranks, what the consolidation
thresholds are, how claims are extracted -- those are the product, and this
page is not where they get published. Every statement below is something the
API actually guarantees, not a description of the implementation that
happens to provide it.
"""

from __future__ import annotations

from .pages import STYLE

_EXTRA = """
  .doc { display:flex; gap:56px; align-items:flex-start; max-width:1140px;
         margin:0 auto; padding:0 28px }
  .toc { width:210px; flex:none; position:sticky; top:32px; padding-top:48px }
  .toc a { display:block; padding:5px 0; color:var(--dim); text-decoration:none;
           font-size:13.5px; border-left:2px solid transparent; padding-left:12px }
  .toc a:hover { color:var(--text) }
  .toc a.sub { padding-left:24px; font-size:13px }
  .toc h3 { font-size:11px; letter-spacing:.12em; text-transform:uppercase;
            color:var(--dim); margin:22px 0 6px 12px; font-weight:600 }
  .body { flex:1; min-width:0; padding:48px 0 96px; max-width:720px }
  .body h2 { font-size:27px; letter-spacing:-.01em; text-transform:none; color:var(--text);
             margin:56px 0 14px; font-weight:640; scroll-margin-top:24px }
  .body h2:first-child { margin-top:0 }
  .body h3 { font-size:17px; margin:32px 0 8px; font-weight:620 }
  .body p { margin:0 0 16px }
  .body ul { margin:0 0 16px; padding-left:22px }
  .body li { margin:6px 0 }
  pre { background:#1b1b19; color:#f2f0ec; border-radius:10px; padding:16px 18px;
        overflow-x:auto; font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;
        margin:0 0 20px }
  pre .c { color:#8b9a8f }
  code { font:13px ui-monospace,SFMono-Regular,Menlo,monospace;
         background:var(--hair); padding:1.5px 6px; border-radius:5px }
  pre code { background:none; padding:0 }
  table { border-collapse:collapse; width:100%; margin:0 0 22px; font-size:14px }
  th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line);
           vertical-align:top }
  th { font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--dim) }
  .note { border-left:3px solid var(--text); background:var(--hair);
          padding:13px 16px; border-radius:0 8px 8px 0; margin:0 0 22px }
  .note strong { display:block; margin-bottom:3px }
  @media (max-width:900px) { .toc { display:none } .doc { padding:0 20px } }
"""

_TOC = """
<nav class="toc">
  <h3>Start</h3>
  <a href="#what">What it is</a>
  <a href="#quickstart">Quickstart</a>
  <a href="#how">How it works</a>
  <a href="#numbers">Numbers</a>
  <h3>Concepts</h3>
  <a href="#spaces">Spaces</a>
  <a href="#memories">Memories</a>
  <a href="#time">Event time</a>
  <a href="#currency">Currency</a>
  <a href="#relations">Relations</a>
  <h3>Guides</h3>
  <a href="#writing">Writing well</a>
  <a href="#searching">Searching</a>
  <a href="#context">Asking why</a>
  <a href="#facts">Keeping facts current</a>
  <a href="#erasure">Erasure</a>
  <h3>Operating</h3>
  <a href="#errors">Errors</a>
  <a href="#keys">Keys and spaces</a>
  <a href="#reference">Reference</a>
</nav>
"""

_BODY = """
<div class="body">

<h2 id="what">What it is</h2>
<p>mapi stores what your agent learns about a person and gives it back when it
is relevant. It is not a vector database with a nicer client: the difference is
that it tracks <em>currency</em> and <em>provenance</em>, so an agent can tell
what is still true and say why it believes it.</p>
<p>Three things follow from that, and they are the reason to use this over a
plain index:</p>
<ul>
  <li>A fact that has been overtaken stops being returned, without being deleted.</li>
  <li>Any memory can report what replaced it, what it replaced, what it was computed from, and what disagrees with it.</li>
  <li>Deleting a source marks anything derived from it stale, so a conclusion cannot outlive its evidence.</li>
</ul>

<h2 id="quickstart">Quickstart</h2>
<pre><code>pip install mapi-sdk</code></pre>
<pre><code>from mapi_sdk import Mapi

client = Mapi(api_key="sm_...")          <span class="c"># or set MAPI_API_KEY</span>
client.spaces.get_or_create("ada")

client.memories.add("Prefers window seats", space="ada", tags=["travel"])

for hit in client.search.execute("seating preference", space="ada"):
    print(hit.score, hit.content)</code></pre>
<p>Async is the same surface, awaited:</p>
<pre><code>from mapi_sdk import AsyncMapi

async with AsyncMapi() as client:
    await client.memories.add("Allergic to shellfish", space="ada")
    hits = await client.search.execute("allergies", space="ada")</code></pre>
<p>Or without the SDK:</p>
<pre><code>curl -X POST https://memory-api-7b178bde9ecc.herokuapp.com/v1/spaces/$SPACE/memories \\
  -H "Authorization: Bearer $MAPI_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Prefers window seats", "tags": ["travel"]}'</code></pre>

<h2 id="how">How it works</h2>
<p>Three stages, described at the level you need to use it well. The specifics
of each — how candidates are ranked, what makes something count as a
replacement, how facts are pulled out of text — are the product, and are not
documented here.</p>

<h3>On write</h3>
<p>Text is normalised, split where it is too long to index as one unit, and
embedded. Identical content is folded into what is already stored rather than
duplicated. If you ask for it, the write also compares against what the space
already holds, and records what this new memory <em>replaces</em> or
<em>disagrees with</em>.</p>

<h3>On read</h3>
<p>A query runs against both meaning and wording, and the two rankings are
combined — semantic search alone misses exact strings like order numbers and
surnames, and keyword search alone misses paraphrase. Recency is weighed in,
results that a returned memory has superseded are removed, and each result can
report why it is there.</p>

<h3>Over time</h3>
<p>Memories accumulate typed relations, which is what makes the store answer
questions an index cannot: what is current, what changed, what disagrees, and
what a conclusion rests on. Deleting evidence marks the conclusions drawn from
it stale rather than leaving them standing.</p>

<h2 id="numbers">Numbers</h2>
<p>Measured on a public benchmark, run end to end — retrieve,
answer, and grade — not retrieval-only, because retrieval quality that never
becomes a correct answer is not worth reporting.</p>
<table>
  <tr><th>benchmark</th><th>what it tests</th><th>questions</th><th>accuracy</th></tr>
  <tr><td>LongMemEval-S</td><td>recall across ~50 sessions of chat history per question</td><td>470</td><td><strong>82.6%</strong></td></tr>
</table>
<p>Reported per capability, never as one blended score. A system can be
excellent at recall and dangerous at knowing when a fact went stale, and one
number hides exactly that.</p>

<h3>LongMemEval-S — 470 questions</h3>
<table>
  <tr><th>capability</th><th>n</th><th>accuracy</th></tr>
  <tr><td>single-session assistant</td><td>56</td><td>94.6%</td></tr>
  <tr><td>single-session user</td><td>70</td><td>94.3%</td></tr>
  <tr><td>knowledge update</td><td>78</td><td>87.2%</td></tr>
  <tr><td>temporal reasoning</td><td>133</td><td>79.7%</td></tr>
  <tr><td>multi-session</td><td>133</td><td>72.9%</td></tr>
  <tr><td>single-session preference</td><td>30</td><td>66.7%</td></tr>
</table>

<h3>Finding the evidence</h3>
<p>Accuracy above is the whole pipeline. These are the retrieval stage alone,
scored against the evidence each benchmark labels — no model, no judge, so
none of it carries grading variance:</p>
<table>
  <tr><th>what it measures</th><th>result</th></tr>
  <tr><td><strong>Every piece of evidence delivered.</strong> The strict one: a question needing four sources counts only if all four arrive.</td><td>96.8%</td></tr>
  <tr><td><strong>Some evidence delivered.</strong> At least one correct source in the window.</td><td>99.4%</td></tr>
  <tr><td><strong>Correct evidence ranked first.</strong> How near the top the first right answer lands.</td><td>0.939</td></tr>
  <tr><td><strong>Ranking quality.</strong> Credit for correct sources weighted by position.</td><td>0.937</td></tr>
</table>
<p>Complete evidence reaches the answer stage for 96.8% of
questions, so most remaining errors are answering errors rather than recall
errors — the honest reading, and the reason the weakest capability rows above
are the ones being worked on.</p>

<h3>Explainability</h3>
<p>Every result can report why it is there. Ask for it with
<code>explain=True</code> and each hit carries the reasons it survived to the
top — which stage promoted it, and what moved its position. That is the
difference between a ranking you can debug and a number you have to trust.</p>
<pre><code>result = client.search.execute("allergies", space="ada", explain=True)
for hit in result:
    print(hit.score, hit.explain)</code></pre>
<div class="note">
  <strong>How these were graded.</strong>
  With LongMemEval\'s own published judge prompts, and by a model from a
  different family than the one answering — a model grading its own output
  scores itself generously, and the size of that effect is published alongside
  the results rather than quietly absorbed into them.
</div>

<h2 id="spaces">Spaces</h2>
<p>A space is one person's memory. Retrieval never crosses a space boundary, so
one user's memories cannot appear in another's results — that is a property of
the storage layer, not a filter you have to remember to apply.</p>
<p>Most applications create one space per end user. Spaces are cheap; a space
per user is the intended shape, not an abuse of it.</p>
<pre><code>client.spaces.get_or_create("user-8134")     <span class="c"># slug you choose</span>
client.spaces.list()</code></pre>
<p>Every call takes a space slug or its id. Slugs are resolved once per process
and cached, so using the readable name costs nothing per call.</p>

<h2 id="memories">Memories</h2>
<p>A memory is a piece of text with a time, and optionally tags, metadata and a
source. Two kinds exist:</p>
<table>
  <tr><th>kind</th><th>what it is</th><th>answers</th></tr>
  <tr><td><code>episodic</code></td><td>something that happened, stored as it was said</td><td>what happened, and in what order</td></tr>
  <tr><td><code>derived</code></td><td>a standing fact, computed from episodes</td><td>what is true now</td></tr>
</table>
<p>You write episodes. Derived memories appear when you ask for them, and always
carry links back to the episodes they came from — which is what makes erasure
propagate.</p>

<h2 id="time">Event time</h2>
<div class="note">
  <strong>occurred_at is when it happened, not when you wrote it.</strong>
  Leaving it out on a backfill makes a year of history look like it all
  happened today, and every question about order is wrong afterwards.
</div>
<pre><code>from datetime import UTC, datetime

client.memories.add(
    "Ran the charity 5K in 27:12",
    space="ada",
    occurred_at=datetime(2023, 5, 20, tzinfo=UTC),
)</code></pre>
<p>Event time is separate from the time the system learned something. Both are
kept, which is why a memory can be read as of a past moment.</p>

<h2 id="currency">Currency</h2>
<p>When a newer memory replaces an older one, the older is marked
<code>superseded</code> and drops out of search results. It is not deleted: it
stays readable, and it points at whatever replaced it.</p>
<p>This matters more than it sounds. An index that returns both a person's old
address and their new one has technically retrieved correctly and practically
misinformed the agent — and the agent has no way to tell which is which.</p>
<pre><code>hits = client.search.execute("address", space="ada")
<span class="c"># returns the current address only</span>

hits = client.search.execute("address", space="ada", include_superseded=True)
<span class="c"># returns the history too, each marked with its status</span></code></pre>

<h2 id="relations">Relations</h2>
<p>Memories relate to each other in four ways. These are claims about the
memories, not links between documents:</p>
<table>
  <tr><th>relation</th><th>meaning</th></tr>
  <tr><td><code>supersedes</code></td><td>this replaced that</td></tr>
  <tr><td><code>contradicts</code></td><td>these cannot both be true</td></tr>
  <tr><td><code>derived_from</code></td><td>this was computed from that</td></tr>
  <tr><td><code>references</code></td><td>this mentions that</td></tr>
</table>
<p>Contradictions are surfaced, never resolved. Both memories stay active and
both are returned, because either may be the true one — an agent told "these
two disagree" can ask the user, where one handed a silent winner cannot.</p>

<h2 id="writing">Writing well</h2>
<p>Three options change what a write does. All are off by default, because each
costs something and none is right for every application.</p>
<table>
  <tr><th>option</th><th>what it does</th><th>when</th></tr>
  <tr><td><code>detect_conflicts</code></td><td>flags memories this one disagrees with</td><td>when contradictions matter more than write latency</td></tr>
  <tr><td><code>auto_supersede</code></td><td>marks older memories this one replaces</td><td>when the same fact is restated over time</td></tr>
  <tr><td><code>extract</code></td><td>also stores the standalone facts this text states</td><td>when you want a graph of claims, not just documents</td></tr>
</table>
<div class="note">
  <strong>extract is a choice, not an upgrade.</strong>
  It stores atomic claims alongside the original — never instead of it — which
  makes relationships between facts visible. It also spends a model call per
  write and suits questions about what is <em>true</em> better than questions
  about what <em>happened</em>. Turn it on when the graph is the point.
</div>
<p>Writes are deduplicated by default, so replaying the same content is safe and
will not create a second copy.</p>

<h2 id="searching">Searching</h2>
<pre><code>result = client.search.execute(
    "what does she eat",
    space="ada",
    limit=10,
    tags=["health"],        <span class="c"># narrow to tagged memories</span>
    min_score=0.1,          <span class="c"># drop weak matches entirely</span>
    explain=True,           <span class="c"># why each result is here</span>
)

for hit in result:
    print(hit.score, hit.content)

result.conflicts    <span class="c"># pairs of returned ids that disagree</span></code></pre>
<div class="note">
  <strong>Use min_score when "nothing matched" is a real answer.</strong>
  Search returns the closest memories it has. Without a floor, a question about
  something never mentioned still returns the ten least-unrelated memories, and
  an agent cannot tell that from a real answer.
</div>
<p><code>score</code> is comparable within one response and not across them.
Do not store it or threshold on it between queries.</p>

<h2 id="context">Asking why</h2>
<p>The call that separates a memory store from an index:</p>
<pre><code>ctx = client.memories.context(memory_id, space="ada")

ctx.is_current      <span class="c"># False if something replaced it</span>
ctx.current_head    <span class="c"># what replaced it</span>
ctx.replaced        <span class="c"># what it replaced</span>
ctx.derived_from    <span class="c"># the episodes it was computed from</span>
ctx.derivatives     <span class="c"># what was computed from it</span>
ctx.contradicts     <span class="c"># what disagrees with it</span></code></pre>
<p>One request resolves the whole neighbourhood. This is what an agent should
show a user who asks "why do you think that", and what you should log when an
answer was wrong.</p>

<h2 id="facts">Keeping facts current</h2>
<p>When you know a memory replaces another, say so:</p>
<pre><code>client.memories.relate(
    new_id, space="ada", target_id=old_id, relation="supersedes",
)</code></pre>
<p>The older memory becomes <code>superseded</code> immediately and leaves
search results. Its content, its history and the link between them all remain
readable.</p>
<p>If you would rather the system notice on its own, write with
<code>auto_supersede=True</code>. It is deliberately conservative: a missed
supersession leaves both memories visible and rankable, where a wrong one hides
a true memory from every answer. The two mistakes are not equally cheap.</p>

<h2 id="erasure">Erasure</h2>
<p>Two operations, for two different obligations:</p>
<table>
  <tr><th>call</th><th>effect</th></tr>
  <tr><td><code>memories.delete</code></td><td>removes it from results, keeps the audit trail</td></tr>
  <tr><td><code>memories.erase</code></td><td>destroys the content everywhere it can be reached, including history</td></tr>
</table>
<p><code>erase</code> returns an attestation: what was destroyed, how much, and
which memories had claimed derivation from it. Those are marked stale, because a
conclusion must not outlive the evidence it was drawn from — a summary of
deleted data is still that data.</p>
<pre><code>report = client.memories.erase(memory_id, space="ada")
report["stale"]      <span class="c"># what stopped being trustworthy</span></code></pre>

<h2 id="errors">Errors</h2>
<p>Every error is JSON with a stable <code>code</code>, a human
<code>detail</code>, and a <code>request_id</code>. Quote the request id when
reporting a problem; it is how a specific call gets found in the logs.</p>
<table>
  <tr><th>status</th><th>code</th><th>means</th></tr>
  <tr><td>401</td><td><code>unauthorized</code></td><td>key missing, malformed or revoked</td></tr>
  <tr><td>403</td><td><code>forbidden</code></td><td>key is valid but lacks the scope</td></tr>
  <tr><td>404</td><td><code>not_found</code></td><td>no such resource — also what another organization's resources look like</td></tr>
  <tr><td>409</td><td><code>conflict</code></td><td>collides with something stored</td></tr>
  <tr><td>422</td><td><code>validation_error</code></td><td>rejected before anything was written</td></tr>
  <tr><td>429</td><td><code>rate_limited</code></td><td>too many requests; honour <code>Retry-After</code></td></tr>
</table>
<p>The SDK maps these to typed exceptions and retries 429 and gateway errors
with backoff. It does not retry 500: a request that made the server fail will
usually fail again, and retrying hides it.</p>

<h2 id="keys">Keys and spaces</h2>
<p>An API key authorises requests for one organization. It does not identify a
person — two people on a team legitimately share a key, so a key can never tell
you who acted. Sign-in identifies people; keys authorise programs.</p>
<p>Create keys in the dashboard. A key is shown once, at creation, and stored
hashed — if it is lost it cannot be recovered, only replaced.</p>

<h2 id="reference">Reference</h2>
<p>Every endpoint, parameter and response shape is in the
<a href="/reference">API reference</a>, generated from the service itself so it
cannot drift from what the server actually accepts.</p>
<p>This page explains what things mean. The reference tells you exactly what to
send.</p>

</div>
"""

PAGE = (
    "<!doctype html>\n"
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    "<title>Documentation — mapi</title>\n"
    '<link rel="icon" href="/static/mapi-logo.png">\n'
    f"<style>{STYLE}{_EXTRA}</style>\n"
    '<header><div class="wrap">'
    '<a href="/"><img class="logo" src="/static/mapi-logo.png" alt="mapi"></a>'
    '<nav><a href="/docs">Docs</a><a href="/reference">Reference</a>'
    '<a href="/orgs">Dashboard</a></nav>'
    "</div></header>\n"
    f'<div class="doc">{_TOC}{_BODY}</div>\n'
)

__all__ = ["PAGE"]
