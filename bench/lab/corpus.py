"""Ten large sessions carrying every kind of data the system claims to handle.

Deliberately built so each stage has something to bite on:

  * multi-fact paragraphs        -> extraction must decompose them
  * a fact restated later        -> supersession must fire, once, correctly
  * a genuine disagreement       -> contradiction must surface both sides
  * near-identical records       -> dedup must NOT collapse distinct ones
  * quantities and currencies    -> counting and comparison
  * dates across a year          -> temporal windows and ordering
  * stated preferences           -> the advice path
  * shared entities              -> association edges
"""

from datetime import UTC, datetime


def when(month: int, day: int) -> datetime:
    return datetime(2026, month, day, 10, 0, tzinfo=UTC)


SESSIONS: list[tuple[str, datetime, list[str], str]] = [
    (
        "s01",
        when(1, 12),
        ["infra", "vendors"],
        """
Kicked off the infrastructure review this morning. Our primary database is
Postgres 16 running on Cloud SQL in us-central1, with pgvector 0.8.1 for the
embedding columns. Redis 7 backs the rate limiter across processes. The API
runs on Heroku with two web dynos on the heroku-24 stack, Python 3.13.

Spend so far this quarter: Cloud SQL is 340 dollars a month, Redis is 45
dollars a month, and Heroku dynos come to 100 dollars a month. Vertex AI
embeddings have been about 190 dollars a month at current volume.

I prefer managed services over self-hosting anything stateful. Learned that
the hard way running our own Postgres in 2024.

Spent the first hour reading dashboards before touching anything. Infrastructure reviews go badly when you start from the bill instead of the architecture, because you optimise the wrong line item. So I drew the request path first: client hits the dyno, dyno talks to Postgres over the proxy, embeddings go to Vertex, rate limit checks hit Redis. Four hops, three of them stateful. Aaditi asked whether we should consolidate onto fewer providers. Provider count is not the cost driver at our size; the driver is whether anything is oversized for its load. Also went through alerting, which is thinner than I would like: we alert on 5xx rate and dyno restarts and nothing else. No alert on embedding latency, none on the rate limiter degrading.
""",
    ),
    (
        "s02",
        when(1, 28),
        ["invoices", "vendors"],
        """
Processed the January vendor invoices. Invoice INV-4471 from Cloudflare for
220 dollars, dated 14 January. Invoice INV-4472 from Datadog for 890 dollars,
dated 16 January. Invoice INV-4473 from Twilio for 310 dollars, dated 21
January.

All three are paid. INV-4472 needed approval from Aaditi because it crossed
the 500 dollar threshold.

Invoice week, slower than it should be because three vendors send PDFs with the amount in a different place each month. The approval threshold caused friction: anything over 500 dollars needs a second signature, which is correct policy and also means a routine monitoring bill sits in a queue for two days. We discussed raising it to 1000 for recurring vendors where the amount is predictable. Did not change it. I reconciled against the bank feed rather than the vendor statements, because last quarter a vendor double-charged us and their own statement showed one line.
""",
    ),
    (
        "s03",
        when(2, 9),
        ["hiring", "people"],
        """
Interviewed three candidates for the founding engineer role this week. Priya
Raman, strongest on systems, currently at Stripe. Marcus Bell, strongest on
product sense, currently at a seed-stage company. Dana Okoro, strongest on
infrastructure, currently at Cloudflare.

We are moving Priya and Dana to the final round. Marcus was a pass on depth.

I care much more about shipping velocity than credentials. Someone who has
shipped a product end to end beats someone with a better resume.

Same format for all three: a systems discussion, a live debugging session on a real trace from our own logs, then a conversation about something they shipped and owned end to end. The debugging session separates people. It is an actual production trace with an actual bug, and I watch how they narrow rather than whether they find it. Priya narrowed by ruling out layers. Marcus guessed and checked, which works until the system is big. Dana asked for the deploy timeline first, which is what I would have done. We discussed compensation ranges up front, so nobody reached the final round and discovered the number was wrong for them.
""",
    ),
    (
        "s04",
        when(3, 3),
        ["infra", "migration"],
        """
Migrated off Heroku Postgres onto Cloud SQL today. The proxy approach worked:
the app connects over localhost and authenticates with IAM, so the database
keeps no authorized networks at all.

Migration took 40 minutes end to end. Zero downtime because we ran both in
parallel for an hour and cut over after verifying row counts matched.

Wrote the runbook the night before and stuck to it, which is the only reason this went cleanly. The proxy decision is worth recording because I went back and forth. The alternative was authorizing dyno IP ranges, and Heroku dynos have no stable outbound addresses, so that meant opening the database to the whole internet and relying on TLS and the password. A managed add-on does exactly that, so it is not unreasonable, but it is strictly weaker than requiring an IAM principal. Verified by comparing row counts per table, then spot-checking the vector columns, because an encoding difference would not show up in a count.
""",
    ),
    (
        "s05",
        when(3, 21),
        ["research", "venues"],
        """
The Persistence of Vision draft is at 9 pages. Primary target venue is PoPETs
2027 Issue 2, deadline 31 August. ARTMAN at ACSAC 2026 is a parallel
non-archival submission. IEEE TIFS is the highest eventual-acceptance venue.

The attack recovers name at 1.00, ID at 0.85 and date of birth at 0.82 top-1
from a 200-item lineup. Chance is 0.005. The Cataract defense holds the
strongest adaptive attack to 10 percent success while keeping 87.5 percent of
retrieval quality, at 0.64 milliseconds per page.

The paper is structurally fine but related work is thin and the threat model needs stating earlier. The result I keep returning to is the single-vector control, because it makes the finding a finding rather than an observation. If the attack worked equally well against a single-vector retriever the story would just be retrievers leak, which is known. It does not: the control scores 0.08, 0.03 and 0.02 on the same three fields. That gap is the contribution, and it means the leak is a property of the multi-vector representation. Still undecided how much of the defense goes in the body versus an appendix.
""",
    ),
    (
        "s06",
        when(4, 14),
        ["infra", "vendors"],
        """
Reviewed spend again. Cloud SQL has gone up to 410 dollars a month after we
enabled point-in-time recovery. Redis is unchanged at 45. Heroku is now 175
dollars a month because we added a second dyno and a Redis add-on we ended up
not needing.

Cancelled the unused Heroku Redis add-on. That saves 15 dollars a month.

Second spend review, six weeks after the first, and the useful part was comparing against my own notes rather than a budget. Every increase came from a deliberate choice we forgot we made. Point-in-time recovery I would choose again. The second dyno I would choose again. The Redis add-on was added during the migration when we thought we might need a second cache, and then we did not, and nobody went back to it. The lesson is not watch the bill, it is put an expiry on anything added during an incident or a migration, because those are the changes nobody revisits.
""",
    ),
    (
        "s07",
        when(5, 6),
        ["hiring", "people"],
        """
Priya accepted the founding engineer offer and starts 1 June. Dana declined,
she took a staff role at Cloudflare instead.

Aaditi and I split the onboarding: I take the codebase walkthrough, she takes
customers and the GST domain.

Two offers out, one accepted, and the declined one for a reason I respect rather than a counteroffer we could have matched. Dana wanted the scope of a staff role at a company with an existing platform, which we cannot offer and should not pretend to. I would rather lose that honestly than sell someone a job that does not exist. The onboarding split matters more than people think: the failure mode is both founders explaining the same thing differently in week one, so I own everything in the repository, she owns everything a customer touches, and neither improvises on the other's half.
""",
    ),
    (
        "s08",
        when(6, 2),
        ["travel", "preferences"],
        """
Booked the Lisbon trip for the PoPETs deadline crunch. Flying out 20 August,
back 3 September. I prefer aisle seats on flights over four hours, and I will
not take a red-eye if there is any alternative.

Hotel is 180 euros a night for 14 nights. Last year the same trip was 210
euros a night, so this is cheaper per night.

Allergic to shellfish, which matters in Lisbon specifically.

Mostly logistics, but worth writing down because I will forget by August. The flight rules are less preferences than things that cost me a working day when I get them wrong. An aisle seat on anything over four hours means I can work; a middle seat means I cannot. A red-eye means the first day is gone, which on a deadline trip is the whole point of going. On the hotel I deliberately booked further from the venue than last year in exchange for the lower nightly rate, on the theory that a fifteen minute walk is fine and 30 euros a night across fourteen nights is not nothing.
""",
    ),
    (
        "s09",
        when(7, 18),
        ["infra", "incident"],
        """
Incident this morning. The rate limiter stopped enforcing limits for roughly
25 minutes because Redis was unreachable and the limiter fell back to
per-process counters across two dynos.

Root cause was a Redis maintenance window we had not seen. Fix was to alert on
the fallback rather than let it degrade silently.

We are NOT moving off Redis. The fallback behaviour was correct, the alerting
was missing.

Timeline: alerts on elevated request volume at 09:14, no alert on the limiter itself because we have none. Noticed manually at 09:31 when a single API key showed a request count that should have been impossible under the configured limit. Redis connectivity restored by the provider at 09:39. The uncomfortable part is that nothing was broken. The limiter is designed to fall back to per-process counters when Redis is unreachable and it did exactly that: two dynos, per-process counters, effective limit twice what it should be. That is documented and correct, because failing every request because a cache is down would be worse. The bug is that a degraded mode with no alert is indistinguishable from a healthy one.
""",
    ),
    (
        "s10",
        when(8, 25),
        ["research", "venues"],
        """
Submitted The Persistence of Vision to PoPETs 2027 Issue 2 on 25 August, six
days before the deadline. Final version is 14 pages.

Also decided to skip ARTMAN at ACSAC 2026 entirely rather than submit a
parallel non-archival version. Not worth the reviewer overlap.

Six days early, the earliest I have ever been on anything. Final page count came in at 14 after the threat model moved forward and the defense got its own subsection in the body rather than an appendix. I put deployability in the body, reasoning that a reviewer who thinks the defense is impractical will reject regardless of how good the attack is. The ARTMAN decision took longer than it should. The argument for submitting was visibility in a community that cares about exactly this. The argument against, which won, is that the reviewer pools overlap enough that a parallel non-archival submission buys the same feedback twice with a small risk of looking like you are shopping the result.
""",
    ),
]

# Every question shape, plus the pairs that only work if consolidation worked.
QUESTIONS: list[tuple[str, str, str]] = [
    ("direct", "What database do we use?", "Postgres 16"),
    ("direct", "Who accepted the founding engineer offer?", "Priya"),
    ("count", "How many vendor invoices did I process in January?", "3"),
    ("count", "How many candidates did I interview for the founding engineer role?", "3"),
    ("order", "Which invoice was dated earliest in January?", "INV-4471"),
    ("order", "What was the last thing I submitted to a venue?", "The Persistence of Vision"),
    ("date_arith", "How many days before the PoPETs deadline did I submit?", "6"),
    ("compare", "Is the Lisbon hotel cheaper per night than last year?", "yes, 180 vs 210"),
    ("compare", "Which costs more per month, Cloud SQL or Heroku?", "Cloud SQL"),
    ("advice", "Can you recommend how I should book my next long flight?", "aisle, no red-eye"),
    ("list_all", "What is our entire infrastructure?", "postgres redis heroku python vertex"),
    ("revision", "How much does Cloud SQL cost per month?", "410"),
    ("revision", "Are we moving off Redis?", "no"),
    ("temporal", "What did I do in March?", "migration"),
    ("identity", "What is invoice INV-4472 for?", "Datadog 890"),
]
