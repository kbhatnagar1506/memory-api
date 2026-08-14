"""Every capability the memory system claims, against a live model, end to end.

Not LongMemEval and not a retrieval ablation. This ingests a corpus built so
that each capability has a question only that capability can answer, then asks
through the SHIPPING synthesis path -- `service.search` into
`synthesis.chat.answer` -- so what is measured is the product, including the
classifier that routes advice away from the decline contract and the prompts
that were changed this week.

The retrieval-only numbers elsewhere in bench/ carry no judge variance and are
exactly reproducible, which is their value and also their limit: a system can
retrieve perfectly and still answer badly, and the whole argument of the lab
work has been that those two failures need separating. This file deliberately
puts the model back in.

WHAT IS COVERED -- 54 capabilities, because "remembering" is not one skill:

    LOOKUP        direct, order, temporal, multi, list_all, paraphrase
    ARITHMETIC    count, sum, max, min, average, group_by, ratio, delta,
                  date_arith, duration, compare, unit, counterfactual
    CHANGE        supersede (value moved), reversal (the thing was undone),
                  retraction (the first statement was wrong), refinement
                  (vague became exact), partial_update (one field moved and
                  the rest did not), contradict (two sources disagree)
    TIME          as_of (state at a past moment), relative_date, ongoing,
                  same_day, planned, recurring, interval
    SHAPE         negation, attribution (X SAID y, which is not y), conditional,
                  causal, ambiguity, partial
    ENTITY        coreference, role_change, multi_hop, distractor, near_duplicate
    HARDNESS      numeric (18 vs 180), unit-suffixed figures
    PREFERENCE    explicit / implicit / negative / updated / composed
    HONESTY       abstain, negative_existence, unverified, tenancy, citations

The change family is the point. Supersession is the tidy case everyone builds
for; a reversal answered with the original value leaves the user holding a
thing they returned, and a retraction answered with the original quotes a
number that was never true. They fail differently and need testing separately.

CITATIONS ARE CHECKED, not counted. A reply can carry [3] against a claim
memory 3 does not support and score as fully cited, which is the audit trail
failing quietly. `EVIDENCE` maps a question to a substring of the memory that
actually supports it, and the cited ids are checked against it.

RESULT: 60/67, then 62 after the findings below. Roughly a third of every
failure round was this file's own bug -- a scorer that could not read "180cm",
a leak detector that flagged a coincidence of digits, a corpus that wrote "day
rate at 60 pounds an hour" and then expected 240 for four days. Printing the
replies rather than counting them is what makes that separable.

TWO PRODUCT FIXES CAME OUT OF IT, both in `synthesis/chat.CHAT_PROMPT`:
arithmetic over stored figures is answering FROM memory rather than filling a
gap, and a claim must be checked against the items shown beneath it. The second
was the most common failure by far -- "three projects in Q2" over a list
containing a March one, "two of four paid late" over a list of one.

WHAT REMAINS, and it is not more prompt:

  * AGGREGATION NEEDS A SET, AND RANKING DOES NOT RETURN SETS. For "average
    invoice value" the four invoice rows rank 9, 15, 19 and 21. At k=20 three
    arrive and the model declines rather than dividing by the wrong
    denominator; at k=30 all four arrive and it answers correctly. No fixed k
    is right, because the members of a set are not contiguous in a similarity
    ranking -- this is `full_recall@k` versus `hit@k` showing up live, and the
    fix is set-completion retrieval, not a bigger number.
  * DATE COMPARISON. "Had I bought the Bambu when I shipped the Bramwell
    housing?" answers "Yes" and then prints shipped 24 March, bought 15 April.
    The check rule fixed the counting version of this and not the ordering one.
  * OPEN-WORLD CATEGORIES. "Have I ever worked with an automotive client" is
    answered with Bramwell CYCLES.

Note the aggregation trade: before the check rule, `sum` sometimes computed a
right answer and sometimes averaged three of four rows as if it were four. Now
it declines when the set is short. The score went down and the behaviour got
better, which is the correct direction for a memory system and worth stating
plainly rather than hiding in a total.

    python -m bench.lab.capability_live
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from mapi.config import Settings
from mapi.domain.embeddings.gemini import GeminiEmbedder
from mapi.domain.models import Organization, Space
from mapi.domain.retrieval.pipeline import SearchRequest
from mapi.domain.retrieval.rerank import HeuristicReranker
from mapi.domain.synthesis.chat import answer as chat_answer
from mapi.service import MemoryService
from mapi.store.memory import InMemoryStore

from .scoring import Expected, score
from .stats import wilson

MODEL = "gemini-2.5-pro"
PROJECT = "patchguard-reakon"
MAX_INFLIGHT = 8
ASKED_AT = datetime(2026, 8, 14, tzinfo=UTC)


def _d(y: int, m: int, day: int) -> datetime:
    return datetime(y, m, day, 12, 0, tzinfo=UTC)


#: (occurred_at, content, metadata). One persona, one working year, built so
#: that every capability below has evidence and every trap has bait.
MEMORIES: tuple[tuple[datetime, str, dict], ...] = (
    (
        _d(2026, 1, 12),
        "I set my freelance day rate at 500 pounds when I went independent.",
        {},
    ),
    (
        _d(2026, 1, 20),
        "Signed the first contract with Halden Foods for a packaging refresh.",
        {},
    ),
    (_d(2026, 2, 3), "Bought a Prusa MK4 printer for the workshop, 899 pounds.", {}),
    (
        _d(2026, 2, 17),
        "Shipped the Halden Foods packaging refresh. They paid within a week.",
        {},
    ),
    (
        _d(2026, 3, 4),
        "Started the Bramwell Cycles project, a rear light housing in aluminium.",
        {},
    ),
    (_d(2026, 3, 9), "Flew to Milan for the design fair and came back with three leads.", {}),
    (
        _d(2026, 3, 24),
        "Shipped the Bramwell Cycles light housing. Tooling signed off on the first pass.",
        {},
    ),
    (
        _d(2026, 4, 2),
        "Turned down a vape packaging job. I don't take tobacco or vape work, ever.",
        {},
    ),
    (
        _d(2026, 4, 15),
        "Bought a Bambu X1C printer, 1299 pounds, because the Prusa could not keep up.",
        {},
    ),
    (
        _d(2026, 4, 28),
        "Shipped the Okonjo Studio retail fixture. Third project done this year.",
        {},
    ),
    (
        _d(2026, 5, 6),
        "Raised my day rate from 500 to 700 pounds. Nobody pushed back.",
        {},
    ),
    (
        _d(2026, 5, 19),
        "Quoted studio A at 1400 a month and studio B at 1150 a month for the Peckham space.",
        {},
    ),
    (
        _d(2026, 6, 2),
        "Signed with Vireo Health for a device enclosure. Fourth project of the year.",
        {},
    ),
    (
        _d(2026, 6, 11),
        "I work in metric only. An imperial drawing gets sent back, no exceptions.",
        {},
    ),
    (_d(2026, 6, 23), "Shipped the Vireo Health enclosure two days early.", {}),
    (
        _d(2026, 7, 7),
        "Sarah at Bramwell said their annual tooling budget is around 40000 pounds.",
        {"confidence": "unverified"},
    ),
    (
        _d(2026, 7, 8),
        "Bramwell's finance director put the tooling budget at 52000 pounds for the year.",
        {},
    ),
    (
        _d(2026, 7, 15),
        "Third client meeting I've done at eight in the morning. I do my best work before ten.",
        {},
    ),
    (
        _d(2026, 7, 21),
        "Took the sleeper train to Glasgow for the Vireo review rather than fly.",
        {},
    ),
    (
        _d(2026, 7, 22),
        "Second time this year I've taken the train instead of flying to a client.",
        {},
    ),
    (
        _d(2026, 8, 1),
        "Moved off the Bambu to a resin printer for the fine detail work on medical parts.",
        {},
    ),
    # -- CHANGE OVER TIME, in each of the shapes it actually takes -----------
    # Supersession is only the tidiest one. A value can also be REVERSED (the
    # thing was undone), RETRACTED (the first statement was simply wrong),
    # REFINED (vague became exact) or PARTIALLY updated (one field of a
    # compound fact moved and the rest did not). Each fails differently: a
    # reversal answered with the original leaves the user holding a thing they
    # returned, a retraction answered with the original quotes a number that
    # was never true.
    (
        _d(2026, 8, 9),
        "Returned the resin printer. It could not hold tolerance on the "
        "Vireo parts, so I am back on the Bambu for everything.",
        {},
    ),
    (_d(2026, 5, 2), "Invoiced Okonjo Studio 2300 pounds for the retail fixture.", {}),
    (
        _d(2026, 5, 4),
        "Correction: the Okonjo invoice was 3200, not 2300. I had missed "
        "the tooling line entirely.",
        {},
    ),
    (_d(2026, 3, 12), "The Milan trip cost somewhere around 800 all in, I think.", {}),
    (_d(2026, 3, 30), "Final Milan total came to 847.20 once every receipt was in.", {}),
    (_d(2026, 6, 15), "The Vireo enclosure is going in ABS.", {}),
    (
        _d(2026, 7, 2),
        "Switched the Vireo enclosure from ABS to polycarbonate to pass "
        "the drop test. Everything else about the job is unchanged.",
        {},
    ),
    # -- CONTENT SHAPES that are not plain assertions ------------------------
    # A negation, a third-party claim, a conditional and a plan are all things a
    # memory store must hold WITHOUT flattening them into facts. Answering "he
    # owns a Volvo" from "I don't own a car" is the same class of error as
    # answering "aluminium will double" from "Marcus reckons aluminium will
    # double" -- the memory was recorded correctly and read carelessly.
    (
        _d(2026, 1, 5),
        "I do not own a car and never have. Everything moves by train or by courier.",
        {},
    ),
    (_d(2026, 4, 20), "Marcus reckons aluminium prices will double by next spring.", {}),
    (_d(2026, 6, 28), "If Vireo goes above 5000 units I will need a second supplier.", {}),
    (
        _d(2026, 8, 10),
        "Booked the Rotterdam fair for October. Not been to that one before.",
        {},
    ),
    (_d(2026, 2, 10), "Every Tuesday I do a full workshop clean-down.", {}),
    (_d(2026, 1, 3), "I have been renting the Peckham unit since March 2025.", {}),
    # -- MONEY, so aggregation has something real to aggregate ---------------
    (_d(2026, 2, 18), "Invoiced Halden Foods 4200 pounds for the packaging refresh.", {}),
    (_d(2026, 3, 25), "Invoiced Bramwell Cycles 5100 pounds for the light housing.", {}),
    (_d(2026, 6, 23), "Invoiced Vireo Health 6800 pounds the same afternoon I shipped.", {}),
    (_d(2026, 4, 10), "Bramwell paid 40 days after invoice. I had to chase them twice.", {}),
    (_d(2026, 5, 30), "Okonjo paid on the day the invoice landed.", {}),
    (_d(2026, 7, 4), "Vireo paid inside a fortnight, no chasing.", {}),
    # -- RETRIEVAL HARDNESS --------------------------------------------------
    # A near-name that is NOT the client (Haldern vs Halden), a fact stated
    # only in words the question will not use, a three-link chain, and two
    # numbers where one is a prefix of the other.
    (
        _d(2026, 2, 25),
        "Had a call with Haldern Design about a rebrand. Nothing came of it.",
        {},
    ),
    (
        _d(2026, 6, 5),
        "The bench lamp finally gave up, so the whole back wall is now lit "
        "by a single overhead strip that flickers.",
        {},
    ),
    (_d(2026, 1, 25), "My contact at Bramwell Cycles is Sarah Okoye.", {}),
    (_d(2026, 6, 1), "Sarah Okoye left Bramwell and moved to Vireo Health in June.", {}),
    (_d(2026, 4, 29), "The Okonjo fixture stands 180cm tall.", {}),
    (_d(2026, 4, 30), "The shelf pitch on the Okonjo fixture is 18cm.", {}),
    # A NEAR-DUPLICATE of the Bambu purchase. Two memories describing one
    # event is the normal state of an append-only store, and the failure it
    # invites is arithmetic: a printer total of 3497 instead of 2198 means the
    # same 1299 was counted twice. Nothing in the corpus says these are the
    # same purchase; the model has to notice.
    (_d(2026, 4, 16), "Paid 1299 for the Bambu, delivered Thursday.", {}),
)

#: A DIFFERENT space, same org. Nothing here may ever appear in an answer about
#: the space above -- this is the tenancy probe, and it is the one failure in
#: this file that would be a security incident rather than a quality problem.
OTHER_MEMORIES: tuple[tuple[datetime, str], ...] = (
    (_d(2026, 5, 2), "The Caldwell Motors retainer is 4200 pounds a month."),
    (_d(2026, 5, 3), "Caldwell's brand colour is Pantone 3005 C."),
)

#: (capability, question, spec-or-None, note). A None spec means the check is
#: behavioural rather than lexical and is asserted separately below.
CASES: tuple[tuple[str, str, str | None], ...] = (
    ("direct", "What printer did I buy in February?", "prusa"),
    ("direct", "Which client did I sign first?", "halden"),
    ("count", "How many client projects did I ship in 2026?", "four|4"),
    ("order", "Which client did I sign most recently?", "vireo"),
    ("order", "What was the first thing I bought for the workshop?", "prusa"),
    (
        "date_arith",
        "How many days passed between starting and shipping the Bramwell light housing?",
        "20|twenty",
    ),
    (
        "compare",
        "Of the two Peckham studios I quoted, which was cheaper and by how much?",
        "b|1150; 250",
    ),
    ("compare", "Which printer cost me more, the Prusa or the Bambu?", "bambu|x1c"),
    (
        "list_all",
        "List every client I have worked with this year.",
        "halden; bramwell; okonjo; vireo",
    ),
    ("temporal", "What did I do in March?", "bramwell|milan|fair|housing"),
    ("multi", "How much did I spend on 3D printers in total?", "2198|2 198"),
    ("supersede", "What is my current day rate?", "700 + 500"),
    ("supersede", "What should I quote a new client for a day's work?", "700 + 500"),
    ("contradict", "What is Bramwell's annual tooling budget?", "40000|52000"),
    ("unverified", "What did Sarah tell me about Bramwell's budget?", "40000"),
    ("abstain", "What is my accountant's name?", None),
    ("abstain", "Which university did I study at?", None),
    ("abstain", "What did the Caldwell Motors retainer come to?", None),
    ("tenancy", "What is Caldwell's brand colour?", None),
    (
        "pref_explicit",
        "A drawing package is coming in from a new client. What should I insist on?",
        "metric|mm|millimet + imperial|inch",
    ),
    (
        "pref_implicit",
        "I need to book a recurring slot for deep design work. When should I put it?",
        "morning|mornings|early|before ten|10|nine|8",
    ),
    (
        "pref_negative",
        "A nicotine pouch brand wants a packaging refresh at double my rate. Should I take it?",
        "no|decline|turn|refuse|pass|avoid|reject|not",
    ),
    (
        "pref_updated",
        "Quote me for a four-day job for a returning client.",
        "2800|700 + 500|2000",
    ),
    (
        "pref_composed",
        "A tobacco-adjacent startup wants a device enclosure, drawings in inches, "
        "meeting at 8am. Which parts of that should I push back on?",
        "tobacco|nicotine|vape|decline|turn|refuse; imperial|inch|metric",
    ),
    (
        "pref_composed",
        "Plan how I should get to a client review in Edinburgh next month.",
        "train|rail|sleeper + fly|flight|plane",
    ),
    # -- change over time, beyond plain supersession -------------------------
    ("reversal", "Which printer am I using for detail work now?", "bambu|x1c + resin"),
    ("reversal", "Do I still have the resin printer?", "no|returned|sent back|not + bambu"),
    ("retraction", "How much did I invoice Okonjo Studio?", "3200 + 2300"),
    ("refinement", "Exactly what did the Milan trip cost?", "847"),
    ("partial_update", "What material is the Vireo enclosure?", "polycarbonate|pc + abs"),
    (
        "partial_update",
        "Did switching the Vireo material change anything else about that job?",
        "no|nothing|unchanged|only|just|same",
    ),
    # -- content shapes that must not flatten into plain facts ---------------
    ("negation", "What car do I drive?", "no|not|none|don't|do not|never"),
    (
        "attribution",
        "Is aluminium going to double in price by spring?",
        "marcus + reckons|thinks|said|claim|according|believes",
    ),
    (
        "conditional",
        "Do I need a second supplier for Vireo?",
        "if|depends|only|unless|conditional|above|over|5000",
    ),
    ("planned", "Have I been to the Rotterdam fair?", "no|not|haven't|have not|booked|october"),
    ("recurring", "What do I do every Tuesday?", "clean|cleaning|clean-down|workshop"),
    ("interval", "How long have I had the Peckham unit?", "2025|march|year|18|eighteen"),
    # -- aggregation over the invoice set ------------------------------------
    ("sum", "What did I invoice in total across all four projects?", "19300|19 300"),
    ("max", "Which project invoiced the most, and how much?", "vireo; 6800"),
    ("min", "Which was my smallest invoice?", "okonjo|3200"),
    ("average", "What was my average invoice value this year?", "4825"),
    ("group_by", "How many projects did I ship in the second quarter?", "two|2 + okonjo|vireo"),
    (
        "set_difference",
        "Which client was slowest to pay?",
        "bramwell + 40|forty|chase",
    ),
    #: `?` prefix: a decline is ALSO a correct answer here.
    #:
    #: This case originally demanded a flat "no", and that encoded a closed-world
    #: assumption a memory store does not get to make. Absence of an automotive
    #: client in the store is not evidence the user never had one -- they may
    #: simply never have written it down. The first run answered "I have worked
    #: for Bramwell Cycles", which is both wrong and the closed-world reading
    #: taken confidently; the run after answered "I don't have that in memory",
    #: which is the honest one, and the spec scored it a failure. The spec was
    #: what needed fixing.
    (
        "negative_existence",
        "Have I ever worked with an automotive client?",
        "?no|not|never|nothing|nobody|none",
    ),
    # -- retrieval hardness ---------------------------------------------------
    (
        "distractor",
        "What did I do for Halden Foods?",
        "packaging|refresh + haldern|rebrand",
    ),
    (
        "distractor",
        "Did the Haldern Design rebrand go ahead?",
        "no|not|nothing|didn't|did not|never",
    ),
    (
        "paraphrase",
        "Is the lighting in my workshop any good?",
        "flicker|flickers|single|overhead|strip|poor|bad|one",
    ),
    (
        "multi_hop",
        "Who is my contact at Vireo Health, and where did I know them from?",
        "sarah|okoye; bramwell",
    ),
    ("numeric", "How tall is the Okonjo fixture?", "180 + 18"),
    ("numeric", "What is the shelf pitch on the Okonjo fixture?", "18 + 180"),
    # -- BITEMPORAL: state as of a past moment, which is the thing this store's
    # valid_from/valid_to columns exist for. "What is my rate" and "what was my
    # rate in April" have different answers and only one of them is written
    # down. Answering the second with 85 is reading today's value into the past.
    ("as_of", "In April, what did I think my day rate was?", "500 + 700"),
    (
        "as_of",
        "Had I bought the Bambu yet when I shipped the Bramwell housing?",
        "no|not|hadn't|had not|after|later + april|march",
    ),
    # -- relative and derived time --------------------------------------------
    (
        "relative_date",
        "What did I get up to last month?",
        "glasgow|sleeper|train|tooling|budget|polycarbonate|morning|vireo",
    ),
    (
        "duration",
        "How long did the Vireo job take from signing to shipping?",
        "21|twenty-one|three weeks|3 weeks",
    ),
    ("ongoing", "Am I still renting the Peckham unit?", "yes|still|since|continue"),
    (
        "same_day",
        "Did I invoice Vireo before or after I shipped?",
        "after|same|afternoon|later",
    ),
    # -- causal and counterfactual --------------------------------------------
    ("causal", "Why did I buy the Bambu?", "prusa + keep up|slow|capacity|couldn't|could not"),
    (
        "causal",
        "Why did I get rid of the resin printer?",
        "toleranc|precision|accuracy|quality|vireo",
    ),
    (
        "counterfactual",
        "If I had never raised my rate, what would a four-day job bill at?",
        "2000 + 500",
    ),
    ("delta", "How much has my day rate gone up this year?", "200 + 500|700"),
    ("ratio", "What proportion of my clients paid late?", "one|1|quarter|25 + four|4|bramwell"),
    # -- unit handling ---------------------------------------------------------
    ("unit", "How tall is the Okonjo fixture in metres?", "1.8|1 8"),
    # -- deduplication ---------------------------------------------------------
    ("near_duplicate", "What did the Bambu cost me?", "1299 + 2598"),
    # -- entity resolution -----------------------------------------------------
    ("coreference", "Who told me about Bramwell's tooling budget?", "sarah"),
    ("role_change", "Where does Sarah Okoye work now?", "vireo + bramwell"),
    # -- answer quality --------------------------------------------------------
    (
        "partial",
        "What did the resin printer cost me?",
        "?no|not|don't|do not|unknown|unclear|doesn't say|does not say|no record",
    ),
    (
        "ambiguity",
        "How much did the Okonjo fixture come to?",
        "3200 + 2300|invoice|corrected",
    ),
)

#: question -> a substring of the memory that ACTUALLY supports the answer.
#:
#: Counting citations says nothing about whether they point anywhere useful. A
#: reply can carry [3] against a claim memory 3 does not support and score as
#: fully cited, which is the audit trail failing quietly rather than loudly.
#: Only questions with one unambiguous supporting memory are listed -- an
#: aggregate legitimately cites four, and demanding a particular one would
#: measure formatting.
EVIDENCE: dict[str, str] = {
    "What printer did I buy in February?": "Prusa MK4",
    "Which client did I sign first?": "first contract with Halden",
    "What is my current day rate?": "Raised my day rate",
    "How much did I invoice Okonjo Studio?": "Correction: the Okonjo invoice",
    "What material is the Vireo enclosure?": "polycarbonate",
    "What do I do every Tuesday?": "clean-down",
    "Why did I buy the Bambu?": "could not keep up",
    "Where does Sarah Okoye work now?": "moved to Vireo Health",
    "What is the shelf pitch on the Okonjo fixture?": "shelf pitch",
    "How tall is the Okonjo fixture?": "180cm tall",
}


async def main() -> None:
    from bench.harness import build_model_client

    client = build_model_client(MODEL, PROJECT)
    gap = asyncio.Semaphore(MAX_INFLIGHT)

    async def complete(prompt: str) -> str:
        async with gap:
            raw, _tokens = await client.complete(prompt, max_tokens=1500)
        return raw

    settings = Settings(
        environment="test",
        store_backend="memory",
        embedding_backend="deterministic",
        embedding_dimensions=768,
        rerank_backend="heuristic",
        api_key_pepper="capability-live-pepper",
    )
    store = InMemoryStore()
    service = MemoryService(
        store, GeminiEmbedder(dimensions=768), HeuristicReranker(), settings
    )
    org = await store.create_organization(Organization(name="CapabilityLive"))
    main_space = await store.create_space(Space(org_id=org.id, slug="cl-main", name="Studio"))
    other_space = await store.create_space(Space(org_id=org.id, slug="cl-other", name="Other"))

    print(f"ingesting {len(MEMORIES)} memories (+{len(OTHER_MEMORIES)} in a second space) ...")
    for occurred, content, meta in MEMORIES:
        await service.ingest(
            org_id=org.id,
            space_id=main_space.id,
            content=content,
            occurred_at=occurred,
            metadata=meta,
            extract=False,
        )
    for occurred, content in OTHER_MEMORIES:
        await service.ingest(
            org_id=org.id,
            space_id=other_space.id,
            content=content,
            occurred_at=occurred,
            extract=False,
        )

    async def ask(capability: str, question: str, spec: str | None) -> dict:
        response = await service.search(
            SearchRequest(
                query=question,
                org_id=org.id,
                space_id=main_space.id,
                #: Wider than the product default because the aggregation cases
                #: need four separate invoice memories at once, and a sum that
                #: fails for want of a row is a RETRIEVAL result being reported
                #: as an arithmetic one. Still well under the corpus size, so
                #: the ranker is exercised rather than bypassed.
                limit=20,
                asked_at=ASKED_AT,
            )
        )
        result = await chat_answer(question, response.results, complete)
        reply = result.reply
        low = reply.lower()

        # Did the citation point at the memory that actually supports this?
        # None when the question has no single unambiguous source.
        marker = EVIDENCE.get(question)
        by_id = {h.memory.id: h.memory.content for h in response.results}
        cited_right: bool | None = None
        if marker is not None:
            cited_right = any(
                marker.lower() in by_id.get(mid, "").lower() for mid in result.cited
            )

        declined = any(
            phrase in low
            for phrase in (
                "don't have",
                "do not have",
                "not in memory",
                "no memory",
                "nothing in",
            )
        )
        # Leakage is checked against the OTHER space's distinctive values, not
        # against a similarity -- either the string is in the reply or it is not.
        #
        # Every token here must be UNIQUE to the other space. "4200" was in this
        # list and flagged two CORRECT answers, because 4200 is also the Halden
        # invoice in the main space: the detector was reporting a coincidence of
        # digits as a tenancy breach. A leak check that cries wolf is worse than
        # no leak check, because the one alarm that matters stops being read.
        leaked = any(token in low for token in ("caldwell", "pantone", "3005"))
        if spec is None:
            #: abstain/tenancy: the ONLY correct behaviour is to decline, and a
            #: fluent invented answer is the failure this catches.
            correct = declined and not leaked
        elif spec.startswith("?"):
            #: open-world: either an explicit negative or an honest decline.
            correct = declined or score(reply, Expected.parse(spec[1:])).correct
        else:
            correct = score(reply, Expected.parse(spec)).correct
        return {
            "capability": capability,
            "question": question,
            "correct": correct,
            "declined": declined,
            "leaked": leaked,
            "cited": len(result.cited),
            "cited_right": cited_right,
            "used_unverified": result.used_unverified,
            "reply": reply,
        }

    print(f"asking {len(CASES)} questions through the production chat path ({MODEL}) ...\n")
    rows = await asyncio.gather(*(ask(c, q, s) for c, q, s in CASES))

    n = len(rows)
    hits = sum(r["correct"] for r in rows)
    low, high = wilson(hits, n)
    print(f"OVERALL  {hits}/{n}  {hits / n:.1%}   95% CI [{low:.1%}, {high:.1%}]\n")

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["capability"], []).append(r)
    print(f"{'capability':16} {'score':>8}   {'cited':>6}")
    for cap, items in groups.items():
        c = sum(i["correct"] for i in items)
        cited = sum(i["cited"] for i in items)
        mark = "" if c == len(items) else "   <-- "
        print(f"{cap:16} {c:4}/{len(items):<3} {cited:6}{mark}")

    leaks = [r for r in rows if r["leaked"]]
    print(f"\nTENANCY: {len(leaks)} answers leaked another space's content", end="")
    print(" -- CLEAN" if not leaks else f" -- {[r['question'] for r in leaks]}")

    abstain = [r for r in rows if r["capability"] in {"abstain", "tenancy"}]
    refused = sum(r["declined"] for r in abstain)
    print(f"ABSTENTION: declined {refused}/{len(abstain)} unanswerable")

    unver = [r for r in rows if r["capability"] == "unverified"]
    print(f"UNVERIFIED flagged on {sum(r['used_unverified'] for r in unver)}/{len(unver)}")

    checked = [r for r in rows if r["cited_right"] is not None]
    good = sum(bool(r["cited_right"]) for r in checked)
    print(f"CITATIONS point at the supporting memory: {good}/{len(checked)}")
    for r in checked:
        if not r["cited_right"]:
            print(f"  wrong/absent source: {r['question']}")

    print("\nFAILURES:")
    for r in rows:
        if not r["correct"]:
            print(f"\n  [{r['capability']}] {r['question']}")
            print(f"    {r['reply'][:400]}")

    out = Path(__file__).with_name("capability_live_results.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
