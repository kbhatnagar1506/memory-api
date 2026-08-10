"""Seed one person's context into a mapi space, chunked into atomic memories.

The unit here is one FACT, not one document. A profile pasted in as a single
blob becomes one averaged embedding: ask "what visa is he on" and you get the
whole biography back, ranked against every other biography-shaped thing. Split
into atomic claims, the same question retrieves the two sentences that answer
it. That decomposition is what `extract=True` does automatically on the write
path -- doing it by hand here means the chunking is legible and reviewable
rather than a model's guess.

Three properties each memory carries, and each earns its place:

  * `occurred_at` is EVENT time -- when the fact became true, not when it was
    written down. Reakon started May 2026 and that is the date on the memory,
    so "what was he doing last summer" sorts correctly instead of reading as
    though everything happened the day this script ran.
  * `tags` are the coarse filter: identity, reakon, profitwise, research,
    credentials, jobsearch.
  * `metadata.confidence` is `verified` or `unverified`. The source document
    flagged several claims as needing checking, and a memory system that
    launders "the founder told me" into ground truth is worse than no memory
    system -- it will repeat the shaky number back with the same confidence
    as the solid ones, right before an interview.

Usage:
    MAPI_API_KEY=sm_... python scripts/seed_profile.py --space krishna
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk", "src"))

from mapi_sdk import Mapi

DEFAULT_BASE = "https://memory-api-7b178bde9ecc.herokuapp.com"


def when(year: int, month: int, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)

NOW = when(2026, 8, 10)
REAKON = when(2026, 5, 1)
PROFITWISE = when(2025, 7, 1)

#: (content, tags, occurred_at, verified)
#:
#: Written in the THIRD PERSON and self-contained. A memory read six months
#: from now beside forty others has no idea what "I" or "the company" referred
#: to at write time, so every one names its subject.
FACTS: list[tuple[str, list[str], datetime, bool]] = [
    # -- identity ---------------------------------------------------------
    ("Krishna Bhatnagar is a computer science student and a two-time startup "
     "CTO / founding engineer.", ["identity"], NOW, True),
    ("Krishna Bhatnagar is studying for a BS in Computer Science at Georgia "
     "State University, expected to graduate May 2028.", ["identity", "education"],
     when(2024, 8), True),
    ("Krishna Bhatnagar is a non-degree special student at Georgia Tech, "
     "admitted through the CreateX accelerator.", ["identity", "education"],
     PROFITWISE, True),
    ("Krishna Bhatnagar is based in Atlanta, Georgia, and is relocating in "
     "August 2026. His family roots are in Gurugram, India.",
     ["identity"], NOW, True),
    ("Krishna Bhatnagar is on an F-1 visa. He needs CPT authorization to work "
     "during semesters, and must re-enter the United States before "
     "15 September 2026.", ["identity", "visa", "jobsearch"], NOW, True),
    ("Krishna Bhatnagar's email is kbhatnagar1@student.gsu.edu and his phone "
     "number is (404) 247-9391.", ["identity", "contact"], NOW, True),
    ("Krishna Bhatnagar's GitHub username is kbhatnagar1506 and his LinkedIn "
     "is /in/krishna-bhatnagar.", ["identity", "contact"], NOW, True),
    ("Krishna Bhatnagar's working style is terse and direct. He ships fast and "
     "pushes back hard on assumptions.", ["identity", "style"], NOW, True),

    # -- Reakon Labs ------------------------------------------------------
    ("Krishna Bhatnagar is Principal Engineer and co-founder of Reakon Labs "
     "Pvt. Ltd., which he joined in May 2026 and where he still works.",
     ["reakon", "role"], REAKON, True),
    ("Reakon Labs runs a live GST input-tax-credit compliance platform for "
     "Indian businesses and chartered accountancy firms, at app.reakon.in.",
     ["reakon", "product"], REAKON, True),
    ("Reakon's stack is Next.js 15, TypeScript, and Supabase/Postgres deployed "
     "on Heroku. It is multi-tenant with row-level security.",
     ["reakon", "stack"], REAKON, True),
    ("Krishna shipped 200+ production releases at Reakon in 90 days as the "
     "sole engineer and sole reviewer.", ["reakon", "impact"], when(2026, 8), True),
    ("Reakon's engineering governance is CODEOWNERS branch protection, "
     "PR-only merges, a four-job GitHub Actions CI pipeline (typecheck, lint, "
     "build, gitleaks), and Dependabot.", ["reakon", "stack"], REAKON, True),
    ("Reakon's ingestion pipeline captures invoices over WhatsApp via Twilio, "
     "extracts them with Gemini OCR plus a LoRA fine-tuned open-weight Llama "
     "trained on Indian invoices, runs a three-stage reconciliation cascade "
     "(exact, fuzzy, AI), and ends in a statutory verdict engine.",
     ["reakon", "architecture"], REAKON, True),
    ("Reakon's verdict engine encodes Indian GST statute directly: section "
     "16(4), section 17(5), and the 180-day reversal rule, producing six "
     "explainable verdict buckets over a rolling 12-month window.",
     ["reakon", "architecture"], REAKON, True),
    ("Reakon's self-gating sync layer cut government API calls by roughly 90% "
     "across the GSTR-2B, 2A, 3B and 1 endpoints.", ["reakon", "impact"],
     REAKON, True),
    ("Reakon prevented ₹5.57 lakh of phantom input tax credit for a customer "
     "by modelling filing state.", ["reakon", "impact"], REAKON, True),
    ("Aaditi Singhal is Krishna's co-founder and co-CEO at Reakon Labs.",
     ["reakon", "people"], REAKON, True),
    ("Krishna closed Reakon's paying customers himself by cold-walking "
     "industrial estates in Pace City II and Udyog Vihar, Gurugram.",
     ["reakon", "gtm"], REAKON, True),
    ("Reakon has two chartered accountancy channel partners. Whether these are "
     "two firms or two individual CAs (Sujeet Chaudhary, and Aryan Goyal of "
     "VD & Co) is UNVERIFIED and should be checked before being claimed.",
     ["reakon", "gtm"], REAKON, False),
    ("Whether Reakon's fine-tuned Llama serves live production extraction, or "
     "whether Gemini is still the primary extractor, is UNVERIFIED. It "
     "determines whether the in-house training-data claim holds.",
     ["reakon", "architecture"], REAKON, False),
    ("Whether Gmail invoice ingestion actually shipped in Reakon is "
     "UNVERIFIED — it does not appear in the work-history bullets.",
     ["reakon", "product"], REAKON, False),

    # -- ProfitWise -------------------------------------------------------
    ("Krishna Bhatnagar was CTO and co-founder of ProfitWise Inc. from July "
     "2025 to April 2026.", ["profitwise", "role"], PROFITWISE, True),
    ("ProfitWise was an AI accounting and cash-flow platform for US small and "
     "medium businesses, built through the Georgia Tech CreateX Startup "
     "Launch accelerator.", ["profitwise", "product"], PROFITWISE, True),
    ("ProfitWise raised a $5,000 SAFE and roughly $150,000 in cloud credits. "
     "Whether the SAFE was a SAFE or a grant, and whether the credits came "
     "from CreateX or directly from AWS/Google/Microsoft/NVIDIA, is "
     "UNVERIFIED — Krishna has stated both versions.",
     ["profitwise", "funding"], PROFITWISE, False),
    ("ProfitWise integrated nine providers: Plaid, QuickBooks, Xero, Zoho, "
     "Ramp, Shopify, Gmail, Slack and WhatsApp, with webhooks and idempotent "
     "async handling.", ["profitwise", "architecture"], PROFITWISE, True),
    ("Krishna built ProfitWise's agent memory layer — embeddings, a vector "
     "store and retrieval ranking — plus the orchestration around it, and a "
     "transaction reconciliation and cash-flow forecasting engine.",
     ["profitwise", "architecture"], PROFITWISE, True),
    ("ProfitWise won both the Drive Capital track and the Google Cloud track "
     "at AI ATL.", ["profitwise", "credentials"], when(2025, 11), True),
    ("Krishna's ProfitWise co-founders were Aaditi Singhal and Mohit Kokane.",
     ["profitwise", "people"], PROFITWISE, True),
    ("The claim that ProfitWise was the only team admitted to CreateX without "
     "prior Georgia Tech affiliation is UNVERIFIED. Unless CreateX staff "
     "stated it, it should be softened to 'one of very few'.",
     ["profitwise", "credentials"], PROFITWISE, False),
    ("ORBIT is an AI log monitor Krishna built that routes pull-request "
     "approvals through WhatsApp. It is deployed across both ProfitWise and "
     "Reakon.", ["profitwise", "reakon", "projects"], when(2026, 1), True),

    # -- research ---------------------------------------------------------
    ("Krishna Bhatnagar is sole author of 'The Persistence of Vision: "
     "State-of-the-Art Privacy for Multi-Vector VLM Retrievers', in "
     "preparation under the Reakon Labs affiliation.", ["research"],
     when(2026, 6), True),
    ("The Persistence of Vision presents a training-free dictionary attack on "
     "multi-vector VLM retrievers ColPali and ColQwen2. It recovers name, ID "
     "and date of birth at 1.00, 0.85 and 0.82 top-1 accuracy from a K=200 "
     "lineup, where chance is 0.005.", ["research", "results"], when(2026, 6), True),
    ("A BiPali single-vector control scoring 0.08/0.03/0.02 isolates the "
     "multi-vector architecture as the cause of the privacy leak, rather than "
     "the underlying vision model.", ["research", "results"], when(2026, 6), True),
    ("The research shows redaction fails GDPR Article 17 erasure: recovery "
     "stays at 1.00 even through dilated erasure. An adaptive attacker also "
     "inverts the learned reshaping defense at cosine similarity 0.998.",
     ["research", "results"], when(2026, 6), True),
    ("Cataract is Krishna's proposed defense: an index-time subspace "
     "projection that holds the strongest adaptive attack to 10% success "
     "while preserving 87.5% of retrieval quality, at 0.64ms per page and "
     "zero storage or query overhead.", ["research", "results"], when(2026, 6), True),
    ("The primary target venue for The Persistence of Vision is PoPETs 2027 "
     "Issue 2, deadline 31 August. ARTMAN at ACSAC 2026 is a parallel "
     "non-archival submission, and IEEE TIFS is flagged as the highest "
     "eventual-acceptance venue.", ["research", "plan"], when(2026, 6), True),
    ("Author status on The Persistence of Vision is INCONSISTENT in Krishna's "
     "own records — some say sole author, others name Prerna Singhal as "
     "co-author. This must be resolved before submission.",
     ["research"], when(2026, 6), False),

    # -- projects ---------------------------------------------------------
    ("Mapi, in its Visual Memory Layer form, is Krishna's concept for turning "
     "a typed knowledge graph into a 2D visual canvas passed to multimodal "
     "models as an image payload. There is an 11-page concept document.",
     ["projects", "mapi"], when(2026, 7), True),
    ("The Mapi visual-canvas hypothesis needs a two-week ablation with "
     "explicit kill criteria before any build starts.", ["projects", "mapi"],
     when(2026, 7), True),
    ("Krishna's memory system was integrated into MemoryBench as a competing "
     "provider. It is strong on single-session preference questions and has a "
     "measured gap on knowledge updates compared to Supermemory.",
     ["projects", "mapi"], when(2026, 7), True),
    ("Krishna's other named projects are HivePath AI, BuildWise, EcoAI, "
     "SafePay, NutriLens AR (built at ImmerseGT), and MetabolX.",
     ["projects"], when(2025, 6), True),

    # -- credentials ------------------------------------------------------
    ("Krishna Bhatnagar has 7 hackathon wins across roughly 24 competitions. "
     "The figure is 7 wins, not 24 — he has corrected this before.",
     ["credentials"], NOW, True),
    ("Krishna won 1st place at HackMIT 2025 on the Infosys track, against a "
     "field of more than 1,000 hackers.", ["credentials"], when(2025, 9), True),
    ("Krishna's named hackathon wins include TreeHacks, HackHarvard, "
     "HackPrinceton, HackGT and AI ATL.", ["credentials"], when(2025, 11), True),
    ("Krishna was in the DevHouse SF 2025 Top 100 Builders cohort, where he "
     "built an investor due-diligence agent on browser-use alongside its "
     "founding engineer. This was a collaboration — he was not himself a "
     "founding engineer at browser-use.", ["credentials"], when(2025, 10), True),
    ("Krishna's GitHub shows 1,612 contributions in a year, verified from a "
     "screenshot of his profile.", ["credentials"], NOW, True),

    # -- job search -------------------------------------------------------
    ("Krishna Bhatnagar is targeting founding engineer and forward-deployed "
     "engineer roles at seed to Series A AI-native startups.",
     ["jobsearch"], NOW, True),
    ("Krishna is in an active process with Noto — contact AJ Ding, backed by "
     "Base10 and SPC, after-school SaaS, around $170K posted. A pre-call "
     "build is underway.", ["jobsearch", "pipeline"], NOW, True),
    ("Krishna is pursuing Dedalus Labs (YC S25, contact Cathy Di, agent "
     "infrastructure) for a systems intern role, which he considers a reach.",
     ["jobsearch", "pipeline"], NOW, True),
    ("Krishna has applications out through WaaS and Underdog.",
     ["jobsearch", "pipeline"], NOW, True),
    ("Because Krishna graduates in May 2028, full-time roles need a CPT or "
     "part-time-now framing. He cannot take 1099 gig work on an F-1 visa.",
     ["jobsearch", "visa"], NOW, True),

    # -- open actions -----------------------------------------------------
    ("Krishna has an open action item to pin and write READMEs for six GitHub "
     "repositories. It is unresolved and blocked on him listing his public "
     "repos.", ["todo"], NOW, True),
    ("Krishna is considering publishing a sanitized public version of the "
     "Reakon repository, since the real work is private and therefore "
     "invisible to anyone evaluating him.", ["todo"], NOW, True),
    ("Krishna plans to run the autumn hackathon circuit, with AI ATL on "
     "7-9 November as the highest priority.", ["todo"], when(2026, 11, 7), True),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="krishna")
    parser.add_argument("--base-url", default=os.getenv("MAPI_BASE_URL", DEFAULT_BASE))
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Also decompose each fact server-side. Off by default: these are "
        "already atomic, and a second pass would mostly restate them.",
    )
    args = parser.parse_args()

    client = Mapi(base_url=args.base_url)
    client.spaces.get_or_create(args.space)

    written = duplicates = 0
    for content, tags, occurred, verified in FACTS:
        result = client.memories.add(
            content,
            space=args.space,
            tags=tags,
            occurred_at=occurred,
            source="profile",
            metadata={"confidence": "verified" if verified else "unverified"},
            extract=args.extract or None,
        )
        if getattr(result, "created", True):
            written += 1
        else:
            duplicates += 1
        mark = " " if verified else "?"
        print(f"  {mark} {content[:78]}")

    unverified = sum(1 for *_, v in FACTS if not v)
    print(
        f"\n{written} written, {duplicates} already present, "
        f"{unverified} flagged unverified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
