"""The benchmark harness itself.

This code produces the numbers we publish, so it needs the same scrutiny as the
service. Six harness bugs in this project have fabricated or destroyed a result:
a cache key that served answers to questions never asked, patches that silently
failed to apply, chunking that dropped content, API errors scored as wrong
answers, a context assembler that passed 9% of what it retrieved, and a thread
leak that hung a run for two hours. Every test below pins one of those.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bench.cache import DiskVectorCache, cache_key
from bench.datasets.base import Corpus, Document, Question
from bench.datasets.locomo import LoCoMo
from bench.datasets.longmemeval import LongMemEval
from bench.harness import (
    _JUDGE_MAX_TOKENS,
    NO_OUTPUT,
    OFFICIAL_JUDGE_PROMPTS,
    _token_batches,
    judge_prompt,
    parse_answer,
    parse_grade,
)
from bench.metrics import (
    full_recall_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    wilson,
)

DATA = Path(__file__).resolve().parents[2] / "bench" / "data"


# -- metrics -------------------------------------------------------------------


def test_full_recall_requires_every_evidence_item() -> None:
    """The metric fix. hit@k said multi-hop retrieval worked (74%); full
    recall said it did not (22%), and accuracy tracked full recall."""
    retrieved, relevant = ["a", "b", "c"], {"a", "z"}
    assert hit_at_k(retrieved, relevant, 3) == 1.0
    assert full_recall_at_k(retrieved, relevant, 3) == 0.0
    assert full_recall_at_k(["a", "z"], relevant, 3) == 1.0


def test_metrics_respect_k() -> None:
    retrieved, relevant = ["x", "a"], {"a"}
    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert full_recall_at_k(retrieved, relevant, 1) == 0.0


def test_metrics_on_empty_evidence_are_zero_not_one() -> None:
    """Abstention questions have no evidence; they must not score as perfect."""
    for fn in (full_recall_at_k, hit_at_k, recall_at_k, ndcg_at_k):
        assert fn(["a"], set(), 5) == 0.0
    assert mrr(["a"], set()) == 0.0


def test_mrr_and_ndcg_reward_rank() -> None:
    assert mrr(["a", "b"], {"a"}) == 1.0
    assert mrr(["b", "a"], {"a"}) == 0.5
    assert ndcg_at_k(["a", "b"], {"a"}, 2) > ndcg_at_k(["b", "a"], {"a"}, 2)


def test_wilson_is_sane_at_the_extremes() -> None:
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson(10, 10)
    assert 0 < lo < 1 and hi == 1.0
    assert wilson(0, 0) == (0.0, 0.0)


# -- answer parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"facts": "x", "answer": "Target"}', "Target"),
        ('{"answer": "NO_ANSWER"}', "NO_ANSWER"),
        ('{"answer": "7 May 2023"}', "7 May 2023"),
        # Numbers come back as strings; the judge compares text.
        ('{"answer": 3}', "3"),
        # null is a real decline, distinct from no output at all.
        ('{"answer": null}', ""),
        # Fenced despite the instruction: a formatting slip around a real
        # answer, not a wrong answer.
        ('```json\n{"answer": "Target"}\n```', "Target"),
        # Prose either side of the object.
        ('Here you go: {"answer": "Target"} hope that helps', "Target"),
        # Ignored the format entirely but still answered.
        ("bare answer with no format", "bare answer with no format"),
        ('{"answer": "  spaced  "}', "spaced"),
    ],
)
def test_parse_answer(raw: str, expected: str) -> None:
    assert parse_answer(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n  ",
        # Truncated mid-object: the failure this format exists to catch.
        '{"facts": "the user said", "answer": "Tar',
        '{"answer": "the thing is',
    ],
)
def test_no_output_is_distinguishable_from_a_wrong_answer(raw: str) -> None:
    """The whole reason for JSON.

    The FACTS/ANSWER text form could not tell a wrong answer from no answer --
    both arrive as a string. Measured cost: gemini-2.5-pro returned empty on
    64 of 118 "failures" because reasoning tokens consumed a 256-token
    allowance, and the arm reported 0.734 while measuring our ceiling rather
    than the model.
    """
    assert parse_answer(raw) == NO_OUTPUT


def test_only_the_answer_field_is_read() -> None:
    """`facts` is the model's working space, never graded.

    Grading a fact list as if it were an answer is how a formatting slip
    becomes a fake score.
    """
    raw = '{"facts": "the user mentioned Target and coffee", "answer": "coffee"}'
    assert parse_answer(raw) == "coffee"


def test_a_reply_carrying_only_reasoning_is_not_an_answer() -> None:
    assert parse_answer('{"facts": "the user mentioned Target"}') == (
        '{"facts": "the user mentioned Target"}'
    )


# -- token batching ------------------------------------------------------------


def test_batches_respect_the_token_budget() -> None:
    """The 500-corpus run died on a 55,464-token request against a 20,000
    limit: batching by item count alone is only safe for uniform documents."""
    texts = ["x" * 4000] * 10  # ~1000 tokens each
    batches = _token_batches(texts, max_items=32, max_tokens=3000)
    assert len(batches) > 1
    for batch in batches:
        assert sum(len(texts[i]) // 4 for i in batch) <= 3000 + 1000


def test_batches_respect_the_item_cap() -> None:
    batches = _token_batches(["tiny"] * 100, max_items=10, max_tokens=10**9)
    assert all(len(b) <= 10 for b in batches)


def test_batching_covers_every_item_exactly_once() -> None:
    texts = [f"doc {i}" for i in range(57)]
    seen = [i for batch in _token_batches(texts, max_items=8, max_tokens=500) for i in batch]
    assert sorted(seen) == list(range(57))


def test_oversized_single_item_still_emitted() -> None:
    """One document larger than the whole budget must not vanish."""
    batches = _token_batches(["y" * 100_000], max_items=8, max_tokens=100)
    assert [i for b in batches for i in b] == [0]


# -- disk cache ----------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = DiskVectorCache(tmp_path / "v.sqlite")
    key = cache_key("m", 3, "hello")
    cache.put_many([(key, [0.5, -0.25, 0.125])])
    got = cache.get_many([key])
    assert got[key] == pytest.approx([0.5, -0.25, 0.125])
    cache.close()


def test_cache_key_separates_models_and_dimensions() -> None:
    """A cache that mixes embedding spaces returns plausible, wrong rankings
    and raises nothing."""
    assert cache_key("a", 768, "t") != cache_key("b", 768, "t")
    assert cache_key("a", 768, "t") != cache_key("a", 256, "t")
    assert cache_key("a", 768, "t") == cache_key("a", 768, "t")


def test_cache_miss_returns_nothing_and_counts(tmp_path: Path) -> None:
    cache = DiskVectorCache(tmp_path / "v.sqlite")
    assert cache.get_many([cache_key("m", 3, "absent")]) == {}
    assert cache.misses == 1
    cache.close()


def test_cache_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "v.sqlite"
    key = cache_key("m", 2, "persist")
    first = DiskVectorCache(path)
    first.put_many([(key, [1.0, 2.0])])
    first.close()
    second = DiskVectorCache(path)
    assert second.get_many([key])[key] == pytest.approx([1.0, 2.0])
    second.close()


def test_cache_handles_more_keys_than_sqlite_variable_limit(tmp_path: Path) -> None:
    """SQLite caps ~999 variables per statement; a real run asks for 373,687."""
    cache = DiskVectorCache(tmp_path / "v.sqlite")
    keys = [cache_key("m", 2, f"t{i}") for i in range(2500)]
    cache.put_many([(k, [float(i), 0.0]) for i, k in enumerate(keys)])
    assert len(cache.get_many(keys)) == 2500
    cache.close()


# -- dataset loaders -----------------------------------------------------------


def test_document_and_question_are_wired_together() -> None:
    doc = Document(id="d1", text="hello", occurred_at=datetime(2026, 1, 1, tzinfo=UTC))
    q = Question(
        qid="q1",
        corpus_id="c1",
        text="?",
        answer="hello",
        evidence_ids=("d1",),
        category="test",
    )
    corpus = Corpus(corpus_id="c1", documents=[doc], questions=[q])
    assert set(q.evidence_ids) <= {d.id for d in corpus.documents}


@pytest.mark.skipif(not (DATA / "locomo10.json").exists(), reason="LoCoMo data absent")
def test_locomo_evidence_ids_resolve_to_documents() -> None:
    """An evidence id that matches no document silently scores 0 forever."""
    for corpus in LoCoMo(DATA / "locomo10.json").load(limit_corpora=2):
        ids = {d.id for d in corpus.documents}
        for q in corpus.questions:
            assert set(q.evidence_ids) <= ids, f"{q.qid} references missing documents"


@pytest.mark.skipif(not (DATA / "locomo10.json").exists(), reason="LoCoMo data absent")
def test_locomo_marks_adversarial_as_abstention() -> None:
    corpora = LoCoMo(DATA / "locomo10.json").load(limit_corpora=1)
    questions = [q for c in corpora for q in c.questions]
    assert any(q.is_abstention for q in questions)
    assert all(q.category == "adversarial" for q in questions if q.is_abstention)


@pytest.mark.skipif(
    not (DATA / "longmemeval_s.json").exists(), reason="LongMemEval data absent"
)
def test_longmemeval_evidence_ids_resolve_and_are_session_level() -> None:
    dataset = LongMemEval(DATA / "longmemeval_s.json")
    assert dataset.evidence_granularity == "session"
    for corpus in dataset.load(limit_corpora=6):
        ids = {d.id for d in corpus.documents}
        for q in corpus.questions:
            assert set(q.evidence_ids) <= ids, f"{q.qid} references missing sessions"


@pytest.mark.skipif(
    not (DATA / "longmemeval_s.json").exists(), reason="LongMemEval data absent"
)
def test_longmemeval_stratifies_across_capabilities() -> None:
    """A small run must still measure every capability, not whichever sorts first."""
    corpora = LongMemEval(DATA / "longmemeval_s.json").load(limit_corpora=12)
    categories = {q.category for c in corpora for q in c.questions}
    assert len(categories) >= 4


def test_granularities_are_declared_and_differ() -> None:
    """LoCoMo scores turns, LongMemEval scores sessions. Comparing the two
    numbers directly is meaningless, so the difference must be inspectable."""
    assert LoCoMo("x").evidence_granularity == "turn"
    assert LongMemEval("x").evidence_granularity == "session"


# -- judge protocol ------------------------------------------------------------


def test_preference_questions_are_graded_against_a_rubric_not_a_fact() -> None:
    """The `answer` field for single-session-preference is a ~391-char rubric,
    not an answer. Grading it with "conveys the same fact as the reference" is a
    category error a short reply can never satisfy, and it cost us 60 points on
    that capability before it was caught."""
    prompt = judge_prompt(
        "single-session-preference", "q?", "the user prefers X", "A", official=True
    )
    assert "Rubric:" in prompt
    assert "Correct Answer:" not in prompt
    assert "does not need to reflect all the points" in prompt


def test_temporal_judge_carries_the_official_off_by_one_allowance() -> None:
    """Gold answers themselves say "8 days ... is also acceptable". Our old
    prompt said "Numbers must match", overriding the dataset on 133 questions."""
    prompt = judge_prompt("temporal-reasoning", "q?", "18", "19", official=True)
    assert "off-by-one" in prompt
    assert "Numbers must match" not in prompt


def test_knowledge_update_allows_restating_the_superseded_value() -> None:
    prompt = judge_prompt("knowledge-update", "q?", "Y", "was X, now Y", official=True)
    assert "previous information along with an updated answer" in prompt


def test_unknown_category_falls_back_instead_of_crashing_a_four_hour_run() -> None:
    prompt = judge_prompt("brand-new-type", "q?", "ref", "pred", official=True)
    assert "Correct Answer:" in prompt and "q?" in prompt


def test_every_official_template_consumes_all_three_fields() -> None:
    for category in OFFICIAL_JUDGE_PROMPTS:
        prompt = judge_prompt(category, "QQ", "RR", "PP", official=True)
        assert "QQ" in prompt and "RR" in prompt and "PP" in prompt
        assert "{" not in prompt, f"{category} left an unfilled placeholder"


def test_strict_grade_parsing_survives_the_incorrect_substring_trap() -> None:
    """ "INCORRECT" contains "CORRECT". Checking for the positive word alone
    grades every wrong answer as right — and the run still completes."""
    assert parse_grade("CORRECT", official=False) is True
    assert parse_grade("INCORRECT", official=False) is False


def test_official_grade_parsing_uses_yes_no() -> None:
    assert parse_grade("yes", official=True) is True
    assert parse_grade("Yes.", official=True) is True
    assert parse_grade("no", official=True) is False
    assert parse_grade("No, the response omits the date.", official=True) is False


def test_judge_token_budget_leaves_room_for_thinking() -> None:
    """gemini-2.5-pro spent 19 tokens reasoning before its first output token;
    an 8-token cap returned a truncated 'N' that parses as "no" and would have
    scored every answer wrong while looking like a working grader."""
    assert _JUDGE_MAX_TOKENS >= 256


# -- session truncation ----------------------------------------------------
#
# Measured on the real corpus: the derive path's 12,000-char limit cut 9,914
# of 23,867 sessions -- 41.5%, 34.7 million characters -- silently and
# mid-word. It was LOWER than the direct path's limit, while mapping one
# document per call and therefore having the most room in the system.


def test_a_short_session_is_untouched() -> None:
    from bench.harness import clip_session

    assert clip_session("a short session") == "a short session"


def test_a_long_session_is_marked_as_truncated() -> None:
    """A model cannot tell a cut session from one that simply ended."""
    from bench.harness import clip_session

    out = clip_session("word. " * 8000, limit=1000)
    assert "[... session truncated]" in out
    assert len(out) < 1200


def test_the_cut_lands_on_a_boundary_not_mid_word() -> None:
    from bench.harness import clip_session

    body = "This is a complete sentence. " * 200
    out = clip_session(body, limit=1000).replace("\n[... session truncated]", "")
    assert out.endswith("."), f"cut mid-sentence: {out[-40:]!r}"


def test_an_early_boundary_does_not_swallow_the_session() -> None:
    """Only a boundary in the last quarter counts.

    Otherwise a document with one paragraph break near the top loses almost
    everything it had.
    """
    from bench.harness import clip_session

    body = "Intro.\n\n" + ("x" * 5000)
    out = clip_session(body, limit=1000)
    assert len(out) > 900, "an early break truncated the whole session"


def test_the_derive_and_direct_limits_agree() -> None:
    """They disagreed, and the tighter one was on the path with more room."""
    import inspect

    from bench.harness import MAX_SESSION_CHARS

    from mapi.domain.synthesis.derive import derive_answer

    derive_limit = inspect.signature(derive_answer).parameters["max_doc_chars"].default
    assert derive_limit == MAX_SESSION_CHARS == 24_000
