"""Construct similarity instead of deriving it from text.

THE PROBLEM THIS SOLVES. Every retrieval claim worth testing is a claim about
what happens *given a similarity structure*: does the gold fact survive eight
distractors at cosine 0.61, does the vector arm reach a memory sharing no words
with the query, does a near-duplicate flood crowd out the answer. To test any
of that you have to be able to say "put this memory at cosine 0.82 from the
query".

You cannot say that with `DeterministicEmbedder`. It is blake2b feature hashing
over word unigrams and character 3/4-grams -- a real *lexical* embedder and a
non-semantic one. Cosine between two of its vectors is an emergent property of
shared n-grams, so you can observe a similarity but never request one, and two
texts sharing no character trigrams are near-orthogonal by construction. Half
the interesting tests are unwritable and the other half are accidents.

So these helpers build vectors directly, from the geometry up. `at_cosine(0.82,
off=3)` returns a unit vector at exactly cosine 0.82 from the query direction.
The assertion then reads "gold at 0.82 must outrank three distractors at 0.61",
which is a statement about the pipeline rather than about a hash function, and
its failure message names the geometry that broke.

WHAT THE EMBEDDER IS FOR, ONCE THIS EXISTS. `DeterministicEmbedder` keeps its
own contract tests, plus one test asserting that it is lexical and not semantic
(`cos(car, automobile) < cos(car, cart)`) -- a genuine regression guard, since
swapping the hash changes retrieval behaviour everywhere. What it stops being
is the mechanism by which similarity is specified.

THE CONVENTIONS, all load-bearing:

  * **Axis 0 is the query direction.** `at_cosine` measures from it and
    distractors take `off >= 1`. With `DIMS = 128` that leaves 127 orthogonal
    off-directions, which is enough for the widest distractor sweep here
    (d = 64).
  * **pgvector stores float32.** Assert cosines with `abs=1e-6`, not
    `1e-15`, and keep intended bands at least 0.02 apart so that replaying a
    corpus against Postgres cannot reorder it through rounding.
  * **`EmbeddingProvider._validate` re-normalizes** every vector it returns and
    rejects zero and non-finite ones (`embeddings/base.py`). Everything here
    returns a finite unit vector, so a constructed vector survives the
    provider contract unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Vector = list[float]

#: Matches `TEST_DIMENSIONS` in tests/conftest.py and the width the conformance
#: suite expects of a real database (`MAPI_TEST_DIMENSIONS`, default 128).
DIMS = 128

#: The query direction. Reserved: no distractor may use it.
ANCHOR = 0

#: Comparison tolerance. Sized for float32 round-tripping through pgvector
#: rather than for float64 arithmetic, so the same assertion holds on both
#: backends.
COSINE_TOLERANCE = 1e-6


def axis(index: int, dims: int = DIMS) -> Vector:
    """A unit basis vector. `cos(axis(i), axis(j)) == 0` exactly for i != j."""
    if not 0 <= index < dims:
        raise ValueError(f"axis {index} outside [0, {dims})")
    vector = [0.0] * dims
    vector[index] = 1.0
    return vector


def at_cosine(cosine: float, *, off: int = 1, dims: int = DIMS) -> Vector:
    """A unit vector at exactly `cosine` from the anchor.

        v = cos(t)*e_ANCHOR + sin(t)*e_off,   t = acos(cosine)
        <v, e_ANCHOR> = cos(t) = cosine

    Exact to float rounding, because the identity is trigonometric rather than
    fitted. `off` selects which orthogonal direction carries the remainder, so
    two calls with different `off` at the same cosine are equidistant from the
    query and distinguishable from each other.
    """
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"cosine {cosine} outside [-1, 1]")
    if off == ANCHOR:
        raise ValueError("off must not be the anchor axis; distractors start at 1")
    if not 0 <= off < dims:
        raise ValueError(f"off {off} outside [0, {dims})")

    angle = math.acos(max(-1.0, min(1.0, cosine)))
    vector = [0.0] * dims
    vector[ANCHOR] = math.cos(angle)
    vector[off] = math.sin(angle)
    return vector


def band(cosine: float, count: int, *, first_off: int = 1, dims: int = DIMS) -> list[Vector]:
    """`count` vectors all at `cosine` from the anchor, each on its own axis.

    THE THING TO KNOW BEFORE USING THIS: band members are at cosine `cosine`
    from the anchor and at exactly `cosine ** 2` from EACH OTHER. A band at
    0.8 is an internally-0.64 cloud, not a set of mutually distant points. A
    test that needs members far apart is fine; a test that needs them close
    (near-duplicate flooding) wants `cone`, which controls both distances.
    """
    if count < 0:
        raise ValueError("count must not be negative")
    if first_off + count > dims:
        raise ValueError(
            f"{count} vectors from off={first_off} needs {first_off + count} dims, have {dims}"
        )
    return [at_cosine(cosine, off=first_off + i, dims=dims) for i in range(count)]


def cone(
    cosine: float, inner: float, count: int, *, first_off: int = 2, dims: int = DIMS
) -> list[Vector]:
    """`count` vectors at `cosine` from the anchor and ~`inner` from each other.

    Two-level construction, because a band cannot express "close to the query
    AND close to each other":

        v_i = cos(t)*e_ANCHOR + sin(t)*(cos(u)*e_1 + sin(u)*e_{first_off+i})

    The shared `e_1` component is what pulls members together; `u` sets how
    much. Pairwise cosine is `cosine**2 + sin(t)**2 * cos(u)**2`, which is what
    `inner` is solved for. This is the near-duplicate flood: many memories that
    all look like the answer and like each other.
    """
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"cosine {cosine} outside [-1, 1]")
    if count < 0:
        raise ValueError("count must not be negative")
    if first_off + count > dims:
        raise ValueError(f"cone of {count} from off={first_off} exceeds {dims} dims")

    outer = math.acos(max(-1.0, min(1.0, cosine)))
    sin_outer = math.sin(outer)
    if sin_outer <= COSINE_TOLERANCE:
        # The anchor direction itself: every member is the same vector, and
        # `inner` cannot be honoured. Say so rather than divide by ~0.
        raise ValueError("cone at cosine 1.0 is a single point; use at_cosine")

    # Solve inner = cosine**2 + sin_outer**2 * cos(u)**2 for u.
    shared_sq = (inner - cosine * cosine) / (sin_outer * sin_outer)
    if not 0.0 <= shared_sq <= 1.0:
        raise ValueError(
            f"inner {inner} unreachable at cosine {cosine}; "
            f"attainable range is [{cosine * cosine:.4f}, 1.0]"
        )
    shared = math.sqrt(shared_sq)
    unique = math.sqrt(max(0.0, 1.0 - shared_sq))

    out: list[Vector] = []
    for i in range(count):
        vector = [0.0] * dims
        vector[ANCHOR] = math.cos(outer)
        vector[1] = sin_outer * shared
        vector[first_off + i] = sin_outer * unique
        out.append(vector)
    return out


def nudge(vector: Sequence[float], epsilon: float, *, off: int) -> Vector:
    """Perturb by a controlled amount, re-normalized.

    For tie-break determinism: two memories whose scores differ by 1e-9 must
    still order the same way on every run and on both backends. A test that
    needs "almost identical" needs to say how almost.
    """
    if off == ANCHOR:
        raise ValueError("nudge off must not be the anchor axis")
    out = list(vector)
    if not 0 <= off < len(out):
        raise ValueError(f"off {off} outside [0, {len(out)})")
    out[off] += epsilon
    norm = math.sqrt(sum(x * x for x in out))
    if norm == 0.0:
        raise ValueError("nudge produced a zero vector")
    return [x / norm for x in out]


def cosine_of(left: Sequence[float], right: Sequence[float]) -> float:
    """Plain cosine, for asserting that a construction did what it claimed."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0.0 or rn == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (ln * rn)))


def brute_force_topk(
    query: Sequence[float], corpus: Mapping[str, Sequence[float]], k: int
) -> list[str]:
    """Exact kNN. The oracle that separates embedder error from index error.

    Two jobs. Against `InMemoryStore` it proves a test's expectation is a
    property of the geometry and not of the scan; the tie-break is `(-score,
    id)` to match `InMemoryStore.vector_search` exactly, so a disagreement is a
    real disagreement. Against Postgres it is the recall@k reference the HNSW
    arm is measured against -- an approximate index is allowed to differ, and
    this is what quantifies by how much.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    scored = [(cosine_of(query, vector), mid) for mid, vector in corpus.items()]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [mid for _score, mid in scored[:k]]


__all__ = [
    "ANCHOR",
    "COSINE_TOLERANCE",
    "DIMS",
    "Vector",
    "at_cosine",
    "axis",
    "band",
    "brute_force_topk",
    "cone",
    "cosine_of",
    "nudge",
]
