"""Ground truth must be derivable by traversal, unique, and stable.

The independent-recomputation tests below deliberately re-derive answers with
different code than queries.py uses, so a bug would have to be made twice in the
same direction to slip through.
"""

from __future__ import annotations

import pytest

from mapi_ablation.graph import DOMAINS, generate_graph
from mapi_ablation.queries import QUERY_TYPES, generate_queries

SEEDS = [0, 1, 2, 3, 4]
NODE_COUNTS = [20, 40, 80]


@pytest.fixture(scope="module", params=NODE_COUNTS)
def pairs(request):
    out = []
    for s in SEEDS:
        g = generate_graph(s, request.param)
        out.append((g, generate_queries(g)))
    return out


def test_every_answer_is_a_real_node(pairs):
    for g, qs in pairs:
        for q in qs:
            assert q.answer in g.ids


def test_answers_unique_within_type(pairs):
    for _, qs in pairs:
        for qtype in QUERY_TYPES:
            answers = [q.answer for q in qs.by_type(qtype)]
            assert len(set(answers)) == len(answers)


def test_question_never_names_its_own_answer(pairs):
    for _, qs in pairs:
        for q in qs:
            if q.qtype == "temporal_pair":
                continue  # the task is choosing between the two named nodes
            assert q.answer not in q.referenced


def test_temporal_pair_ground_truth(pairs):
    for g, qs in pairs:
        for q in qs.by_type("temporal_pair"):
            a, b = q.args
            assert q.answer == min((a, b), key=lambda x: g.node(x).t)
            assert q.t_gap == abs(g.node(a).t - g.node(b).t)


def test_temporal_pair_gaps_are_spread(pairs):
    """Rule (c): gaps must span a range, else adjacency effects are invisible."""
    for g, qs in pairs:
        gaps = sorted(q.t_gap for q in qs.by_type("temporal_pair"))
        assert gaps[0] <= g.n_nodes // 4
        assert gaps[-1] >= g.n_nodes // 2


def test_oldest_in_domain_ground_truth(pairs):
    for g, qs in pairs:
        for q in qs.by_type("oldest_in_domain"):
            domain = q.args[0]
            assert domain in DOMAINS
            members = g.nodes_in_domain(domain)
            assert q.answer == min(members, key=lambda n: n.t).id
            # uniqueness of the minimum
            assert sum(1 for m in members if m.t == g.node(q.answer).t) == 1


def test_root_cause_ground_truth_by_independent_walk(pairs):
    for g, qs in pairs:
        parent = {e.dst: e.src for e in g.rel_edges("caused")}
        for q in qs.by_type("root_cause"):
            cur = q.args[0]
            steps = 0
            while cur in parent:
                cur = parent[cur]
                steps += 1
                assert steps <= g.n_nodes
            assert cur == q.answer
            assert q.hops == steps
            assert steps >= 1


def test_cross_domain_effect_is_unique_and_crosses(pairs):
    for g, qs in pairs:
        for q in qs.by_type("cross_domain_effect"):
            src, domain = q.args
            hits = [x for x in g.caused_out(src) if g.node(x).domain == domain]
            assert hits == [q.answer]
            assert g.node(src).domain != domain
            assert q.crosses_domain is True


def test_contradiction_is_symmetric_and_same_domain(pairs):
    for g, qs in pairs:
        for q in qs.by_type("contradiction"):
            anchor = q.args[0]
            assert g.contradiction_of(anchor) == q.answer
            assert g.contradiction_of(q.answer) == anchor
            assert g.node(anchor).domain == g.node(q.answer).domain


def test_supersede_respects_direction(pairs):
    for g, qs in pairs:
        for q in qs.by_type("supersede"):
            anchor, direction = q.args
            if direction == "forward":
                # "which node supersedes anchor" -> a strictly newer node
                assert g.superseded_by(anchor) == q.answer
                assert g.node(q.answer).t > g.node(anchor).t
            else:
                # "which node does anchor supersede" -> a strictly older node
                assert g.supersedes_what(anchor) == q.answer
                assert g.node(q.answer).t < g.node(anchor).t


def test_qids_stable_across_regeneration(pairs):
    for g, qs in pairs:
        again = generate_queries(generate_graph(g.seed, g.n_nodes))
        assert [q.qid for q in again] == [q.qid for q in qs]
        assert [q.answer for q in again] == [q.answer for q in qs]


def test_qids_unique_and_typed(pairs):
    for _, qs in pairs:
        ids = [q.qid for q in qs]
        assert len(set(ids)) == len(ids)
        assert {q.qtype for q in qs} == set(QUERY_TYPES)


def test_capacity_is_reported_not_padded(pairs):
    for _, qs in pairs:
        for qtype, cap in qs.capacity.items():
            assert cap["sampled"] == min(cap["requested"], cap["available"])
            assert cap["sampled"] == len(qs.by_type(qtype))
