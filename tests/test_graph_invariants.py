"""The five spec invariants, plus ID/time decorrelation.

These are what make every generated query single-answered. If one of these
starts failing, the experiment is invalid, not merely buggy.
"""

from __future__ import annotations

import pytest

from mapi_ablation.graph import (
    DOMAINS,
    ID_TIME_RHO_MAX,
    generate_graph,
    id_time_rho,
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
NODE_COUNTS = [20, 40, 80]


@pytest.fixture(scope="module", params=NODE_COUNTS)
def graphs(request):
    return [generate_graph(s, request.param) for s in SEEDS]


def test_1_unique_t_and_temporal_edge_direction(graphs):
    for g in graphs:
        ts = [n.t for n in g.nodes]
        assert sorted(ts) == list(range(1, g.n_nodes + 1))
        t = {n.id: n.t for n in g.nodes}
        for e in g.edges:
            if e.rel == "caused":
                assert t[e.src] < t[e.dst]
            elif e.rel == "supersedes":
                assert t[e.src] > t[e.dst]


def test_1b_dates_strictly_increase_with_t(graphs):
    for g in graphs:
        by_t = sorted(g.nodes, key=lambda n: n.t)
        for a, b in zip(by_t, by_t[1:]):
            assert a.date < b.date


def test_2_pair_relations_are_node_disjoint(graphs):
    for g in graphs:
        contra, srcs, dsts = [], [], []
        for e in g.edges:
            if e.rel == "contradicts":
                contra += [e.src, e.dst]
            elif e.rel == "supersedes":
                srcs.append(e.src)
                dsts.append(e.dst)
        assert len(set(contra)) == len(contra)
        assert len(set(srcs)) == len(srcs)
        assert len(set(dsts)) == len(dsts)
        assert not set(contra) & (set(srcs) | set(dsts))
        assert 4 <= len(g.rel_edges("contradicts")) <= 6
        assert 4 <= len(g.rel_edges("supersedes")) <= 6


def test_2b_contradicts_pairs_share_a_domain(graphs):
    for g in graphs:
        dom = {n.id: n.domain for n in g.nodes}
        for e in g.rel_edges("contradicts"):
            assert dom[e.src] == dom[e.dst]


def test_3_chains_disjoint_and_backward_traversal_unique(graphs):
    for g in graphs:
        flat = [n for ch in g.chains for n in ch]
        assert len(set(flat)) == len(flat)
        assert 6 <= len(g.chains) <= 8
        assert all(3 <= len(ch) <= 5 for ch in g.chains)
        for ch in g.chains:
            assert g.caused_in(ch[0]) == []
            for member in ch:
                assert g.root_cause(member) == ch[0]
        for nid in g.ids:
            assert len(g.caused_in(nid)) <= 1
            assert len(g.caused_out(nid)) <= 1


def test_3b_half_the_chains_cross_a_domain_boundary(graphs):
    for g in graphs:
        dom = {n.id: n.domain for n in g.nodes}
        crossing = sum(
            1 for ch in g.chains if any(dom[a] != dom[b] for a, b in zip(ch, ch[1:]))
        )
        assert crossing >= (len(g.chains) + 1) // 2


def test_4_domains_populated_with_unique_earliest(graphs):
    for g in graphs:
        for d in DOMAINS:
            members = g.nodes_in_domain(d)
            assert len(members) >= 3
            earliest = min(m.t for m in members)
            assert sum(1 for m in members if m.t == earliest) == 1


def test_5_same_seed_is_byte_identical(graphs):
    for g in graphs:
        again = generate_graph(g.seed, g.n_nodes)
        assert again.to_canonical_json() == g.to_canonical_json()
        assert again.fingerprint() == g.fingerprint()


def test_5b_different_seeds_differ(graphs):
    prints = {g.fingerprint() for g in graphs}
    assert len(prints) == len(graphs)


def test_6_node_ids_do_not_encode_time_order(graphs):
    """If they did, temporal queries would be answerable without the payload."""
    for g in graphs:
        assert abs(id_time_rho(g)) <= ID_TIME_RHO_MAX


def test_labels_unique_within_graph(graphs):
    for g in graphs:
        labels = [n.label for n in g.nodes]
        assert len(set(labels)) == len(labels)
