"""Arm parity: the property the whole experiment rests on.

Every arm must encode the identical node set and edge set, and the prompt
wrapper must be byte-identical across arms except for the payload and the one
sentence naming the payload type. If either fails, a measured difference between
arms could come from selection rather than representation, and the result would
be worthless regardless of which way it pointed.
"""

from __future__ import annotations

import pytest

from mapi_ablation.arms import ARMS, build, normalize_edge, recover
from mapi_ablation.graph import generate_graph
from mapi_ablation.prompts import CONDITIONS, build_prompt, wrapper_skeleton

SEEDS = [0, 1, 2]
NODE_COUNTS = [20, 40]


def expected_sets(g):
    nodes = {(n.id, n.label, n.type, n.domain, n.date) for n in g.nodes}
    edges = {normalize_edge(e.src, e.dst, e.rel) for e in g.edges}
    return nodes, edges


@pytest.fixture(scope="module", params=NODE_COUNTS)
def payload_sets(request):
    out = []
    for s in SEEDS:
        g = generate_graph(s, request.param)
        payloads = {name: build(name, g) for name in ARMS}
        out.append((g, payloads))
    return out


def test_all_four_arms_exist():
    assert set(ARMS) == {"prose", "json", "layout_text", "canvas"}


def test_every_arm_recovers_the_same_node_set(payload_sets):
    for g, payloads in payload_sets:
        want, _ = expected_sets(g)
        for name, p in payloads.items():
            got, _ = recover(p)
            assert got == want, (
                f"arm {name} node set differs: "
                f"missing={want - got} extra={got - want}"
            )


def test_every_arm_recovers_the_same_edge_set(payload_sets):
    for g, payloads in payload_sets:
        _, want = expected_sets(g)
        for name, p in payloads.items():
            _, got = recover(p)
            assert got == want, (
                f"arm {name} edge set differs: "
                f"missing={want - got} extra={got - want}"
            )


def test_arms_agree_with_each_other_pairwise(payload_sets):
    for _, payloads in payload_sets:
        sets = {name: recover(p) for name, p in payloads.items()}
        reference = sets["json"]
        for name, s in sets.items():
            assert s == reference, f"arm {name} disagrees with arm json"


def test_canvas_never_truncates_a_label(payload_sets):
    """A truncated label is dropped information, which breaks parity."""
    for g, payloads in payload_sets:
        depicted = {t[0]: t[1] for t in payloads["canvas"].meta["_nodes"]}
        for n in g.nodes:
            assert depicted[n.id] == n.label


def test_wrapper_is_identical_across_arms(payload_sets):
    for condition in CONDITIONS:
        skeleton = wrapper_skeleton(condition, "Which node contradicts N15?")
        for _, payloads in payload_sets:
            for name, p in payloads.items():
                built = build_prompt(p, "Which node contradicts N15?", condition)
                text = built.text.replace(p.descriptor, "<DESCRIPTOR>")
                if p.kind == "text":
                    text = text.replace(p.text, "<PAYLOAD>")
                else:
                    text = text.replace("[image above]", "<PAYLOAD>")
                assert text == skeleton, f"arm {name} wrapper diverged ({condition})"


def test_only_the_image_arm_carries_an_image(payload_sets):
    for _, payloads in payload_sets:
        for name, p in payloads.items():
            built = build_prompt(p, "Which node contradicts N15?", "cold")
            assert built.has_image == (name == "canvas")


def test_primed_adds_text_and_cold_does_not(payload_sets):
    _, payloads = payload_sets[0]
    for p in payloads.values():
        cold = build_prompt(p, "Q?", "cold").text
        primed = build_prompt(p, "Q?", "primed").text
        assert len(primed) > len(cold)
        assert "horizontal axis is time" in primed
        assert "horizontal axis is time" not in cold


def test_payload_hash_is_stable_and_arm_specific(payload_sets):
    g, payloads = payload_sets[0]
    again = {name: build(name, g) for name in ARMS}
    for name, p in payloads.items():
        assert again[name].payload_hash() == p.payload_hash()
    hashes = {p.payload_hash() for p in payloads.values()}
    assert len(hashes) == len(payloads)


def test_canvas_render_is_collision_free_at_measured_configs(payload_sets):
    """A collision means two node boxes overlap, which is a broken canvas."""
    for _, payloads in payload_sets:
        assert payloads["canvas"].meta["collisions"] == []


def test_every_pretest_probe_renders_cleanly():
    """No probe may have overlapping boxes or an occluded connector.

    edge_type-04 and -19 originally placed two same-band nodes one date-step
    apart: the boxes touched, the dashed `contradicts` connector had zero
    visible span, and the model scored it `caused`. That reads as a grammar
    failure but is a broken probe.
    """
    import math

    from mapi_ablation.pretest import PROBES
    from mapi_ablation.render.layout import RenderConfig, compute_layout

    for p in PROBES:
        p.png()  # asserts no truncation and no collisions
        if not p.graph.edges:
            continue
        lay = compute_layout(p.graph, RenderConfig(width=1540, subrows=1))
        e = p.graph.edges[0]
        a, b = lay.placements[e.src], lay.placements[e.dst]
        dx = max(abs(a.cx - b.cx) - lay.geom.node_w, 0)
        span = math.hypot(dx, abs(a.cy - b.cy))
        assert span > 40, f"{p.pid}: connector visible span only {span:.0f}px"
