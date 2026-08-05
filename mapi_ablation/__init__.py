"""Mapi representation ablation harness.

Decides one question: does a rendered 2D spatial canvas of a knowledge graph
deliver more usable relational context to a multimodal LLM than a serialized
text encoding of the *same* graph?

The load-bearing property of this package is arm parity: every arm receives an
identical node set and edge set, wrapped in a byte-identical prompt. Only the
encoding varies. Do not break that.
"""

__version__ = "0.1.0"
