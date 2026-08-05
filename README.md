# mapi-ablation

Does a rendered 2D spatial canvas of a knowledge graph deliver more usable
relational context to a multimodal LLM than a serialized text encoding of the
**same** graph? This repo is the experiment that answers that, not the product.

All four arms receive an identical node set and edge set inside a byte-identical
prompt wrapper. Only the encoding varies. That property is enforced by
`tests/test_arm_parity.py` and must survive every change.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                      # 107 tests, no API needed
.venv/bin/python -m mapi_ablation.cli example      # one graph in all 4 arms -> runs/_example/
.venv/bin/python -m mapi_ablation.cli pretest --dry-run   # build the 40 gate canvases
gcloud auth application-default login              # Gemini/Claude via Vertex; or set API keys
.venv/bin/python -m mapi_ablation.cli pretest --verify-resize   # Stage 0 gate + real resize measurement
```

`run`, `stability`, `report` and `sweep` are gated on the Stage 0 pretest and
exit with a message until it has been reviewed. See `DECISIONS.md` for the
tradeoffs and `config/default.yaml` for the matrix.
