
## Erasure, context v2, pgvector 0.8 (2026-08-08)

- ⚑ **`erase_memory` is distinct from `delete_memory`, and both exist on
  purpose.** Delete preserves version history (audit default); erase purges the
  live row, chunks, edges AND the full version history in one transaction, so
  `?as_of=` cannot resurrect the content — an erasure that survives
  point-in-time reads is not an erasure. The attestation carries the content
  hash (proves WHICH content died without retaining it), purge counts, and
  derived-memory ids. Erase is idempotent: a retried compliance request must
  not fail.
- **System time is not spoofable through the public path.** The replay demo
  originally tried to pin `valid_from` by re-upserting with `now=t0`; the
  same-version in-place-correction rule correctly kept the original recorded
  time. The demo now reads real recorded timestamps. This is the bitemporal
  model defending itself.
- **Bench context v2 merges overlapping windows** (each turn emitted once, date
  headers only on change). Dedup-only, so quality risk is zero by construction;
  `--compress` (term-overlap pruning of neighbours) is a separate measured
  flag because it CAN cost accuracy. Alternative rejected: LLMLingua-2 — adds
  a torch dependency and 10-50ms in the read path for parity-at-best on clean
  retrieval, and per-query dynamic compression breaks prompt-cache identity.
- **pgvector 0.8 iterative scans (`relaxed_order`) enabled per vector search**,
  capability-probed once, rollback-on-failure so pgvector < 0.8 degrades
  cleanly. Correctness first: filtered HNSW pre-0.8 silently under-returns.
  `halfvec` deferred: schema migration is unverifiable without a live DB in
  this environment; CI exercises the runtime SETs.
