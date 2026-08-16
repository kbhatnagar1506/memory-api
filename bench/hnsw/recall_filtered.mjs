// FILTERED HNSW recall -- the production shape. Every real query is scoped to
// an org and a space, and a filter is where HNSW is weakest: the index knows
// nothing about `space_id`, so it walks the graph by distance and the filter
// throws away whatever does not match. The scarcer the matching rows, the
// deeper it has to walk to fill k.
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import fs from 'fs'

const buf = fs.readFileSync(new URL('./vectors.f32', import.meta.url))
const N = buf.readInt32LE(0), Q = buf.readInt32LE(4), DIM = buf.readInt32LE(8)
const floats = new Float32Array(buf.buffer, buf.byteOffset + 12, (N + Q) * DIM)
const take = (i) => Array.from(floats.subarray(i * DIM, (i + 1) * DIM))
const corpus = Array.from({ length: N }, (_, i) => take(i))
const queries = Array.from({ length: Q }, (_, i) => take(N + i)).slice(0, 100)
const lit = (v) => '[' + v.join(',') + ']'

// Space sizes as a share of the corpus. A real deployment is mostly small
// spaces inside a big table, which is the left end of this range.
const SPACES = [
  { id: 'sp_001', share: 0.01 },
  { id: 'sp_005', share: 0.05 },
  { id: 'sp_025', share: 0.25 },
  { id: 'sp_050', share: 0.50 },
]
const assign = []
let cursor = 0
for (const s of SPACES) {
  const n = Math.round(N * s.share)
  for (let i = 0; i < n; i++) assign[cursor++] = s.id
}
while (cursor < N) assign[cursor++] = 'sp_rest'

const db = await PGlite.create({ extensions: { vector } })
await db.exec(`CREATE EXTENSION IF NOT EXISTS vector;`)
await db.exec(`CREATE TABLE chunks (id serial primary key, space_id text, embedding vector(${DIM}));`)

console.log(`inserting ${N} vectors across ${SPACES.length + 1} spaces ...`)
const BATCH = 500
for (let i = 0; i < N; i += BATCH) {
  const vals = corpus.slice(i, i + BATCH)
    .map((v, j) => `('${assign[i + j]}','${lit(v)}')`).join(',')
  await db.exec(`INSERT INTO chunks (space_id, embedding) VALUES ${vals};`)
}
await db.exec(`CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`)
await db.exec(`CREATE INDEX ON chunks (space_id);`)   // production has this too
await db.exec(`ANALYZE chunks;`)
console.log('indexes built\n')

const K = 60
async function run(q, space, { exact, ef, iter }) {
  if (exact) {
    await db.exec(`SET enable_indexscan = off; SET enable_bitmapscan = off; SET enable_seqscan = on;`)
  } else {
    await db.exec(`SET enable_indexscan = on; SET enable_bitmapscan = on; SET enable_seqscan = off; SET hnsw.ef_search = ${ef};`)
    try { await db.exec(`SET hnsw.iterative_scan = '${iter}';`) } catch { /* pre-0.8 */ }
  }
  const r = await db.query(
    `SELECT id FROM chunks WHERE space_id = $1 ORDER BY embedding <=> $2 LIMIT ${K}`,
    [space, lit(q)],
  )
  return r.rows.map((x) => x.id)
}

// What plan does a filtered ANN query actually get?
await db.exec(`SET enable_seqscan = off; SET hnsw.ef_search = 180;`)
for (const s of [SPACES[0], SPACES[3]]) {
  const p = await db.query(
    `EXPLAIN SELECT id FROM chunks WHERE space_id = '${s.id}' ORDER BY embedding <=> '${lit(queries[0])}' LIMIT ${K}`)
  console.log(`plan @ ${(s.share * 100).toFixed(0)}% selectivity:`,
    p.rows.map((r) => r['QUERY PLAN'].trim()).join(' | ').slice(0, 150))
}

console.log(`\nk=${K}, ${queries.length} queries, filtered by space_id\n`)
console.log('  space share   ef    iterative      recall@60   full_recall   rows')
for (const s of SPACES) {
  const exactSets = []
  for (const q of queries) exactSets.push(await run(q, s.id, { exact: true }))
  const attainable = exactSets[0].length
  for (const [ef, iter] of [[40, 'off'], [40, 'relaxed_order'], [180, 'relaxed_order']]) {
    let ov = 0, full = 0, rows = 0
    for (let i = 0; i < queries.length; i++) {
      const got = await run(queries[i], s.id, { exact: false, ef, iter })
      const want = new Set(exactSets[i])
      const hit = got.filter((id) => want.has(id)).length
      const denom = Math.max(exactSets[i].length, 1)
      ov += hit / denom
      if (hit === exactSets[i].length) full++
      rows += got.length
    }
    console.log(`  ${String((s.share * 100).toFixed(0) + '%').padEnd(13)} ${String(ef).padEnd(5)} ${iter.padEnd(14)} ${(ov / queries.length * 100).toFixed(2)}%     ${(full / queries.length * 100).toFixed(1)}%       ${(rows / queries.length).toFixed(1)}/${attainable}`)
  }
}
await db.close()
