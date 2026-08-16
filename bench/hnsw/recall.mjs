// Measure pgvector HNSW recall against exact search, at the parameters this
// project actually ships: m=16, ef_construction=64, and ef_search before/after
// the fix. Real text-embedding-004 vectors, so the data distribution is the
// production one rather than random noise.
import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import fs from 'fs'

const buf = fs.readFileSync(new URL('./vectors.f32', import.meta.url))
const N = buf.readInt32LE(0), Q = buf.readInt32LE(4), DIM = buf.readInt32LE(8)
const floats = new Float32Array(buf.buffer, buf.byteOffset + 12, (N + Q) * DIM)
const take = (i) => Array.from(floats.subarray(i * DIM, (i + 1) * DIM))
const corpus = Array.from({ length: N }, (_, i) => take(i))
const queries = Array.from({ length: Q }, (_, i) => take(N + i))
const lit = (v) => '[' + v.join(',') + ']'

const db = await PGlite.create({ extensions: { vector } })
await db.exec(`CREATE EXTENSION IF NOT EXISTS vector;`)
await db.exec(`CREATE TABLE chunks (id serial primary key, embedding vector(${DIM}));`)

console.log(`inserting ${corpus.length} real vectors (dim=${DIM}) ...`)
const BATCH = 500
for (let i = 0; i < corpus.length; i += BATCH) {
  const slice = corpus.slice(i, i + BATCH)
  const values = slice.map((v) => `('${lit(v)}')`).join(',')
  await db.exec(`INSERT INTO chunks (embedding) VALUES ${values};`)
  if ((i / BATCH) % 20 === 0) process.stdout.write(`\r  ${i + slice.length}/${corpus.length}`)
}
console.log('\nbuilding HNSW index (m=16, ef_construction=64) ...')
const t0 = Date.now()
await db.exec(`CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`)
console.log(`  built in ${((Date.now() - t0) / 1000).toFixed(1)}s`)
await db.exec(`ANALYZE chunks;`)

async function topk(q, k, { exact, ef }) {
  if (exact) {
    await db.exec(`SET LOCAL enable_indexscan = off; SET LOCAL enable_bitmapscan = off;`)
  } else {
    await db.exec(`SET LOCAL enable_indexscan = on; SET hnsw.ef_search = ${ef};`)
  }
  const r = await db.query(
    `SELECT id FROM chunks ORDER BY embedding <=> $1 LIMIT ${k}`, [lit(q)],
  )
  return r.rows.map((x) => x.id)
}

// Confirm the index is actually being used when we expect it to be.
await db.exec(`SET hnsw.ef_search = 40;`)
const plan = await db.query(`EXPLAIN SELECT id FROM chunks ORDER BY embedding <=> '${lit(queries[0])}' LIMIT 60`)
console.log('plan(ann):', plan.rows.map((r) => r['QUERY PLAN']).join(' | ').slice(0, 120))

const Ks = [10, 60]          // 10 = API default limit; 60 = what the pipeline fetches
const EFs = [40, 120, 200]   // 40 = old default; 120 = the fix at k=60; 200 = headroom
console.log(`\nrecall over ${queries.length} real queries\n`)
console.log('  k    ef_search   recall@k   full_recall(all k present)')
for (const k of Ks) {
  const exactSets = []
  for (const q of queries) exactSets.push(await topk(q, k, { exact: true }))
  for (const ef of EFs) {
    let overlap = 0, full = 0
    for (let i = 0; i < queries.length; i++) {
      const got = await topk(queries[i], k, { exact: false, ef })
      const want = new Set(exactSets[i])
      const hit = got.filter((id) => want.has(id)).length
      overlap += hit / k
      if (hit === k) full += 1
    }
    const rec = overlap / queries.length
    const fr = full / queries.length
    console.log(`  ${String(k).padEnd(4)} ${String(ef).padEnd(11)} ${(rec * 100).toFixed(2)}%     ${(fr * 100).toFixed(1)}%`)
  }
}
await db.close()
