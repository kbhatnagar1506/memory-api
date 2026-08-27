# Cloud Run deployment

Heroku suspended the app, and two of the three expensive pieces were already on
Google anyway — Cloud SQL holds the data and Vertex AI serves every embedding
and completion. This moves the last piece across, so the deployment stops
straddling two providers.

    ./deploy/cloudrun/deploy.sh

Idempotent. Every step is create-or-skip, so a re-run after a failure resumes.

## The part that matters most: no keys

`gcloud auth application-default login` is a **developer** credential. It
expired four times in a single working day, which is what it is designed to do,
and it is not what a service should run on.

Cloud Run attaches a **service account** to the revision and the container
draws credentials from the metadata server. There is no key file, nothing to
put in an environment variable, nothing to leak into shell history, and nothing
to rotate on a schedule. The service-account key currently outstanding for
rotation would not exist in this setup.

The identity is dedicated (`mapi-run`) rather than the default compute service
account, which carries project Editor — far more authority than an API server
should hold. It gets exactly three roles:

    roles/cloudsql.client            reach the database
    roles/aiplatform.user            call Vertex
    roles/secretmanager.secretAccessor   read its own two secrets

## Cloud SQL, without the proxy script

`bin/with-cloudsql` exists because Heroku dynos have no stable outbound
address, so reaching Cloud SQL over its public IP would mean authorising
`0.0.0.0/0` and relying on the password alone. Its docstring makes that
argument well.

Cloud Run provides the same thing natively: `--add-cloudsql-instances` mounts
the connection as a unix socket at `/cloudsql/INSTANCE`, authenticated by the
service account's IAM identity. So the connection string has no host or port:

    postgresql+asyncpg://USER:PASS@/DBNAME?host=/cloudsql/PROJECT:REGION:INSTANCE

The security property the script argues for is kept, not traded away — the
instance keeps an empty authorized-networks list and remains unreachable from
any address on the internet. The wrapper is simply redundant here.

## Migrations run as a Job, before traffic moves

Cloud Run has no release phase. Running `alembic upgrade head` at web-container
startup would race every replica against every other one, and a failed
migration would surface as a crash loop rather than a failed deploy.

So migrations are a **Cloud Run Job**, executed to completion first. `set -e`
means a failed migration aborts the script and the old revision keeps serving.

## Redis is dropped, deliberately, and it is a real tradeoff

Memorystore's smallest Basic tier is roughly $35/month against
`heroku-redis:mini`, for a component this app uses only to rate limit.

`redis_url` is already optional, and without it `core/ratelimit.py` falls back
to `InMemoryRateLimiter` — a per-process token bucket whose own docstring says
it is "correct for one replica, not for many", and production validation warns
when Redis is unset.

That warning is accurate and worth stating plainly:

  * at `--max-instances 1` (the default here) the in-memory limiter is
    **correct** — there is one process, so there is one bucket.
  * above one instance it becomes **approximate**: each replica enforces the
    full limit independently, so the effective ceiling is `limit x replicas`.

If you raise `MAX_INSTANCES`, either accept a soft limit or put the counter in
the Postgres you are already paying for. $35/month for a rate limiter on this
deployment is poor value; a wrong limit under load is a different question.

## Cost

us-central1 list prices, and worth checking against your own console — these
move.

| item | monthly |
|---|---|
| Cloud Run, `min-instances=0`, light traffic | **$0–5** (free tier: 180k vCPU-s, 2M requests) |
| Cloud Run, `min-instances=1`, 1 vCPU / 512MiB warm | **$45–65** |
| Artifact Registry (one image) | ~$0.10/GB |
| Cloud SQL | **already paying** — the dominant line item |
| Memorystore | **$0** (dropped) |

The default is `MIN_INSTANCES=0`, which is nearly free and pays for it with
cold starts — uvicorn plus model-client init, so a few seconds on the first
request after idle. `MIN_INSTANCES=1` is the difference between a demo that
feels broken and one that does not; it is one flag.

Cloud SQL is unpriced here because the instance tier could not be read at the
time of writing. It is likely 70–90% of the bill: roughly $9–15/month for
`db-f1-micro`, $25–35 for `db-g1-small`, $100+ for 2vCPU/8GB, plus ~$0.17/GB
storage, doubled for HA.

## What this does NOT solve

Cloud Run caps a request at 60 minutes, which is fine for the API and useless
for the benchmarks — a LongMemEval run is ~30 minutes of sustained work and the
laptop-closed problem needs a **Cloud Run Job** or a GCE VM instead. The 3.6GB
embedding cache also does not belong in a container image; it would live in GCS
and be pulled at job start. That is a separate piece of work.
