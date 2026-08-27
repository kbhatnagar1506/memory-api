#!/usr/bin/env bash
# Deploy mapi to Cloud Run, against the Cloud SQL instance it already uses.
#
# Idempotent: every step is create-or-skip, so re-running after a failure
# resumes rather than duplicating. Nothing here is destructive.
#
#   ./deploy/cloudrun/deploy.sh                 # build, migrate, deploy
#   SKIP_BUILD=1 ./deploy/cloudrun/deploy.sh    # redeploy the current image
#
# Read deploy/cloudrun/README.md first -- it explains the choices this script
# only executes.
set -euo pipefail

PROJECT="${PROJECT:-patchguard-reakon}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-mapi}"
SQL_INSTANCE="${SQL_INSTANCE:-patchguard-reakon:us-central1:mapi-db}"
DB_NAME="${DB_NAME:-mapi}"
DB_USER="${DB_USER:-mapi}"
SA="${SA:-mapi-run}"
SA_EMAIL="${SA}@${PROJECT}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/mapi/api"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "preflight"
gcloud auth application-default print-access-token >/dev/null 2>&1 \
  || { echo "ADC is not valid. Run: gcloud auth application-default login"; exit 1; }
gcloud config set project "$PROJECT" >/dev/null

say "enabling APIs (no-op if already on)"
gcloud services enable \
  run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com \
  sqladmin.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com \
  --project "$PROJECT"

say "service account"
# A dedicated identity rather than the default compute SA, which is
# project-editor by default -- far more authority than an API server needs.
gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SA" \
       --display-name "mapi Cloud Run runtime" --project "$PROJECT"

# Exactly three roles: reach the database, call Vertex, read its own secrets.
# No key is ever created -- Cloud Run hands the container credentials from the
# metadata server, which is the whole reason this deployment has no key to leak
# or rotate.
for ROLE in roles/cloudsql.client roles/aiplatform.user roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:${SA_EMAIL}" --role "$ROLE" \
    --condition=None --quiet >/dev/null
done
echo "granted: cloudsql.client, aiplatform.user, secretmanager.secretAccessor"

say "secrets"
ensure_secret() {  # name, prompt
  if gcloud secrets describe "$1" --project "$PROJECT" >/dev/null 2>&1; then
    echo "  $1: exists"
  else
    echo "  $1: creating"
    read -rsp "    enter value for $1 ($2): " V; echo
    printf '%s' "$V" | gcloud secrets create "$1" --data-file=- --project "$PROJECT" >/dev/null
  fi
}
ensure_secret mapi-db-password "the Cloud SQL '${DB_USER}' password"
ensure_secret mapi-api-key-pepper "any long random string; rotating it invalidates every API key"

say "artifact registry"
gcloud artifacts repositories describe mapi --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud artifacts repositories create mapi --repository-format docker \
       --location "$REGION" --description "mapi images" --project "$PROJECT"

if [[ -z "${SKIP_BUILD:-}" ]]; then
  say "building image (Cloud Build, so no local Docker needed)"
  gcloud builds submit --tag "${IMAGE}:latest" --project "$PROJECT" .
else
  echo "SKIP_BUILD set -- reusing ${IMAGE}:latest"
fi

# The database URL uses a UNIX SOCKET, not a host and port. Cloud Run mounts
# the Cloud SQL connection at /cloudsql/INSTANCE, and the proxy sidecar
# bin/with-cloudsql exists for is unnecessary here -- the platform provides it.
# The instance keeps an empty authorized-networks list, so the security
# property that script argues for is preserved rather than traded away.
DB_URL="postgresql+asyncpg://${DB_USER}:PASSWORD_PLACEHOLDER@/${DB_NAME}?host=/cloudsql/${SQL_INSTANCE}"

COMMON_ENV="MAPI_ENVIRONMENT=production"
COMMON_ENV="${COMMON_ENV},MAPI_STORE_BACKEND=postgres"
COMMON_ENV="${COMMON_ENV},MAPI_EMBEDDING_BACKEND=gemini"
COMMON_ENV="${COMMON_ENV},MAPI_SYNTHESIS_BACKEND=gemini"
COMMON_ENV="${COMMON_ENV},MAPI_RERANK_BACKEND=heuristic"
COMMON_ENV="${COMMON_ENV},GOOGLE_CLOUD_PROJECT=${PROJECT}"
COMMON_ENV="${COMMON_ENV},MAPI_DB_USER=${DB_USER}"
COMMON_ENV="${COMMON_ENV},MAPI_DB_NAME=${DB_NAME}"
COMMON_ENV="${COMMON_ENV},MAPI_SQL_INSTANCE=${SQL_INSTANCE}"

say "migrations (a Job, run to completion BEFORE traffic moves)"
# Cloud Run has no release phase. Running `alembic upgrade head` from the web
# container's startup would race every replica against every other one; a Job
# runs exactly once and fails loudly, and the deploy below does not happen if
# it fails.
if ! gcloud run jobs describe mapi-migrate --region "$REGION" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud run jobs create mapi-migrate \
    --image "${IMAGE}:latest" --region "$REGION" --project "$PROJECT" \
    --service-account "$SA_EMAIL" \
    --set-cloudsql-instances "$SQL_INSTANCE" \
    --set-env-vars "$COMMON_ENV" \
    --set-secrets "MAPI_DB_PASSWORD=mapi-db-password:latest,MAPI_API_KEY_PEPPER=mapi-api-key-pepper:latest" \
    --command sh --args "-c,export MAPI_DATABASE_URL=\"postgresql+asyncpg://\$MAPI_DB_USER:\$MAPI_DB_PASSWORD@/\$MAPI_DB_NAME?host=/cloudsql/\$MAPI_SQL_INSTANCE\"; alembic upgrade head" \
    --max-retries 0 --task-timeout 10m
else
  gcloud run jobs update mapi-migrate \
    --image "${IMAGE}:latest" --region "$REGION" --project "$PROJECT" >/dev/null
fi
gcloud run jobs execute mapi-migrate --region "$REGION" --project "$PROJECT" --wait

say "deploying the service"
gcloud run deploy "$SERVICE" \
  --image "${IMAGE}:latest" --region "$REGION" --project "$PROJECT" \
  --service-account "$SA_EMAIL" \
  --add-cloudsql-instances "$SQL_INSTANCE" \
  --set-env-vars "$COMMON_ENV" \
  --set-secrets "MAPI_DB_PASSWORD=mapi-db-password:latest,MAPI_API_KEY_PEPPER=mapi-api-key-pepper:latest" \
  --command sh \
  --args "-c,export MAPI_DATABASE_URL=\"postgresql+asyncpg://\$MAPI_DB_USER:\$MAPI_DB_PASSWORD@/\$MAPI_DB_NAME?host=/cloudsql/\$MAPI_SQL_INSTANCE\"; exec uvicorn mapi.main:app --factory --host 0.0.0.0 --port \$PORT" \
  --cpu 1 --memory 512Mi --concurrency 40 --timeout 120 \
  --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" \
  --allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format 'value(status.url)')
say "deployed"
echo "  $URL"
curl -s -o /dev/null -w "  health: %{http_code} in %{time_total}s\n" "${URL}/health" || true
