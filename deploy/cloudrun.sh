#!/usr/bin/env bash
# Mapi on Cloud Run in agentcompile-prod: AgentCompile's memory of proven recipe functions.
#   Postgres: database `mapi` on the agentcompile-db instance (pgvector), reached over the
#             Cloud SQL socket; secrets mapi-database-url, mapi-api-key-pepper
#   Embeddings: Gemini (gemini-embedding-001, 768 dims) on Vertex AI with the service's own
#               identity: no API key anywhere
#   Migrations: run by the service as it starts (bin/start), on its own Cloud SQL connection
#
#   deploy/cloudrun.sh            build and deploy (the service migrates as it starts)
#   Access: every endpoint but /health needs a Mapi API key. Opening it to callers outside the
#   project is one owner command (the deploy account can't change IAM):
#     gcloud run services update mapi --project agentcompile-prod --region us-central1 \
#       --no-invoker-iam-check
#   BOOTSTRAP=1 deploy/cloudrun.sh   also seed the first admin key from mapi-bootstrap-key
#                                    (once; deploy again without it right after)
set -euo pipefail
cd "$(dirname "$0")/.."
P=agentcompile-prod
R=us-central1
SQL="$P:$R:agentcompile-db"
SA="agentcompile-run@$P.iam.gserviceaccount.com"
TAG="$R-docker.pkg.dev/$P/agentcompile/mapi:$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M)"
ENV="MAPI_ENVIRONMENT=staging,MAPI_STORE_BACKEND=postgres,MAPI_EMBEDDING_BACKEND=gemini,MAPI_EMBEDDING_MODEL=gemini-embedding-001,MAPI_EMBEDDING_DIMENSIONS=768,MAPI_GOOGLE_CLOUD_PROJECT=$P,MAPI_GOOGLE_CLOUD_LOCATION=$R"
SECRETS="MAPI_DATABASE_URL=mapi-database-url:latest,MAPI_API_KEY_PEPPER=mapi-api-key-pepper:latest"
if [[ "${BOOTSTRAP:-}" == 1 ]]; then SECRETS="$SECRETS,MAPI_BOOTSTRAP_ADMIN_KEY=mapi-bootstrap-key:latest"; fi

gcloud builds submit --project $P --region $R --tag "$TAG" .

gcloud run deploy mapi --project $P --region $R --image "$TAG" --port 8000 \
  --command /app/bin/start \
  --service-account "$SA" --add-cloudsql-instances "$SQL" \
  --set-env-vars "$ENV" --set-secrets "$SECRETS" \
  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 3
gcloud run services describe mapi --project $P --region $R --format='value(status.url)'
