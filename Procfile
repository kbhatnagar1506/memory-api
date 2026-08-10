web: bin/with-cloudsql uvicorn mapi.main:app --factory --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'
release: bin/with-cloudsql alembic upgrade head
