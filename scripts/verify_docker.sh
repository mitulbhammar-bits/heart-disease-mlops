#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/evidence
command -v docker >/dev/null || { echo "Install and start Docker Desktop first."; exit 1; }
docker build -t heart-risk-api:1.0.0 . 2>&1 | tee reports/evidence/docker_build.log
container_id=$(docker run -d -p 127.0.0.1:8001:8000 heart-risk-api:1.0.0)
trap 'docker logs "$container_id" > reports/evidence/docker_runtime.log 2>&1; docker rm -f "$container_id" >/dev/null' EXIT
for attempt in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8001/ready >/dev/null; then break; fi
  sleep 2
done
python scripts/verify_endpoint.py --url http://127.0.0.1:8001 --output reports/evidence/docker_api_verification.json
docker inspect "$container_id" > reports/evidence/docker_container_inspect.json
