#!/usr/bin/env bash
# Re-fetches the runtime .env from Secrets Manager. user_data only fetches
# secrets once at first boot (it's frozen via lifecycle.ignore_changes so
# routine code deploys don't replace the instance), so a secret rotation
# needs this to reach the running instance instead of a hand-edited .env.
set -euo pipefail
aws secretsmanager get-secret-value \
  --secret-id "arn:aws:secretsmanager:us-east-1:492411260135:secret:podiatry-coder-app-env-OrPhcB" \
  --region "us-east-1" \
  --query SecretString --output text \
  | jq -r 'to_entries[] | "\(.key)=\(.value)"' > /opt/app/.env
chmod 600 /opt/app/.env
echo "Refreshed /opt/app/.env from Secrets Manager. Existing long-running containers (e.g. a batch started with 'docker compose run -d') keep their old env — recreate them to pick up the new value. New 'docker compose run' invocations pick it up automatically."