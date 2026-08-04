#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "install.sh must run as root" >&2
  exit 1
fi

repo_root=/opt/gpt-medical-coder
service_user=ec2-user
install -d -o "$service_user" -g "$service_user" /var/lib/gpt-medical-coder "$repo_root/build/snapshots"
install -d -m 0700 /etc/gpt-medical-coder
if [[ ! -f /etc/gpt-medical-coder/service.env ]]; then
  install -m 0600 "$repo_root/deploy/service.env.example" /etc/gpt-medical-coder/service.env
  echo "Configure /etc/gpt-medical-coder/service.env, then rerun this installer." >&2
  exit 2
fi
if [[ "$(stat -c '%a' /etc/gpt-medical-coder/service.env)" != "600" ]]; then
  echo "service.env must have mode 0600" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
source /etc/gpt-medical-coder/service.env
set +a
if [[ -z "${OPENAI_API_KEY:-}" || "$OPENAI_API_KEY" == replace-with-* ]]; then
  echo "service.env does not contain a configured OpenAI credential" >&2
  exit 2
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" || "$ANTHROPIC_API_KEY" == replace-with-* ]]; then
  echo "service.env does not contain a configured Anthropic credential" >&2
  exit 2
fi
if [[ "${PHI_TRANSMISSION_ALLOWED:-}" != "true" ]]; then
  echo "PHI_TRANSMISSION_ALLOWED must be explicitly set to true" >&2
  exit 2
fi
if [[ -z "${CODER_SERVICE_TOKEN:-}" || "$CODER_SERVICE_TOKEN" == replace-with-* || ${#CODER_SERVICE_TOKEN} -lt 32 ]]; then
  echo "CODER_SERVICE_TOKEN must be a non-placeholder secret of at least 32 characters" >&2
  exit 2
fi

sudo -u "$service_user" python3 -m venv "$repo_root/.venv"
sudo -u "$service_user" "$repo_root/.venv/bin/python" -m pip install --disable-pip-version-check --no-cache-dir "$repo_root"
sudo -u "$service_user" "$repo_root/.venv/bin/python" -m compileall -q "$repo_root/medical_coder" "$repo_root/tools"
sudo -u "$service_user" "$repo_root/.venv/bin/python" -m unittest discover -s "$repo_root/tests"
sudo -u "$service_user" "$repo_root/.venv/bin/python" "$repo_root/tools/check_no_hardcoding.py"

snapshot_json="$(sudo -u "$service_user" "$repo_root/.venv/bin/medical-coder" compile-sources \
  --repository-root "$repo_root" --pack "$repo_root/source-packs/authoritative/pack.json" --output "$repo_root/build/snapshots")"
snapshot_id="$(sudo -u "$service_user" "$repo_root/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["snapshot_id"])' <<<"$snapshot_json")"
sudo -u "$service_user" "$repo_root/.venv/bin/medical-coder" activate-snapshot \
  --snapshot "$repo_root/build/snapshots/$snapshot_id" --link "$repo_root/build/snapshots/current"

install -m 0644 "$repo_root/deploy/gpt-medical-coder.service" /etc/systemd/system/gpt-medical-coder.service
install -m 0644 "$repo_root/deploy/gpt-medical-coder-refresh.service" /etc/systemd/system/gpt-medical-coder-refresh.service
install -m 0644 "$repo_root/deploy/gpt-medical-coder-refresh.timer" /etc/systemd/system/gpt-medical-coder-refresh.timer
chmod 0755 "$repo_root/deploy/refresh-and-activate.sh"
systemctl daemon-reload
systemctl enable --now gpt-medical-coder.service gpt-medical-coder-refresh.timer
systemctl is-active --quiet gpt-medical-coder.service
curl --fail --silent --show-error http://127.0.0.1:8080/readyz >/dev/null
