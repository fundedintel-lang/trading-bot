#!/usr/bin/env bash
# Provision APEX-SMC on a fresh Ubuntu 22.04/24.04 VM.  Run as root.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/apex-smc}
ENV_DIR=${ENV_DIR:-/etc/apex-smc}
SERVICE_USER=${SERVICE_USER:-apex}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "==> Creating service user and directories"
id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --shell /usr/sbin/nologin --home "$APP_DIR" "$SERVICE_USER"
mkdir -p "$APP_DIR" "$ENV_DIR" "$APP_DIR/logs" "$APP_DIR/reports" "$APP_DIR/data"

echo "==> Copying application files"
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'logs/*' --exclude 'reports/*' --exclude '*.sqlite3*' \
  "$REPO_DIR"/ "$APP_DIR"/

echo "==> Creating the virtual environment"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip wheel >/dev/null
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "==> Preparing the secrets file"
if [ ! -f "$ENV_DIR/apex.env" ]; then
  cp "$APP_DIR/.env.example" "$ENV_DIR/apex.env"
  chmod 600 "$ENV_DIR/apex.env"
  echo "    Created $ENV_DIR/apex.env — fill in your credentials before starting."
fi

chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"
chown root:"$SERVICE_USER" "$ENV_DIR/apex.env"
chmod 640 "$ENV_DIR/apex.env"

echo "==> Installing systemd units"
install -m 644 "$APP_DIR/deploy/apex-smc.service" /etc/systemd/system/apex-smc.service
install -m 644 "$APP_DIR/deploy/apex-smc-dashboard.service" /etc/systemd/system/apex-smc-dashboard.service
install -m 644 "$APP_DIR/deploy/logrotate.conf" /etc/logrotate.d/apex-smc
systemctl daemon-reload

echo "==> Validating configuration"
sudo -u "$SERVICE_USER" "$APP_DIR/.venv/bin/python" "$APP_DIR/main.py" validate-config || true

cat <<TXT

Done. Next steps:
  1. Edit $ENV_DIR/apex.env (OANDA token, account id, Telegram token/chat id).
  2. systemctl enable --now apex-smc            # trading engine
  3. systemctl enable --now apex-smc-dashboard  # dashboard on 127.0.0.1:8501
  4. journalctl -u apex-smc -f                  # follow the structured logs
TXT
