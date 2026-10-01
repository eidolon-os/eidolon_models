#!/bin/sh
# Deploy laya to a Linux server over ssh. Idempotent; re-run to update.
#
#   deploy/ecs/deploy.sh [ssh-target]          default: root@eidolon
#
# Knobs (environment, all optional):
#   EIDOLON_LAYA_APP_DIR        where the code goes                  /opt/eidolon-laya
#   EIDOLON_LAYA_HF_ENDPOINT    where fetch downloads from           https://hf-mirror.com
#   EIDOLON_LAYA_DEPLOY_BACKEND backend written on FIRST deploy      torch
#   EIDOLON_LAYA_DEPLOY_THREADS threads written on FIRST deploy      1
#   EIDOLON_LAYA_ONNX_SOURCE    server: export there; local: rsync   server
#                               this machine's models/*/onnx/
#   EIDOLON_LAYA_PYPI_MIRROR    PyPI mirror for installs; empty =    https://mirrors.aliyun.com/pypi/simple/
#                               the lock's own URLs (uv sync)
#   EIDOLON_LAYA_NGINX_SERVER_NAME  opt-in: serve through the host   (empty: off; the service
#                               nginx under this name, 443 + 80->443  listens on 0.0.0.0:8771)
#   EIDOLON_LAYA_NGINX_SSL_DIR  holds fullchain.pem + privkey.pem    /etc/nginx/ssl/yangtzeailab.com
#
# An existing /etc/eidolon-laya/services.toml is never overwritten: change the
# backend there and `systemctl restart eidolon-laya`. A server still on the old
# eidolon-laya.env is migrated once, keeping its backend, address and key.
set -eu

TARGET=${1:-root@eidolon}
APP=${EIDOLON_LAYA_APP_DIR:-/opt/eidolon-laya}
HF_ENDPOINT=${EIDOLON_LAYA_HF_ENDPOINT:-https://hf-mirror.com}
BACKEND=${EIDOLON_LAYA_DEPLOY_BACKEND:-torch}
THREADS=${EIDOLON_LAYA_DEPLOY_THREADS:-1}
ONNX_SOURCE=${EIDOLON_LAYA_ONNX_SOURCE:-server}
PYPI_MIRROR=${EIDOLON_LAYA_PYPI_MIRROR-https://mirrors.aliyun.com/pypi/simple/}
NGINX_NAME=${EIDOLON_LAYA_NGINX_SERVER_NAME:-}
NGINX_SSL=${EIDOLON_LAYA_NGINX_SSL_DIR:-/etc/nginx/ssl/yangtzeailab.com}
LAYA=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
MODEL_REL=models/laya-multilingual/1c5edc17

echo "==> sync $LAYA -> $TARGET:$APP"
ssh "$TARGET" "mkdir -p '$APP'"
rsync -az --delete \
  --exclude .venv --exclude .python --exclude .env \
  --exclude '__pycache__' --exclude .pytest_cache --exclude .ruff_cache \
  --exclude "/$MODEL_REL/torch" --exclude "/$MODEL_REL/onnx" \
  "$LAYA/" "$TARGET:$APP/"

if [ "$ONNX_SOURCE" = local ]; then
  echo "==> rsync local ONNX export (~1.3 GB)"
  rsync -a "$LAYA/$MODEL_REL/onnx/" "$TARGET:$APP/$MODEL_REL/onnx/"
fi

ssh "$TARGET" APP="$APP" MODEL_REL="$MODEL_REL" HF_ENDPOINT="$HF_ENDPOINT" BACKEND="$BACKEND" THREADS="$THREADS" \
  ONNX_SOURCE="$ONNX_SOURCE" PYPI_MIRROR="'$PYPI_MIRROR'" NGINX_NAME="'$NGINX_NAME'" \
  NGINX_SSL="$NGINX_SSL" sh -s <<'REMOTE'
set -eu
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found on the server; install it first" >&2; exit 1; }
cd "$APP"

echo "==> service user"
id eidolon-laya >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin eidolon-laya

echo "==> dependencies (Python under $APP/.python so the service user can read it)"
export UV_PYTHON_INSTALL_DIR="$APP/.python" UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy
if [ -n "$PYPI_MIRROR" ]; then
  # uv.lock pins files.pythonhosted.org, which crawls from mainland China. Keep the
  # lock's exact versions and hashes, fetch the same files from a mirror instead.
  # A version the mirror has not synced yet (laya 0.3.7, networkx 3.7 at the time
  # of writing) falls back to pypi.org. uv ranks --extra-index-url ABOVE
  # --index-url, so the mirror goes first as an extra and pypi.org is the
  # default, i.e. last: listed the other way round, every big wheel came from
  # pypi.org at 0.3 MB/s.
  uv venv --quiet --allow-existing --python 3.13 .venv
  REQ=$(mktemp)
  uv export --locked --all-extras --no-emit-project --quiet > "$REQ"
  uv pip install --quiet --python .venv/bin/python -r "$REQ" \
    --extra-index-url "$PYPI_MIRROR" --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-url https://pypi.org/simple --index-strategy unsafe-best-match
  uv pip install --quiet --python .venv/bin/python --no-deps --index-url "$PYPI_MIRROR" -e .
  rm -f "$REQ"
else
  uv sync --locked --all-extras --quiet
fi

echo "==> weights"
scripts/eidolon-laya --model-dir "$MODEL_REL" fetch --endpoint "$HF_ENDPOINT"
if [ "$ONNX_SOURCE" = server ]; then
  # Export needs the model twice over in RAM; cap it so a small host is not squeezed.
  systemd-run --scope --quiet -p MemoryMax=5G scripts/eidolon-laya --model-dir "$MODEL_REL" export-onnx \
    || echo "!! ONNX export failed; torch backend still works. Retry with EIDOLON_LAYA_ONNX_SOURCE=local"
fi

echo "==> config"
mkdir -p /etc/eidolon-laya
CONF=/etc/eidolon-laya/services.toml
KEYFILE=/etc/eidolon-laya/api-key
OLD_ENV=/etc/eidolon-laya/eidolon-laya.env
if [ ! -f "$CONF" ]; then
  # A server deployed before the services file carries its choices in the old env file: keep them.
  if [ -f "$OLD_ENV" ]; then
    old() { sed -n "s/^$1=//p" "$OLD_ENV" | tail -1; }
    BACKEND=$(old EIDOLON_LAYA_BACKEND); BACKEND=${BACKEND:-torch}
    THREADS=$(old EIDOLON_LAYA_THREADS); THREADS=${THREADS:-1}
    LISTEN=$(old EIDOLON_LAYA_HOST); LISTEN=${LISTEN:-127.0.0.1}
    PORT=$(old EIDOLON_LAYA_PORT); PORT=${PORT:-8771}
    KEY=$(old EIDOLON_LAYA_API_KEY)
  else
    # Behind nginx the service only needs loopback; without it, it faces the network.
    if [ -n "$NGINX_NAME" ]; then LISTEN=127.0.0.1; else LISTEN=0.0.0.0; fi
    PORT=8771
    KEY=
  fi
  [ -n "$KEY" ] || KEY=$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(24))')
  umask 027
  printf '%s\n' "$KEY" > "$KEYFILE"
  chown root:eidolon-laya "$KEYFILE"
  chmod 640 "$KEYFILE"
  {
    echo "# This server's Laya service (written once by deploy/ecs/deploy.sh; edit and restart eidolon-laya)."
    echo "schema_version = 1"
    echo
    echo "[systemone]"
    echo "host = \"$LISTEN\""
    echo "port = $PORT"
    echo "threads = $THREADS"
    echo "max_pending = 4"
    echo "api_key_file = \"$KEYFILE\""
    echo
    echo "[systemone.$BACKEND]"
    echo "model_dir = \"$MODEL_REL\""
  } > "$CONF"
  chmod 644 "$CONF"
  if [ -f "$OLD_ENV" ]; then mv "$OLD_ENV" "$OLD_ENV.migrated"; fi
  echo "   wrote $CONF and $KEYFILE"
else
  echo "   kept existing $CONF"
fi
echo "==> systemd"
install -m 644 deploy/systemd/eidolon-laya.service /etc/systemd/system/eidolon-laya.service
systemctl daemon-reload
systemctl enable --quiet eidolon-laya
systemctl restart eidolon-laya
PORT=$(sed -n 's/^port *= *//p' "$CONF" | head -1); PORT=${PORT:-8771}
for _ in $(seq 1 120); do
  curl -fsS -m 2 "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -fsS -m 2 "http://127.0.0.1:$PORT/readyz"; then
  echo "!! not ready; journalctl -u eidolon-laya -n 50" >&2
  journalctl -u eidolon-laya -n 30 --no-pager >&2
  exit 1
fi
echo

if [ -n "$NGINX_NAME" ]; then
  echo "==> nginx site $NGINX_NAME"
  [ -f "$NGINX_SSL/fullchain.pem" ] && [ -f "$NGINX_SSL/privkey.pem" ] \
    || { echo "!! no fullchain.pem/privkey.pem in $NGINX_SSL" >&2; exit 1; }
  SITE=/etc/nginx/conf.d/$NGINX_NAME.conf
  NEW=$(mktemp)
  sed -e "s|__SERVER_NAME__|$NGINX_NAME|g" -e "s|__SSL_DIR__|$NGINX_SSL|g" \
      -e "s|__PORT__|$PORT|g" deploy/nginx/eidolon-laya.conf > "$NEW"
  if [ -f "$SITE" ] && cmp -s "$NEW" "$SITE"; then
    echo "   unchanged"
    rm -f "$NEW"
  else
    [ -f "$SITE" ] && cp -p "$SITE" "$SITE.bak"
    install -m 644 "$NEW" "$SITE"
    rm -f "$NEW"
    # nginx -t checks the whole config: a failure rolls back ours and leaves the
    # running nginx (and every other site) exactly as it was.
    if nginx -t 2>/tmp/eidolon-laya-nginx-test.log; then
      systemctl reload nginx
      rm -f "$SITE.bak"
      echo "   installed and reloaded"
    else
      cat /tmp/eidolon-laya-nginx-test.log >&2
      if [ -f "$SITE.bak" ]; then mv "$SITE.bak" "$SITE"; else rm -f "$SITE"; fi
      echo "!! nginx -t failed; rolled back, nginx not reloaded" >&2
      exit 1
    fi
  fi
  echo "==> up: https://$NGINX_NAME  (key: $KEYFILE)"
else
  echo "==> up: http://<public-ip>:$PORT  (key: $KEYFILE)"
fi
REMOTE
