#!/usr/bin/env bash
# Update configs/config.yaml SERVER.IP for wired/wifi/custom.

if [ -z "$BASH_VERSION" ]; then
    echo "[server-ip][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

log() { echo "[server-ip] $*"; }
warn() { echo "[server-ip][warn] $*" >&2; }
error() { echo "[server-ip][error] $*" >&2; exit 1; }

CONFIG_FILE=${CONFIG_FILE:-/root/pyro_vision/configs/config.yaml}
BACKUP_FILE="${CONFIG_FILE}.bak"
WIRED_IP_DEFAULT="192.168.200.1"
WIFI_IP_DEFAULT="192.168.50.178"
SERVICE_NAME=${SERVICE_NAME:-pyro_vision.service}

require_root() {
    if [ "$(id -u)" != "0" ]; then
        error "Run as root (sudo)."
    fi
}

ensure_file() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Config file not found: $CONFIG_FILE"
    fi
}

write_ip() {
    local ip="$1"
    python3 - "$ip" "$CONFIG_FILE" <<'PY' || exit 1
import re, sys, pathlib
ip = sys.argv[1]
path = pathlib.Path(sys.argv[2])
text = path.read_text()
pattern = r'(SERVER:\s*\n\s*IP:\s*)[^\n]+'
new = re.sub(pattern, r"\1'"+ip+"'", text, count=1)
if new == text and not re.search(pattern, text):
    sys.exit("pattern not found")
path.write_text(new)
PY
}

main() {
    require_root
    ensure_file

    echo "Select SERVER.IP:"
    echo "  1) wired (${WIRED_IP_DEFAULT})"
    echo "  2) wifi  (${WIFI_IP_DEFAULT})"
    echo "  3) custom"
    read -rp "Choice [1]: " choice
    choice=${choice:-1}

    local target_ip=""
    case "$choice" in
        1) target_ip="$WIRED_IP_DEFAULT" ;;
        2) target_ip="$WIFI_IP_DEFAULT" ;;
        3)
            read -rp "Enter custom IP: " target_ip
            ;;
        *) error "Invalid choice." ;;
    esac

    if [[ -z "$target_ip" ]]; then
        error "IP is required."
    fi

    log "Backing up config to $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"

    log "Updating SERVER.IP -> $target_ip"
    write_ip "$target_ip"

    log "Done. Current SERVER.IP:"
    grep -A2 "^SERVER:" "$CONFIG_FILE" | sed 's/^/  /'

    echo ""
    read -rp "Restart ${SERVICE_NAME} now to apply? (Y/n): " restart_ans
    restart_ans=${restart_ans:-Y}
    if [[ "$restart_ans" =~ ^[Yy]$ ]]; then
        log "Restarting ${SERVICE_NAME}..."
        systemctl restart "${SERVICE_NAME}"
        log "Restarted ${SERVICE_NAME}."
    else
        warn "Skipped restart. Apply changes later with: systemctl restart ${SERVICE_NAME}"
    fi
}

main "$@"
