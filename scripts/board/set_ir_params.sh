#!/usr/bin/env bash
# Update IR TAU and FIRE_MIN_TEMP in configs/config.yaml.

if [ -z "$BASH_VERSION" ]; then
    echo "[ir-set][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

log() { echo "[ir-set] $*"; }
warn() { echo "[ir-set][warn] $*" >&2; }
error() { echo "[ir-set][error] $*" >&2; exit 1; }

CONFIG_FILE=${CONFIG_FILE:-/root/pyro_vision/configs/config.yaml}
BACKUP_FILE="${CONFIG_FILE}.bak"
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

show_current() {
    python3 - "$CONFIG_FILE" <<'PY' || exit 1
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
text = path.read_text()
def find_num(key):
    m = re.search(rf'{key}:\s*([0-9.]+)', text)
    return m.group(1) if m else "N/A"
print(f"Current: TAU={find_num('TAU')}, FIRE_MIN_TEMP={find_num('FIRE_MIN_TEMP')}")
PY
}

update_yaml() {
    local tau="$1"
    local fire_min="$2"
    python3 - "$CONFIG_FILE" "$tau" "$fire_min" <<'PY' || exit 1
import pathlib, re, sys
cfg_path = pathlib.Path(sys.argv[1])
tau = sys.argv[2]
fire = sys.argv[3]
text = cfg_path.read_text()
changed = re.sub(r'(TAU:\s*)([0-9.]+)', rf'\g<1>{tau}', text, count=1)
changed = re.sub(r'(FIRE_MIN_TEMP:\s*)([0-9.]+)', rf'\g<1>{fire}', changed, count=1)
cfg_path.write_text(changed)
PY
}

main() {
    require_root
    ensure_file
    show_current

    read -rp "TAU (e.g., 0.5): " tau
    read -rp "FIRE_MIN_TEMP (Celsius, e.g., 80): " fire_min
    if [[ -z "$tau" || -z "$fire_min" ]]; then
        error "Both TAU and FIRE_MIN_TEMP are required."
    fi

    log "Backing up config to $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"

    log "Updating TAU -> $tau, FIRE_MIN_TEMP -> $fire_min"
    update_yaml "$tau" "$fire_min"

    log "Done. Current values:"
    show_current

    echo ""
    read -rp "Restart ${SERVICE_NAME} now to apply? (Y/n): " restart_ans
    restart_ans=${restart_ans:-Y}
    if [[ "$restart_ans" =~ ^[Yy]$ ]]; then
        log "Restarting ${SERVICE_NAME}..."
        systemctl restart "${SERVICE_NAME}"
        log "Restarted ${SERVICE_NAME}."
    else
        warn "Skipped restart. Apply later with: systemctl restart ${SERVICE_NAME}"
    fi
}

main "$@"
