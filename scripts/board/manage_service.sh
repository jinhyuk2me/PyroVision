#!/usr/bin/env bash
# Simple systemd service control helper.

if [ -z "$BASH_VERSION" ]; then
    echo "[svc][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

log() { echo "[svc] $*"; }
warn() { echo "[svc][warn] $*" >&2; }
error() { echo "[svc][error] $*" >&2; exit 1; }

require_root() {
    if [ "$(id -u)" != "0" ]; then
        error "Run as root (sudo)."
    fi
}

SERVICE_NAME=${SERVICE_NAME:-pyro_vision.service}

main() {
    require_root

    echo "Target service: ${SERVICE_NAME}"
    echo "Choose action:"
    echo "  1) start"
    echo "  2) stop"
    echo "  3) restart"
    echo "  4) status"
    read -rp "Action [4]: " action
    action=${action:-4}

    case "$action" in
        1) log "Starting ${SERVICE_NAME}"; systemctl start "${SERVICE_NAME}" ;;
        2) log "Stopping ${SERVICE_NAME}"; systemctl stop "${SERVICE_NAME}" ;;
        3) log "Restarting ${SERVICE_NAME}"; systemctl restart "${SERVICE_NAME}" ;;
        4) log "Status of ${SERVICE_NAME}"; systemctl status "${SERVICE_NAME}" --no-pager || true ;;
        *) error "Invalid action." ;;
    esac

    log "Done."
}

main "$@"
