#!/usr/bin/env bash
# Board helper menu to run common scripts.

if [ -z "$BASH_VERSION" ]; then
    echo "[menu][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[menu] $*"; }
warn() { echo "[menu][warn] $*" >&2; }
error() { echo "[menu][error] $*" >&2; exit 1; }

require_root() {
    if [ "$(id -u)" != "0" ]; then
        error "Run as root (sudo)."
    fi
}

run_script() {
    local script="$1"
    shift || true
    if [[ ! -x "$script" ]]; then
        warn "$script is not executable. Trying with bash."
        bash "$script" "$@"
    else
        "$script" "$@"
    fi
}

main() {
    require_root

    cat <<EOF
==========================
  PyroVision Board Menu
==========================
1) Setup eth0 DHCP                (setup_board_dhcp.sh)
2) Set SERVER.IP + restart svc    (set_server_ip.sh)
3) Select MODEL/LABEL             (set_model_path.sh)
4) Manage pyro_vision.service         (manage_service.sh)
5) Switch wired/wifi interfaces   (switch_network_interface.sh)
6) Initial board setup            (setup_pyro_vision.sh)
7) Set resolutions (IR/RGB/TARGET) (set_resolution.sh)
8) Set IR TAU / FIRE_MIN_TEMP     (set_ir_params.sh)
q) Quit
EOF

    read -rp "Select: " sel
    case "$sel" in
        1) run_script "$SCRIPT_DIR/setup_board_dhcp.sh" ;;
        2) run_script "$SCRIPT_DIR/set_server_ip.sh" ;;
        3) run_script "$SCRIPT_DIR/set_model_path.sh" ;;
        4) run_script "$SCRIPT_DIR/manage_service.sh" ;;
        5) run_script "$SCRIPT_DIR/switch_network_interface.sh" ;;
        6) run_script "$SCRIPT_DIR/setup_pyro_vision.sh" ;;
        7) run_script "$SCRIPT_DIR/set_resolution.sh" ;;
        8) run_script "$SCRIPT_DIR/set_ir_params.sh" ;;
        q|Q) log "Bye."; exit 0 ;;
        *) error "Invalid selection." ;;
    esac
}

main "$@"
