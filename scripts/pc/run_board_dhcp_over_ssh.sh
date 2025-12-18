#!/usr/bin/env bash
# Run board DHCP setup over SSH so the board side does not need manual input.

set -euo pipefail

log() { echo "[pc->board] $*"; }
warn() { echo "[pc->board][warn] $*" >&2; }
error() { echo "[pc->board][error] $*" >&2; exit 1; }

BOARD_HOST=${BOARD_HOST:-192.168.50.166}
BOARD_USER=${BOARD_USER:-root}
BOARD_SCRIPT=${BOARD_SCRIPT:-/root/pyro_vision/scripts/board/setup_board_dhcp.sh}

read -rp "Board SSH host [${BOARD_HOST}]: " input_host
BOARD_HOST=${input_host:-$BOARD_HOST}

read -rp "Board SSH user [${BOARD_USER}]: " input_user
BOARD_USER=${input_user:-$BOARD_USER}

read -rp "Board DHCP script path [${BOARD_SCRIPT}]: " input_script
BOARD_SCRIPT=${input_script:-$BOARD_SCRIPT}

log "Running $BOARD_SCRIPT on ${BOARD_USER}@${BOARD_HOST}"

# Quick connectivity check
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${BOARD_USER}@${BOARD_HOST}" "echo ok" >/dev/null 2>&1; then
    warn "Passwordless SSH may not be set. You might be prompted for a password."
fi

ssh "${BOARD_USER}@${BOARD_HOST}" "cd /root/pyro_vision && yes y | bash $BOARD_SCRIPT"

log "Done."
