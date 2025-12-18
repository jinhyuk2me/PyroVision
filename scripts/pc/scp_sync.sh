#!/usr/bin/env bash
set -euo pipefail

# Interactive scp sync loop: choose what to sync from PC -> board.
# Defaults can be overridden via env before running:
#   DEST     : user@host:/path/ (default: root@192.168.50.166:/root/pyro_vision/)
#   INTERVAL : seconds between copies (default: 5)
#   SCP_OPTS : extra scp options (default: "-p")

DEST=${DEST:-root@192.168.50.166:/root/pyro_vision/}
INTERVAL=${INTERVAL:-5}
SCP_OPTS=${SCP_OPTS:-"-p"}

ROOT_DIR="/home/yocto/work/pyro_vision"

choose_source() {
  echo "Select what to sync:"
  echo "  1) Full project (${ROOT_DIR}/)"
  echo "  2) Configs only (${ROOT_DIR}/configs/)"
  echo "  3) Receiver only (${ROOT_DIR}/receiver.py)"
  echo "  4) Sender only (${ROOT_DIR}/sender.py)"
  echo "  5) App only (${ROOT_DIR}/app.py)"
  echo "  6) Enter custom file/dir path"
  read -rp "Choice [1-6]: " choice
  case "$choice" in
    1) SRC="${ROOT_DIR}/" ;;
    2) SRC="${ROOT_DIR}/configs/" ;;
    3) SRC="${ROOT_DIR}/receiver.py" ;;
    4) SRC="${ROOT_DIR}/sender.py" ;;
    5) SRC="${ROOT_DIR}/app.py" ;;
    6)
      read -rp "Enter absolute or repo-relative path: " custom
      if [[ -z "$custom" ]]; then
        echo "No path entered, defaulting to full project."
        SRC="${ROOT_DIR}/"
      elif [[ "$custom" = /* ]]; then
        SRC="$custom"
      else
        SRC="${ROOT_DIR}/${custom}"
      fi
      ;;
    *)
      echo "Invalid choice, defaulting to full project."
      SRC="${ROOT_DIR}/"
      ;;
  esac
}

choose_source
echo "SRC:  ${SRC}"
echo "DEST: ${DEST}"
echo "INTERVAL: ${INTERVAL}s"
echo "SCP_OPTS: ${SCP_OPTS}"
echo "Press Ctrl+C to stop."

trap 'echo "Stopping scp sync loop"; exit 0' INT TERM

while true; do
  ts=$(date +"%F %T")
  if scp -r ${SCP_OPTS} "${SRC}" "${DEST}"; then
    echo "[$ts] OK -> ${DEST}"
  else
    echo "[$ts] FAILED -> ${DEST}" >&2
  fi
  sleep "${INTERVAL}"
done
