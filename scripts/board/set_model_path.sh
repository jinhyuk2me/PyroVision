#!/usr/bin/env bash
# Update configs/config.yaml MODEL (and optionally LABEL) by picking from available .tflite models.

if [ -z "$BASH_VERSION" ]; then
    echo "[model-set][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

log() { echo "[model-set] $*"; }
warn() { echo "[model-set][warn] $*" >&2; }
error() { echo "[model-set][error] $*" >&2; exit 1; }

CONFIG_FILE=${CONFIG_FILE:-/root/pyro_vision/configs/config.yaml}
BACKUP_FILE="${CONFIG_FILE}.bak"
MODEL_ROOT=${MODEL_ROOT:-/root/pyro_vision/model}
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

list_models() {
    find "$MODEL_ROOT" -type f -name "*.tflite" -print 2>/dev/null | sort
}

pick_model() {
    local models=()
    local idx=1
    while IFS= read -r m; do
        models+=("$m")
    done < <(list_models)

    local target=""
    if [[ ${#models[@]} -eq 0 ]]; then
        warn "No .tflite models found under $MODEL_ROOT"
        read -rp "Enter full path to .tflite: " target
    else
        echo "Available models:" >&2
        for m in "${models[@]}"; do
            echo "  $idx) $m" >&2
            idx=$((idx + 1))
        done
        echo "  c) custom path" >&2
        echo "" >&2
        read -rp "Select model number or 'c': " sel
        sel=${sel:-1}
        if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel <= ${#models[@]} )); then
            target="${models[$((sel-1))]}"
        else
            read -rp "Enter full path to .tflite: " target
        fi
    fi

    if [[ ! -f "$target" ]]; then
        error "Model file not found: $target"
    fi
    echo "$target"
}

update_yaml() {
    local model_path="$1"
    local label_path="$2"
    python3 - "$CONFIG_FILE" "$model_path" "$label_path" <<'PY' || exit 1
import pathlib, re, sys
cfg_path = pathlib.Path(sys.argv[1])
model = sys.argv[2]
label = sys.argv[3]
text = cfg_path.read_text()
changed = text
changed = re.sub(r'(MODEL:\s*)[^\n]+', r'\1' + model, changed, count=1)
if label:
    changed = re.sub(r'(LABEL:\s*)[^\n]+', r'\1' + label, changed, count=1)
if changed == text:
    # No change (already set) is treated as success.
    sys.exit(0)
cfg_path.write_text(changed)
PY
}

main() {
    require_root
    ensure_file

    log "Scanning models under $MODEL_ROOT"
    MODEL_PATH=$(pick_model)

    local label_path=""
    local suggested_label
    suggested_label="$(dirname "$MODEL_PATH")/labels.txt"
    if [[ -f "$suggested_label" ]]; then
        read -rp "Update LABEL to $suggested_label? (Y/n): " ans
        ans=${ans:-Y}
        if [[ "$ans" =~ ^[Yy]$ ]]; then
            label_path="$suggested_label"
        fi
    else
        read -rp "Specify LABEL path? (leave empty to keep current): " custom_label
        label_path="$custom_label"
    fi

    log "Backing up config to $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"

    log "Updating MODEL -> $MODEL_PATH"
    if [[ -n "$label_path" ]]; then
        log "Updating LABEL -> $label_path"
    fi
    update_yaml "$MODEL_PATH" "$label_path"

    log "Done. Current MODEL/LABEL:"
    grep -E "^(MODEL|LABEL):" "$CONFIG_FILE" | sed 's/^/  /'

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
