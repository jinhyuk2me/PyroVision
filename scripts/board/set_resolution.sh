#!/usr/bin/env bash
# Update camera resolutions in configs/config.yaml.

if [ -z "$BASH_VERSION" ]; then
    echo "[res-set][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

log() { echo "[res-set] $*"; }
warn() { echo "[res-set][warn] $*" >&2; }
error() { echo "[res-set][error] $*" >&2; exit 1; }

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
def extract(label):
    m = re.search(rf'{label}:\s*(?:\n[^\n]*)*?RES:\s*\[([^\]]+)\]', text, re.MULTILINE)
    return m.group(1).strip() if m else "N/A"
ir = extract("IR")
rgb = extract("RGB_FRONT")
mt = re.search(r'TARGET_RES:\s*\[([^\]]+)\]', text)
target = mt.group(1).strip() if mt else "N/A"
print(f"Current: IR=[{ir}], RGB_FRONT=[{rgb}], TARGET_RES=[{target}]")
PY
}

ask_res() {
    local name="$1"
    shift
    local presets=("$@")
    echo "" >&2
    echo "Select ${name} resolution:" >&2
    local i=1
    for p in "${presets[@]}"; do
        echo "  $i) $p" >&2
        i=$((i + 1))
    done
    echo "  c) custom (format: W,H)" >&2
    read -rp "Choice [1]: " choice
    choice=${choice:-1}
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#presets[@]} )); then
        echo "${presets[$((choice-1))]}"
    else
        read -rp "Enter custom ${name} resolution (W,H): " custom
        echo "$custom"
    fi
}

normalize_res() {
    local res="$1"
    echo "$res" | sed 's/[[:space:]]//g'
}

update_yaml() {
    local ir_res="$1"
    local rgb_res="$2"
    local target_res="$3"
    python3 - "$CONFIG_FILE" "$ir_res" "$rgb_res" "$target_res" <<'PY' || exit 1
import sys, re, pathlib
cfg_path = pathlib.Path(sys.argv[1])
ir = sys.argv[2]
rgb = sys.argv[3]
target = sys.argv[4]
text = cfg_path.read_text()

def replace_res(label, value):
    pattern = rf'({label}:\s*\n(?:[^\n]*\n){{0,10}}?\s*RES:\s*)\[[^\]]+\]'
    new = re.sub(pattern, rf"\1[{value}]", text_res['value'], count=1, flags=re.MULTILINE)
    if new == text_res['value']:
        raise SystemExit(f"{label} RES not found")
    text_res['value'] = new

text_res = {'value': text}
replace_res("IR", ir)
replace_res("RGB_FRONT", rgb)

pattern_target = r'(TARGET_RES:\s*)\[[^\]]+\]'
new_text = re.sub(pattern_target, rf"\1[{target}]", text_res['value'], count=1)
if new_text == text_res['value']:
    raise SystemExit("TARGET_RES not found")

cfg_path.write_text(new_text)
PY
}

main() {
    require_root
    ensure_file

    show_current

    local ir_sel rgb_sel target_sel
    ir_sel=$(ask_res "IR" "160,120" "320,240")
    rgb_sel=$(ask_res "RGB_FRONT" "320,240" "640,480" "1280,720" "1920,1080")
    target_sel=$(ask_res "TARGET_RES" "960,540" "640,360" "same_as_rgb")

    ir_sel=$(normalize_res "$ir_sel")
    rgb_sel=$(normalize_res "$rgb_sel")
    target_sel=$(normalize_res "$target_sel")

    if [[ "$target_sel" == "same_as_rgb" ]]; then
        target_sel="$rgb_sel"
    fi

    log "Backing up config to $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"

    log "Updating RES: IR=[$ir_sel], RGB_FRONT=[$rgb_sel], TARGET_RES=[$target_sel]"
    update_yaml "$ir_sel" "$rgb_sel" "$target_sel"

    log "Done. Current settings:"
    grep -E "^(  IR|  RGB_FRONT|TARGET_RES)" -n "$CONFIG_FILE" | sed 's/^/  /'

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
