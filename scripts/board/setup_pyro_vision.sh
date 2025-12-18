#!/usr/bin/env bash
# PyroVision 보드 자동 설정 스크립트
# - udev 심볼릭 링크 생성(pyro_rgb_cam, pyro_ir_cam)
# - configs/config.yaml의 DEVICE를 링크로 갱신
# - systemd 서비스 설치/기동

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/root/pyro_vision}
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/configs/config.yaml}"
UDEV_RULE=/etc/udev/rules.d/99-pyro-vision-camera.rules
SERVICE_FILE=/etc/systemd/system/pyro_vision.service

log() { echo "[setup] $*"; }
warn() { echo "[setup][warn] $*" >&2; }

require_root() {
    if [ "$(id -u)" != "0" ]; then
        warn "root 권한으로 실행하세요."
        exit 1
    fi
}

detect_ir_device() {
    local found=""
    for devpath in /sys/class/video4linux/video*; do
        [ -e "$devpath" ] || continue
        local props
        props=$(udevadm info --query=property --path "$devpath" 2>/dev/null || true)
        local vendor product index node name
        vendor=$(echo "$props" | grep -m1 '^ID_VENDOR_ID=' | cut -d= -f2)
        product=$(echo "$props" | grep -m1 '^ID_MODEL_ID=' | cut -d= -f2)
        name=$(echo "$props" | grep -m1 '^ID_V4L_PRODUCT=' | cut -d= -f2)
        index=$(cat "$devpath/index" 2>/dev/null || echo "")
        # PureThermal (1e4e:0100), index 0를 우선
        if [ "$vendor" = "1e4e" ] && [ "$product" = "0100" ] && [ "$index" = "0" ]; then
            node="/dev/$(basename "$devpath")"
            found="$node"
            break
        fi
        # 이름에 PureThermal 포함 시 후보
        if [ -z "$found" ] && echo "$name" | grep -qi "PureThermal"; then
            node="/dev/$(basename "$devpath")"
            found="$node"
        fi
    done
    echo "$found"
}

detect_rgb_device() {
    local found=""
    for devpath in /sys/class/video4linux/video*; do
        [ -e "$devpath" ] || continue
        local props
        props=$(udevadm info --query=property --path "$devpath" 2>/dev/null || true)
        local product vendor name
        vendor=$(echo "$props" | grep -m1 '^ID_VENDOR_ID=' | cut -d= -f2)
        product=$(echo "$props" | grep -m1 '^ID_V4L_PRODUCT=' | cut -d= -f2)
        name=$(echo "$props" | grep -m1 '^ID_V4L_PRODUCT=' | cut -d= -f2)
        # PureThermal(1e4e:0100)은 IR이므로 스킵
        if [ "$vendor" = "1e4e" ]; then
            continue
        fi
        # mxc_isi.*capture 이름이면 우선 선택
        if echo "$name" | grep -qi "mxc_isi.*capture"; then
            found="/dev/$(basename "$devpath")"
            break
        fi
        # Vivante 등 다른 제품명 우선 선택
        if echo "$product" | grep -qi "viv"; then
            found="/dev/$(basename "$devpath")"
            break
        fi
        # 아직 없으면 첫 비-IR 장치를 후보로
        if [ -z "$found" ]; then
            found="/dev/$(basename "$devpath")"
        fi
    done
    echo "$found"
}

write_udev_rules() {
    cat > "$UDEV_RULE" <<EOF
# PyroVision camera stable symlinks
# IR: PureThermal USB (1e4e:0100)
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1e4e", ATTRS{idProduct}=="0100", SYMLINK+="pyro_ir_cam"
# RGB: Vivante vvcam-video (name=viv_v4l20)
SUBSYSTEM=="video4linux", ATTR{name}=="viv_v4l20", SYMLINK+="pyro_rgb_cam"
EOF
    log "udev 규칙 작성 완료: $UDEV_RULE"
    udevadm control --reload-rules
    udevadm trigger --subsystem-match=video4linux
    # 규칙 적용 완료까지 기다림(장치 인덱스가 늦게 올라올 수 있음)
    udevadm settle --timeout=5 || true
}

wait_symlink() {
    local target="$1" retries=10
    local i=0
    while [ $i -lt $retries ]; do
        [ -e "$target" ] && return 0
        sleep 0.5
        i=$((i + 1))
    done
    return 1
}

ensure_config_devices() {
    local cfg="$1"
    if [ ! -f "$cfg" ]; then
        warn "설정 파일이 없습니다: $cfg"
        return 1
    fi
    python3 - "$cfg" <<'PY'
import re, sys
path = sys.argv[1]
lines = open(path, 'r', encoding='utf-8').read().splitlines()
out = []
section = None
sub = None
for line in lines:
    if re.match(r'^CAMERA:\s*$', line):
        section = 'CAMERA'; sub = None
    elif section == 'CAMERA' and re.match(r'^\s{2}IR:\s*$', line):
        sub = 'IR'
    elif section == 'CAMERA' and re.match(r'^\s{2}RGB_FRONT:\s*$', line):
        sub = 'RGB_FRONT'
    elif section == 'CAMERA' and re.match(r'^\s{2}\S', line):
        sub = None

    if sub == 'IR' and re.match(r'^\s{4}DEVICE:', line):
        out.append('    DEVICE: "/dev/pyro_ir_cam"')
        continue
    if sub == 'RGB_FRONT' and re.match(r'^\s{4}DEVICE:', line):
        out.append('    DEVICE: "/dev/pyro_rgb_cam"')
        continue
    out.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
PY
    log "config DEVICE를 /dev/pyro_ir_cam, /dev/pyro_rgb_cam으로 갱신: $cfg"
}

install_service() {
    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=PyroVision app
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
Environment=CONFIG_PATH=configs/config.yaml
Environment=APP_MODE=cli
Environment=FUSION_VIS_MODE=temp
ExecStart=/usr/bin/python3 app.py
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
    log "systemd 서비스 작성 완료: $SERVICE_FILE"
    systemctl daemon-reload
    systemctl enable --now pyro_vision.service
    log "서비스가 활성화 및 기동되었습니다."
}

main() {
    require_root

    log "RGB/IR 장치 감지 중..."
    ir_dev=$(detect_ir_device || true)
    rgb_dev=$(detect_rgb_device || true)
    if [ -z "$ir_dev" ]; then
        warn "IR(PureThermal) 장치를 찾지 못했습니다. 연결을 확인하세요."
    else
        log "IR 장치: $ir_dev"
    fi
    if [ -z "$rgb_dev" ]; then
        warn "RGB 장치를 찾지 못했습니다. 연결을 확인하세요."
    else
        log "RGB 장치: $rgb_dev"
    fi

    write_udev_rules

    # udev 적용 후 링크 확인
    if wait_symlink /dev/pyro_rgb_cam && wait_symlink /dev/pyro_ir_cam; then
        log "심볼릭 링크 확인 완료: /dev/pyro_rgb_cam -> $(readlink /dev/pyro_rgb_cam), /dev/pyro_ir_cam -> $(readlink /dev/pyro_ir_cam)"
    else
        warn "심볼릭 링크가 생성되지 않았습니다. 장치 연결과 udev 규칙을 확인하세요."
    fi

    ensure_config_devices "$CONFIG_FILE"
    install_service

    log "설정 완료. 상태 확인: systemctl status pyro_vision.service"
}

main "$@"
