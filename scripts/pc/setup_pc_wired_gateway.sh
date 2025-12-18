#!/usr/bin/env bash
# PC 인터넷 공유 설정 스크립트
# - PC를 공유기처럼 만들어 보드에 인터넷 연결
# - 인터페이스 이름을 자동으로 찾아서 사용자가 선택

set -euo pipefail

# 보드 정보 (기본값은 프롬프트 기본값으로만 사용)
BOARD_MAC="${BOARD_MAC:-b2:4f:1a:07:1e:d0}"    # 1번 보드 기본 MAC (프롬프트 기본값)
BOARD_IP="${BOARD_IP:-192.168.200.11}"         # 1번 보드 기본 IP (프롬프트 기본값)
BOARD_HOSTS=()                                 # 프롬프트로 수집한 MAC/IP 리스트
PC_IP="192.168.200.1"
DHCP_RANGE_START="192.168.200.50"
DHCP_RANGE_END="192.168.200.150"

log() { echo "[setup] $*"; }
warn() { echo "[setup][warn] $*" >&2; }
error() { echo "[setup][error] $*" >&2; exit 1; }

require_root() {
    if [ "$(id -u)" != "0" ]; then
        error "root 권한으로 실행하세요. (sudo 사용)"
    fi
}

# 유선 인터페이스 목록 찾기
detect_wired_interfaces() {
    local interfaces=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local ifname
        ifname=$(echo "$line" | awk '{print $2}' | tr -d ':')
        if [[ -z "$ifname" ]] || [[ "$ifname" == "lo" ]]; then
            continue
        fi
        # Wi-Fi 인터페이스 제외
        if [[ "$ifname" =~ ^(wlan|wlp|mlan|wl) ]]; then
            continue
        fi
        # Docker, 브리지 등 제외
        if [[ "$ifname" =~ ^(docker|br-|veth) ]]; then
            continue
        fi
        # IP 주소 확인
        local ip="(no IP)"
        if ip addr show "$ifname" 2>/dev/null | grep -q "inet "; then
            ip=$(ip addr show "$ifname" 2>/dev/null | grep "inet " | head -1 | awk '{print $2}' | cut -d/ -f1)
        fi
        interfaces+=("$ifname|$ip")
    done < <(ip link show 2>/dev/null | grep -E "^[0-9]+:")
    if [[ ${#interfaces[@]} -gt 0 ]]; then
        printf '%s\n' "${interfaces[@]}"
    fi
}

# Wi-Fi 인터페이스 목록 찾기
detect_wifi_interfaces() {
    local interfaces=()
    while IFS= read -r line; do
        local ifname
        ifname=$(echo "$line" | awk '{print $2}' | tr -d ':')
        if [[ -n "$ifname" ]] && [[ "$ifname" != "lo" ]]; then
            # Wi-Fi 인터페이스인지 확인 (wlan, wlp, mlan 등)
            if [[ "$ifname" =~ ^(wlan|wlp|mlan|wl) ]]; then
                local ip
                ip=$(ip addr show "$ifname" 2>/dev/null | grep "inet " | head -1 | awk '{print $2}' | cut -d/ -f1 || echo "(no IP)")
                interfaces+=("$ifname|$ip")
            fi
        fi
    done < <(ip link show | grep -E "^[0-9]+:")
    printf '%s\n' "${interfaces[@]}"
}

# 인터페이스 선택 (전역 변수에 저장)
choose_interface() {
    local type="$1"  # "wired" or "wifi"
    local title="$2"
    local var_name="$3"  # 결과를 저장할 변수명
    local interfaces=()
    local count=0
    local choices=()
    local selected=""
    local if_info ifname ip choice
    
    # 제목 출력
    echo ""
    echo "$title"
    echo ""
    
    # 인터페이스 감지 및 목록 표시 (먼저 목록을 표시)
    if [[ "$type" == "wired" ]]; then
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            count=$((count + 1))
            IFS='|' read -r ifname ip <<< "$line"
            echo "  $count) $ifname ($ip)"
            choices+=("$ifname")
            interfaces+=("$line")
        done < <(detect_wired_interfaces)
    else
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            count=$((count + 1))
            IFS='|' read -r ifname ip <<< "$line"
            echo "  $count) $ifname ($ip)"
            choices+=("$ifname")
            interfaces+=("$line")
        done < <(detect_wifi_interfaces)
    fi
    
    if [[ $count -eq 0 ]]; then
        error "$type 인터페이스를 찾을 수 없습니다."
    fi
    
    # 선택 입력 받기 (목록 표시 후)
    echo ""
    read -rp "선택하세요 [1-${count}]: " choice
    
    # 입력 검증
    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt "$count" ]]; then
        error "잘못된 선택입니다."
    fi
    
    selected="${choices[$((choice - 1))]}"
    # 전역 변수에 저장
    eval "$var_name=\"$selected\""
}

auto_detect_mac() {
    local ifname="$1"
    local timeout_sec="${2:-20}"
    local capture_count="${3:-50}"

    ip link set "$ifname" up >/dev/null 2>&1 || true

    # 링크가 내려가 있으면 미리 알려준다.
    if [[ -f "/sys/class/net/${ifname}/carrier" ]] && [[ "$(cat "/sys/class/net/${ifname}/carrier" 2>/dev/null)" != "1" ]]; then
        warn "${ifname} 링크가 DOWN 상태입니다. 케이블/전원을 확인하세요."
    fi

    # 먼저 ARP/네이버 캐시에 있는 MAC 시도
    local cached_mac
    cached_mac=$(ip neigh show dev "$ifname" 2>/dev/null | awk '!/FAILED/ && $5 ~ /([0-9a-f]{2}:){5}[0-9a-f]{2}/ {print $5; exit}')
    if [[ -n "$cached_mac" ]]; then
        echo "$cached_mac"
        return 0
    fi

    if ! command -v tcpdump >/dev/null 2>&1; then
        warn "tcpdump가 없어 MAC 자동 감지를 건너뜁니다. 필요 시 설치해주세요."
        return 1
    fi
    if ! command -v timeout >/dev/null 2>&1; then
        warn "timeout 명령이 없어 MAC 자동 감지를 건너뜁니다. coreutils 설치를 확인하세요."
        return 1
    fi

    warn "MAC 자동 감지를 위해 ${timeout_sec}초 동안 ${ifname} 트래픽을 듣습니다. (보드를 켜거나 케이블을 재연결하세요)"
    local capture_output
    capture_output=$(timeout "$timeout_sec" tcpdump -len -c "$capture_count" -i "$ifname" '(arp or (udp and (port 67 or 68)) or ether broadcast)' 2>/dev/null || true)
    if [[ -z "$capture_output" ]]; then
        return 1
    fi

    local self_mac
    self_mac=$(cat "/sys/class/net/${ifname}/address" 2>/dev/null | tr 'A-F' 'a-f')

    local mac
    mac=$(echo "$capture_output" \
        | grep -Eio '([0-9a-f]{2}:){5}[0-9a-f]{2}' \
        | tr 'A-F' 'a-f' \
        | grep -vi '^ff:ff:ff:ff:ff:ff$' \
        | grep -vi "^${self_mac}$" \
        | head -1)
    if [[ -n "$mac" ]]; then
        echo "$mac"
        return 0
    fi

    warn "감지된 트래픽은 있었지만 MAC을 읽지 못했습니다. (아래는 첫 몇 줄)"
    echo "$capture_output" | head -n 5 >&2

    return 1
}

# IP 포워딩 설정
setup_ip_forwarding() {
    log "IP 포워딩 활성화..."
    sysctl -w net.ipv4.ip_forward=1 > /dev/null
    log "IP 포워딩 활성화 완료"
}

# NAT 설정
setup_nat() {
    local wired_if="$1"
    local wifi_if="$2"
    
    log "NAT 설정 중..."
    
    # 기존 규칙 확인 및 추가 (중복 방지)
    if ! iptables -t nat -C POSTROUTING -o "$wifi_if" -j MASQUERADE 2>/dev/null; then
        iptables -t nat -A POSTROUTING -o "$wifi_if" -j MASQUERADE
        log "  - MASQUERADE 규칙 추가"
    fi
    
    if ! iptables -C FORWARD -i "$wired_if" -o "$wifi_if" -j ACCEPT 2>/dev/null; then
        iptables -A FORWARD -i "$wired_if" -o "$wifi_if" -j ACCEPT
        log "  - FORWARD 규칙 추가 (wired -> wifi)"
    fi
    
    if ! iptables -C FORWARD -i "$wifi_if" -o "$wired_if" -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null; then
        iptables -A FORWARD -i "$wifi_if" -o "$wired_if" -m state --state ESTABLISHED,RELATED -j ACCEPT
        log "  - FORWARD 규칙 추가 (wifi -> wired, established)"
    fi
    
    log "NAT 설정 완료"
}

# DHCP 서버 시작
start_dhcp_server() {
    local wired_if="$1"
    
    log "DHCP 서버 시작 중..."
    
    # 기존 dnsmasq 종료
    pkill dnsmasq 2>/dev/null || true
    sleep 1
    
    # dnsmasq 설치 확인
    if ! command -v dnsmasq &> /dev/null; then
        warn "dnsmasq가 설치되어 있지 않습니다. 설치하시겠습니까? (y/N)"
        read -rp "> " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            if command -v apt-get &> /dev/null; then
                apt-get update && apt-get install -y dnsmasq
            elif command -v yum &> /dev/null; then
                yum install -y dnsmasq
            else
                error "패키지 매니저를 찾을 수 없습니다. dnsmasq를 수동으로 설치하세요."
            fi
        else
            error "dnsmasq가 필요합니다."
        fi
    fi
    
    if [[ ${#BOARD_HOSTS[@]} -eq 0 ]]; then
        error "DHCP 예약 대상이 없습니다. MAC/IP 입력을 확인하세요."
    fi
    # dhcp-host 옵션 구성
    local dhcp_hosts=("${BOARD_HOSTS[@]}")

    # dnsmasq 백그라운드 실행
    local cmd=(
        dnsmasq
        --interface="$wired_if"
        --bind-interfaces
        --port=0
        --dhcp-range="${DHCP_RANGE_START},${DHCP_RANGE_END},12h"
        --dhcp-option=3,"${PC_IP}"
        --dhcp-option=6,8.8.8.8
        --log-dhcp
        --no-daemon
    )
    local host
    for host in "${dhcp_hosts[@]}"; do
        cmd+=(--dhcp-host="$host")
    done

    "${cmd[@]}" &
    
    local dnsmasq_pid=$!
    sleep 1
    
    # dnsmasq 프로세스 확인
    if ! kill -0 "$dnsmasq_pid" 2>/dev/null; then
        error "dnsmasq 시작 실패. 로그를 확인하세요."
    fi
    
    log "DHCP 서버 시작 완료 (PID: $dnsmasq_pid)"
    log "보드 IP 목록:"
    for host in "${dhcp_hosts[@]}"; do
        log "  - ${host}"
    done
    log "게이트웨이: $PC_IP"
    log "DNS: 8.8.8.8"
}

# 유선 인터페이스 IP 설정
setup_wired_interface() {
    local wired_if="$1"
    
    log "유선 인터페이스 IP 설정 중..."
    
    # 기존 IP 제거
    ip addr flush dev "$wired_if" 2>/dev/null || true
    
    # 새 IP 추가
    ip addr add "${PC_IP}/24" dev "$wired_if"
    
    log "유선 인터페이스 IP 설정 완료: ${PC_IP}/24"
}

# 설정 확인
verify_setup() {
    local wired_if="$1"
    
    log "설정 확인 중..."
    
    # IP 포워딩 확인
    if [[ "$(sysctl -n net.ipv4.ip_forward)" != "1" ]]; then
        warn "IP 포워딩이 활성화되지 않았습니다."
    else
        log "✓ IP 포워딩 활성화됨"
    fi
    
    # 유선 인터페이스 IP 확인
    if ip addr show "$wired_if" | grep -q "${PC_IP}"; then
        log "✓ 유선 인터페이스 IP 설정됨: ${PC_IP}"
    else
        warn "유선 인터페이스 IP가 설정되지 않았습니다."
    fi
    
    # dnsmasq 프로세스 확인
    if pgrep -x dnsmasq > /dev/null; then
        log "✓ DHCP 서버 실행 중"
    else
        warn "DHCP 서버가 실행되지 않았습니다."
    fi
    
    # 보드 연결 확인 (ARP 테이블)
    echo ""
    log "보드 연결 확인 (ARP 테이블):"
    ip neigh show dev "$wired_if" 2>/dev/null | grep -v "FAILED" || warn "보드가 아직 연결되지 않았습니다."
}

main() {
    require_root
    
    echo "=========================================="
    echo "  PC 인터넷 공유 설정"
    echo "=========================================="
    echo ""

    # 인터페이스 선택을 먼저 진행 (MAC 자동 감지에 필요)
    choose_interface "wired" "보드와 연결된 유선 인터페이스를 선택하세요:" "WIRED_IF"
    choose_interface "wifi" "인터넷에 연결된 Wi-Fi 인터페이스를 선택하세요:" "WIFI_IF"
    echo ""
    
    # 보드 MAC/IP를 프롬프트로 입력받아 예약 목록 구성
    echo "고정 IP로 예약할 보드 수를 입력하세요. (엔터 시 1)"
    read -rp "보드 수: " board_count
    if [[ -z "$board_count" ]]; then
        board_count=1
    elif ! [[ "$board_count" =~ ^[0-9]+$ ]] || (( board_count < 1 )); then
        error "보드 수는 1 이상의 숫자여야 합니다."
    fi

    for ((i=1; i<=board_count; i++)); do
        local_mac=""
        local_ip=""
        # 1번 보드는 기본값을 프롬프트에 노출
        if (( i == 1 )); then
            local_mac="$BOARD_MAC"
            local_ip="$BOARD_IP"
        else
            # 2번부터는 192.168.200.12, 13 ... 제안
            local_ip="192.168.200.$((10 + i))"
        fi

        while true; do
            prompt_mac="보드 ${i} MAC 주소"
            if [[ -n "$local_mac" ]]; then
                prompt_mac+=" [기본: ${local_mac}]"
            fi
            prompt_mac+=" (auto 입력 시 ${WIRED_IF}에서 자동 감지 시도): "
            read -rp "$prompt_mac" input_mac
            if [[ "$input_mac" == "auto" ]]; then
                local detected_mac
                detected_mac=$(auto_detect_mac "$WIRED_IF" 20 || true)
                if [[ -n "$detected_mac" ]]; then
                    log "자동 감지 성공: ${detected_mac}"
                    input_mac="$detected_mac"
                else
                    warn "MAC 감지 실패. 보드 전원을 껐다 켜거나 연결을 확인한 뒤 다시 시도하세요."
                    continue
                fi
            fi
            if [[ -z "$input_mac" && -n "$local_mac" ]]; then
                input_mac="$local_mac"
            fi
            if [[ -z "$input_mac" ]]; then
                warn "MAC 주소는 필수입니다."
                continue
            fi
            # 형식 검증 (aa:bb:cc:dd:ee:ff)
            if ! [[ "$input_mac" =~ ^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$ ]]; then
                warn "MAC 주소 형식이 올바르지 않습니다. 예) aa:bb:cc:dd:ee:ff"
                continue
            fi
            local_mac=$(echo "$input_mac" | tr 'A-F' 'a-f')
            break
        done

        while true; do
            prompt_ip="보드 ${i} IP 주소"
            if [[ -n "$local_ip" ]]; then
                prompt_ip+=" [기본: ${local_ip}]"
            fi
            prompt_ip+=": "
            read -rp "$prompt_ip" input_ip
            if [[ -z "$input_ip" && -n "$local_ip" ]]; then
                input_ip="$local_ip"
            fi
            if [[ -z "$input_ip" ]]; then
                warn "IP 주소는 필수입니다."
                continue
            fi
            local_ip="$input_ip"
            break
        done

        BOARD_HOSTS+=("${local_mac},${local_ip}")
    done
    echo ""
    
    echo ""
    log "설정 요약:"
    log "  유선 인터페이스: $WIRED_IF"
    log "  Wi-Fi 인터페이스: $WIFI_IF"
    log "  DHCP 고정 예약:"
    for host in "${BOARD_HOSTS[@]}"; do
        log "    - $host"
    done
    echo ""
    
    # 확인
    read -rp "계속하시겠습니까? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log "취소되었습니다."
        exit 0
    fi
    echo ""
    
    # 설정 실행
    setup_wired_interface "$WIRED_IF"
    setup_ip_forwarding
    setup_nat "$WIRED_IF" "$WIFI_IF"
    start_dhcp_server "$WIRED_IF"
    
    echo ""
    verify_setup "$WIRED_IF"
    
    echo ""
    log "설정 완료!"
    echo ""
    log "다음 단계:"
    log "  1. 보드에서 /etc/systemd/network/10-eth0-static.network 파일을 DHCP 모드로 변경"
    log "  2. 보드에서 'sudo systemctl restart systemd-networkd' 실행"
    log "  3. 보드에서 'ping 8.8.8.8' 테스트"
    echo ""
    log "DHCP 서버 종료: sudo pkill dnsmasq"
}

main "$@"
