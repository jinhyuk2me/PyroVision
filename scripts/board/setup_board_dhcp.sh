#!/usr/bin/env bash
# Board DHCP setup script
# - Show eth0 MAC address
# - Switch eth0 to DHCP mode
# - Restart network

if [ -z "$BASH_VERSION" ]; then
    echo "[setup][error] This script requires bash. Run with 'bash $0'." >&2
    exit 1
fi

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/root/pyro_vision}
NETWORK_FILE="/etc/systemd/network/10-eth0-static.network"
BACKUP_FILE="/etc/systemd/network/10-eth0-static.network.backup"

log() { echo "[setup] $*"; }
warn() { echo "[setup][warn] $*" >&2; }
error() { echo "[setup][error] $*" >&2; exit 1; }

require_root() {
    if [ "$(id -u)" != "0" ]; then
        error "Run as root (use sudo)."
    fi
}

enable_networkd_services() {
    # Ensure services are enabled so DHCP config persists across reboots
    systemctl enable systemd-networkd >/dev/null 2>&1 || warn "Could not enable systemd-networkd"
    systemctl enable systemd-resolved >/dev/null 2>&1 || warn "Could not enable systemd-resolved"
}

# Get eth0 MAC
get_eth0_mac() {
    local mac
    mac=$(cat /sys/class/net/eth0/address 2>/dev/null || echo "")
    if [[ -z "$mac" ]]; then
        error "eth0 interface not found."
    fi
    echo "$mac"
}

# Check current network config
check_current_config() {
    if [[ ! -f "$NETWORK_FILE" ]]; then
        warn "Network config file not found: $NETWORK_FILE"
        return 1
    fi
    
    if grep -q "DHCP=yes" "$NETWORK_FILE" 2>/dev/null; then
        log "Already set to DHCP mode."
        return 0
    fi
    
    return 1
}

# Switch to DHCP
setup_dhcp() {
    log "Backing up network config..."
    if [[ -f "$NETWORK_FILE" ]]; then
        cp "$NETWORK_FILE" "$BACKUP_FILE"
        log "Backup done: $BACKUP_FILE"
    fi
    
    log "Switching to DHCP mode..."
    cat > "$NETWORK_FILE" <<EOF_CONF
[Match]
Name=eth0

[Link]
ActivationPolicy=always-up
RequiredForOnline=no
ConfigureWithoutCarrier=yes

[Network]
DHCP=yes
LinkLocalAddressing=no
DefaultRouteOnDevice=no
EOF_CONF
    
    log "Network config updated"
}

# Restart network
restart_network() {
    log "Restarting systemd-networkd..."
    systemctl restart systemd-networkd
    
    # Wait for network to stabilize
    sleep 3
    
    log "Restart complete"
}

# Verify network status
verify_network() {
    log "Checking network status..."
    
    # eth0 IP
    local ip
    ip=$(ip addr show eth0 2>/dev/null | grep "inet " | head -1 | awk '{print $2}' | cut -d/ -f1 || echo "")
    
    if [[ -n "$ip" ]]; then
        log "OK eth0 IP: $ip"
    else
        warn "No IP on eth0."
    fi
    
    # Default gateway
    local gateway
    gateway=$(ip route show default 2>/dev/null | grep "dev eth0" | awk '{print $3}' || echo "")
    
    if [[ -n "$gateway" ]]; then
        log "OK Default gateway: $gateway"
    else
        warn "No default gateway on eth0."
    fi
    
    # DNS
    if grep -q "nameserver" /etc/resolv.conf 2>/dev/null; then
        log "OK DNS server configured"
    else
        warn "No DNS server configured."
    fi
}

# Internet connectivity test
test_internet() {
    log "Testing internet reachability..."
    
    if ping -c 2 -W 2 8.8.8.8 >/dev/null 2>&1; then
        log "OK Internet reachable (8.8.8.8)"
        return 0
    else
        warn "Internet not reachable (8.8.8.8)"
        return 1
    fi
}

main() {
    require_root
    
    echo "=========================================="
    echo "  Board DHCP setup"
    echo "=========================================="
    echo ""
    
    # Show MAC
    log "Checking eth0 MAC..."
    ETH0_MAC=$(get_eth0_mac)
    echo ""
    log "eth0 MAC: $ETH0_MAC"
    echo ""
    log "Use this MAC in scripts/pc/setup_pc_wired_gateway.sh on the PC."
    echo ""
    
    # Current config
    if check_current_config; then
        read -rp "Already in DHCP mode. Restart systemd-networkd now? (y/N): " restart_confirm
        if [[ "$restart_confirm" =~ ^[Yy]$ ]]; then
            enable_networkd_services
            restart_network
            verify_network
            test_internet
        else
            log "Cancelled."
            exit 0
        fi
    else
        # Switch to DHCP
        read -rp "Switch eth0 to DHCP mode? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            log "Cancelled."
            exit 0
        fi
        echo ""
        
        setup_dhcp
        enable_networkd_services
        restart_network
        verify_network
        
        echo ""
        log "Run internet connectivity test? (y/N): "
        read -rp "> " test_confirm
        if [[ "$test_confirm" =~ ^[Yy]$ ]]; then
            test_internet
        fi
    fi
    
    echo ""
    log "Done!"
    echo ""
    log "Summary:"
    log "  MAC: $ETH0_MAC"
    log "  Config file: $NETWORK_FILE"
    if [[ -f "$BACKUP_FILE" ]]; then
        log "  Backup file: $BACKUP_FILE"
    fi
    echo ""
    log "Next on PC: run sudo /root/pyro_vision/scripts/pc/setup_pc_wired_gateway.sh"
    log "  sudo /root/pyro_vision/scripts/pc/setup_pc_wired_gateway.sh"
    echo ""
}

main "$@"
