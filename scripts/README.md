# Scripts

PC/보드용 유틸리티 스크립트 모음. 기본 경로: `/root/pyro_vision/scripts`.

## 디렉토리
- `pc/`: PC용 (인터넷 공유, 파일 동기화, 원격 DHCP 실행)
- `board/`: 보드용 (DHCP/Wi‑Fi 설정, 서버 IP 전환, 서비스/udev 설정)

## 빠른 실행 흐름
1) PC: 유선 게이트웨이/DHCP 시작 → `sudo ./pc/setup_pc_wired_gateway.sh` (MAC 자동 감지 `auto` 지원)  
2) 보드: DHCP 전환 → `sudo ./board/setup_board_dhcp.sh` (또는 PC에서 `bash ./pc/run_board_dhcp_over_ssh.sh`)  
3) 보드: 장치/udev/서비스 설정 → `sudo ./board/setup_pyro_vision.sh`  
4) 필요 시 서버 IP 전환 → `sudo ./board/set_server_ip.sh` (끝나면 `pyro_vision.service` 재시작 여부 질문)

## PC 스크립트
- `setup_pc_wired_gateway.sh`: PC를 공유기처럼 설정(NAT, dnsmasq). 인터페이스 선행 선택, MAC/IP 입력으로 고정 예약 생성, `auto` 입력 시 tcpdump 기반 MAC 자동 감지.
- `diagnose_board_ip.sh`: ARP/dnsmasq lease 조회로 보드 실제 IP 추적, SSH 포트 간단 확인.
- `scp_sync.sh`: 선택 경로를 보드로 주기적 scp 동기화.
- `run_board_dhcp_over_ssh.sh`: 보드의 `setup_board_dhcp.sh`를 SSH로 비대화형 실행.
- `linklocal_connect.sh`: 링크로컬(169.254.x.x) 네트워크(라우터 없이 케이블 직결 시 자동 부여)에서 IP를 수동 지정하고 SSH 등으로 접속할 때 보조.

## 보드 스크립트
- `setup_board_dhcp.sh`: eth0 DHCP 설정, systemd-networkd/resolved enable, 재시작 및 연결 테스트.
- `setup_board_wifi.sh`: Wi‑Fi 스캔/선택/접속. `SCAN_LIMIT`(기본 30, 0이면 무제한)으로 목록 제한. ctrl_interface 그룹/소켓 자동 설정, 필요 시 wpa_supplicant 기동.
- `setup_pyro_vision.sh`: 카메라 자동 감지(PureThermal 이름 매칭 포함), udev 규칙 생성 후 settle/링크 대기, config DEVICE 갱신, `pyro_vision.service` 설치/시작.
- `set_server_ip.sh`: `configs/config.yaml`의 `SERVER.IP`를 유선/무선/커스텀으로 변경 후 서비스 재시작 선택.
- `manage_service.sh`: 기본 대상 `pyro_vision.service` start/stop/restart/status.
- `switch_network_interface.sh`: eth0/mlan0 업/다운 조합 선택(유선만/무선만/둘다).

## 사용 팁
- Receiver 키맵: `1`(IR 90° 회전), `4`(RGB 90° 회전), `0`(회전 리셋), `[` `]` `{` `}` 등으로 스케일 조정.
- 셋업 후 카메라 링크 확인: `/dev/pyro_rgb_cam`, `/dev/pyro_ir_cam` 존재/심볼릭 확인.

## 요구사항/권한
- 대부분 root 필요: `sudo`로 실행.
- pc 게이트웨이: dnsmasq, iptables, tcpdump/timeout(auto MAC) 설치 필요.
- wifi 스크립트: wpa_supplicant, wpa_cli 필요.
- 설정 파일 백업은 필요 시 `.bak`로 별도 관리.
