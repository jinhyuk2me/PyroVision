# PyroVision

NXP i.MX8M Plus 환경을 우선 대상으로 하는 듀얼 카메라 화재 감지 파이프라인입니다. RGB(객체 탐지)와 IR(화점 탐지)을 동시에 처리하고, TCP로 전송하거나 GUI에서 모니터링/제어할 수 있습니다.

## 핵심 기능
- 듀얼 카메라: RGB(V4L2/GStreamer) + IR(PureThermal, Y16)
- YOLOv8 TFLite 추론: NPU delegate 사용 가능, CPU/XNNPACK 대체 경로
- IR 화점 탐지: RAW16 기반 온도 분석, 파라미터 런타임 조정
- GUI: 탭 기반 설정(Input/Inference/IR Hotspot/Overlay/Capture), 실시간 프리뷰/플롯/로그
- CLI: 키보드 단축키로 카메라 회전/반전 제어
- TCP 송신: RGB/IR/IR16/Det 프레임을 JPEG로 전송
- 캡처: RGB/IR 비디오, RAW16 npy, 메타데이터, 옵션으로 추론 결과 JSONL

## 요구사항
### 하드웨어
- 보드: NXP i.MX8M Plus (Vivante NPU) 권장
- 카메라: RGB `/dev/video*`, IR PureThermal(`/dev/video*`, VID:PID 1e4e:0100)

### 소프트웨어
- Python 3.10+
- OpenCV (GStreamer 지원), PyQt6(GUI), tflite-runtime, pyyaml, numpy
- 보드용 NPU delegate `.so` (예: `/usr/lib/libvx_delegate.so`)

## 설치 / 환경 준비
### PC(개발/테스트)
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. 기본 PC 프로파일은 `configs/config_pc.yaml` (모델/라벨 경로는 `./model/` 기준). 다른 프로파일을 쓰려면 `CONFIG_PATH` 환경변수로 지정.

### 연결/네트워크 설정 요약
- **유선 공유 + 고정 IP**: PC에서 `scripts/pc/setup_pc_wired_gateway.sh` 실행 후 보드 eth0를 DHCP로 전환(`scripts/board/setup_board_dhcp.sh`). 실행 시 프롬프트로 보드 MAC/IP를 입력하면 dnsmasq의 `--dhcp-host`로 예약(IP 192.168.200.11/12/13 등).
- **무선 연결**: 보드에서 `scripts/board/setup_board_wifi.sh` 실행 → SSID 목록 선택/비밀번호 입력 → Wi‑Fi 연결. PC도 동일 SSID에 접속.
- 상세 흐름은 `docs/connection_guide.md` 참조.

### 보드(i.MX8M Plus)
- BSP에 포함된 OpenCV / tflite-runtime 사용을 권장합니다. `requirements.txt`의 `opencv-python-headless`/`tflite-runtime` 라인은 PC 기본값이므로, 보드에서 충돌 시 생략하고 `numpy`, `pyyaml` 등만 설치하세요.
- delegate 경로(`DELEGATE`)가 실제 `.so` 파일을 가리키는지 확인하세요. 보통 `/usr/lib/libvx_delegate.so`.
- 나머지 Python 패키지는 `pip install -r requirements.txt`로 설치 가능하나, 보드별 패키지 제공 정책에 따라 조정하세요.

## 실행 방법

### 기본 실행
```bash
# CLI 모드 (기본)
CONFIG_PATH=configs/config.yaml python3 app.py

# CLI + Display 모드 (OpenCV 윈도우 표시)
CONFIG_PATH=configs/config.yaml python3 app.py
# config.yaml에서 DISPLAY.ENABLED: true 설정 필요

# GUI 모드
CONFIG_PATH=configs/config.yaml APP_MODE=gui python3 app.py
# 또는
CONFIG_PATH=configs/config.yaml python3 app.py --mode gui
```

### 프로파일 선택 예시
- 고해상도/민감 설정(기본): `CONFIG_PATH=configs/config.yaml python3 app.py`
- 보수 설정(낮은 해상도/높은 IR 임계): `CONFIG_PATH=configs/config_prod.yaml python3 app.py`

### 환경변수 옵션

#### 필수 환경변수
- `CONFIG_PATH`: 설정 파일 경로 (기본값: `configs/config.yaml`)

#### 실행 모드 제어
- `APP_MODE`: 실행 모드 선택
  - `cli`: CLI 모드 (기본값, 키보드 제어 가능)
  - `gui`: GUI 모드 (PyQt6 필요)

#### 입력 소스 오버라이드 (RGB)
- `RGB_INPUT_MODE`: RGB 입력 모드
  - `live`: 실제 카메라 (기본값)
  - `video`: 비디오 파일
  - `mock`: 테스트용 모의 입력
- `RGB_VIDEO_PATH`: 비디오 파일 경로 (MODE=video일 때)
  - 단일 파일: `/path/to/video.mp4`
  - 여러 파일: `/path/to/video1.mp4;/path/to/video2.mp4` (세미콜론 구분)
- `RGB_LOOP`: 비디오 반복 재생 (true/false, 1/0)
- `RGB_FRAME_INTERVAL_MS`: 프레임 간격 (밀리초, 재생 속도 조절)
- `RGB_DEVICE`: 카메라 장치 오버라이드 (예: 0, /dev/video5)

#### 입력 소스 오버라이드 (IR)
- `IR_INPUT_MODE`: IR 입력 모드
  - `live`: 실제 카메라 (기본값)
  - `video`: 비디오 파일
  - `mock`: 테스트용 모의 입력
- `IR_VIDEO_PATH`: 비디오 파일 경로 (MODE=video일 때)
- `IR_LOOP`: 비디오 반복 재생 (true/false, 1/0)
- `IR_FRAME_INTERVAL_MS`: 프레임 간격 (밀리초)

#### 화재 탐지 시각화 모드
- `FUSION_VIS_MODE`: 화재 annotation 표시 모드
  - `test`: 모든 탐지 결과 표시 (기본값)
    - EO-only (RGB만 탐지): 노란색 박스
    - 확정 화재 (IR+RGB 탐지): 빨간색 박스
  - `temp`: 확정 화재만 노란색으로 표시
    - EO-only 박스는 숨김
    - 확정 화재는 노란색으로 표시

### 실행 예제

#### 1. 보드에서 실제 카메라로 실행 (CLI)
```bash
CONFIG_PATH=configs/config.yaml python3 app.py
```

#### 2. PC에서 GUI로 실행
```bash
CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py
```

#### 3. 비디오 파일로 테스트 (RGB만)
```bash
RGB_INPUT_MODE=video RGB_VIDEO_PATH=/data/fire.mp4 RGB_LOOP=true IR_INPUT_MODE=mock python3 app.py
```

#### 4. 여러 비디오 파일 순차 재생
```bash
RGB_INPUT_MODE=video RGB_VIDEO_PATH="/data/video1.mp4;/data/video2.mp4" python3 app.py
```

#### 5. 특정 카메라 장치 지정
```bash
RGB_DEVICE=/dev/video5 python3 app.py
```

## 설정/프로파일
- 기본: `configs/config.yaml` (고해상도/민감 설정)
- 보수: `configs/config_prod.yaml` (낮은 해상도/높은 IR 임계)
- PC: `configs/config_pc.yaml`
- 핵심 키: `CAMERA.*.DEVICE`(udev 링크 `/dev/pyro_rgb_cam`, `/dev/pyro_ir_cam` 사용), `MODEL`/`LABEL`/`DELEGATE` 경로, `TARGET_RES`, `SERVER.IP/PORT`, `DISPLAY.ENABLED`.
- 장치 링크는 `scripts/board/setup_pyro_vision.sh`로 생성/갱신됩니다. 필요 시 `CONFIG_PATH`로 다른 프로파일을 지정하세요.

## CLI 키보드 제어

CLI 모드에서는 키보드로 실시간 제어가 가능합니다. 앱을 포그라운드로 실행해야 합니다.

### IR 카메라 제어
- `1`: IR 시계방향 90도 회전
- `2`: IR 좌우반전 토글
- `3`: IR 상하반전 토글

### RGB 카메라 제어
- `4`: RGB 시계방향 90도 회전
- `5`: RGB 좌우반전 토글
- `6`: RGB 상하반전 토글

### 양쪽 카메라 제어
- `7`: 양쪽 좌우반전 토글
- `8`: 양쪽 상하반전 토글

### 기타
- `s`: 현재 상태 표시
- `h`: 도움말 표시
- `q`: 애플리케이션 종료

**참고:**
- 백그라운드(&)로 실행하면 키보드 입력이 작동하지 않습니다
- SSH 터미널에서 직접 키를 누르면 즉시 반영됩니다
- Config 파일의 ROTATE/FLIP 설정은 초기값만 지정하고, 런타임에 키보드로 자유롭게 변경 가능합니다

## GUI 모드

### 실행
```bash
CONFIG_PATH=configs/config.yaml APP_MODE=gui python3 app.py
```

### 탭 구성
- **Input**: RGB/IR 모드(live/video/mock), 경로, Loop, Device 선택(+Browse 버튼)
- **Inference**: 모델/라벨/Delegate 경로, 클래스 필터(smoke/fire), 적용 시 탐지 워커 재시작
- **IR Hotspot**: 화점 탐지 on/off, MinTemp, Thr, RawThr, Tau 런타임 적용
- **Overlay**: IR↔RGB 정렬 Offset/Scale 조정, Nudge 버튼
- **Capture**: 출력 경로/Duration/MaxFrames 설정, `Start Capture`로 `capture.py` 실행

### 상단 버튼
- IR/RGB 회전 버튼 (90도씩)
- Start/Stop Sender (TCP 전송)
- Start/Stop Capture (캡처)

### 화면 구성
- **프리뷰**: RGB/Det/IR/Overlay 4분할 화면
- **상태 라벨**: Det/IR/RGB FPS, SYNC 상태 표시
- **플롯**: Det/RGB/IR FPS 롤링 그래프
- **로그 창**: GUI 이벤트/오류 표시

## 성능 최적화

### 임베디드 보드에서 성능 문제 발생 시

#### 1. RGB 해상도 낮추기
CPU 부담이 크면 해상도를 낮춰 성능을 확보하세요. 해상도는 V4L2 제약(너비 16의 배수, 높이 8의 배수)을 따릅니다.

#### 2. FPS 조정
프레임레이트를 낮추고 SLEEP을 늘려 여유를 확보합니다.

#### 3. TARGET_RES 조정
탐지 전 리사이징 해상도를 낮추면 NPU/CPU 부담이 줄어듭니다.

#### 4. JPEG 압축률 조정
네트워크 대역폭이 좁을 때는 JPEG 품질(압축률)을 조정합니다.

## 수신(Receiver)

### 기본 실행
```bash
python3 receiver.py
```

### 옵션
- 기본 포트: 9999
- 저장 경로:
  - RGB: `save/visible/`
  - IR: `save/lwir/`

송신 측(SERVER.IP, SERVER.PORT)과 수신 측 포트를 맞춰주세요.

## 캡처 & 재사용

### CLI에서 캡처
```bash
python3 capture.py --output ./capture_session [--duration SEC] [--max-frames N] [--save-det]
```

### 옵션
- `--output DIR`: 출력 디렉토리
- `--duration SEC`: 캡처 지속 시간 (초)
- `--max-frames N`: 최대 프레임 수
- `--save-det`: 탐지 결과도 저장 (JSONL)

### 저장되는 파일
- `rgb.mp4`: RGB 비디오
- `ir_vis.mp4`: IR 가시화 비디오
- `ir16/*.npy`: IR RAW16 데이터 (프레임별)
- `metadata.csv`: 타임스탬프 및 메타데이터
- `det.jsonl`: 탐지 결과 (옵션)

### 재사용 예제
```python
from utils.capture_loader import CaptureLoader

for item in CaptureLoader("./capture_session"):
    rgb = item["rgb"]           # RGB 프레임 (numpy array)
    ir = item["ir"]             # IR 가시화 프레임
    ir_raw = item["ir_raw"]     # IR RAW16 데이터
    # 처리...
```

## 테스트

### 기본 테스트
```bash
pip install -r requirements-dev.txt
pytest
```

### 특정 테스트
```bash
# 비디오 소스 테스트 (sample/fire_sample.mp4 필요)
pytest tests/test_video_sources.py

# 특정 테스트 케이스
pytest tests/test_video_sources.py::test_rgb_video_source
```

## NPU/Delegate

### Delegate 사용
- `DELEGATE` 경로가 지정되어 있으면 NPU 사용 시도
- `.so` 파일 로드 실패 시 자동으로 CPU/XNNPACK으로 폴백
- i.MX8M Plus: `/usr/lib/libvx_delegate.so`

### CPU 전용 모드
```yaml
DELEGATE: ""    # 빈 문자열 또는 주석 처리
```

## 보드 자동 설정 및 서비스 등록
- 스크립트: `scripts/board/setup_pyro_vision.sh` (root 권한 필요)
- 수행 내용:
  1) V4L2 장치 스캔 → PureThermal IR(index0)/RGB(VIV 등) 자동 감지  
  2) udev 규칙 생성: `/dev/pyro_ir_cam`, `/dev/pyro_rgb_cam` 심볼릭 링크 부여  
  3) `configs/config.yaml`의 `CAMERA.IR/RGB_FRONT.DEVICE`를 위 링크로 갱신  
  4) `pyro_vision.service`를 `/etc/systemd/system`에 설치하고 enable+start
- 실행:
```bash
sudo bash scripts/board/setup_pyro_vision.sh
sudo systemctl status pyro_vision.service
```
- 검증 팁:
  - 장치/링크: `v4l2-ctl --list-devices`, `ls -l /dev/pyro_*`
  - 로그: `journalctl -u pyro_vision.service -n 50 --no-pager`

## 자주 겪는 문제

### RGB 장치가 다시 안 열릴 때
1. 장치 점유 확인: `fuser /dev/video5`
2. VideoCapture.release() 확인 (최근 수정으로 처리됨)
3. 컨테이너 실행 시 `--device /dev/videoX` 포함 확인

### IR가 mock→live 전환 후 멈출 때
1. `IRCamera.stop()`에서 cleanup 완료 (최근 수정됨)
2. `/dev/video*` 인덱스 변동 여부 확인
3. `v4l2-ctl --list-devices`로 장치 확인

### Delegate 로드 실패로 CPU로 떨어질 때
1. `DELEGATE` 경로 존재 여부 확인
2. 보드용 `.so` 파일인지 확인
3. 로그에서 delegate 로드 메시지 확인

### GStreamer 파이프라인 실패
1. RGB 해상도가 Step 제약을 만족하는지 확인 (너비: 16의 배수, 높이: 8의 배수)
2. GStreamer 플러그인 설치 확인: `gst-inspect-1.0 v4l2src`
3. 장치가 NV12 포맷을 지원하는지 확인: `v4l2-ctl -d /dev/video5 --list-formats-ext`

### PyQt6 버전 충돌 (GUI 모드)
1. 시스템 Qt 버전과 PyQt6 버전 확인
2. 보드에서는 CLI 모드 사용 권장 (GUI는 개발/테스트용)
3. `pip install PyQt6==X.Y.Z` 버전 다운그레이드 시도

### CPU 사용량이 높을 때
1. RGB 해상도 낮추기 (성능 최적화 섹션 참고)
2. FPS 낮추기
3. TARGET_RES 낮추기
4. 불필요한 기능 비활성화 (DISPLAY, SYNC 등)

## 리포지토리 구조

```
pyro_vision/
├── app.py                  # 메인 엔트리 포인트
├── capture.py              # 캡처 스크립트
├── receiver.py             # TCP 수신 서버
├── sender.py               # TCP 송신 모듈
├── camera/                 # 카메라 소스
│   ├── rgbcam.py          # RGB 카메라
│   ├── ircam.py           # IR 카메라 인터페이스
│   ├── purethermal/       # PureThermal 드라이버
│   ├── frame_source.py    # 프레임 소스 베이스 클래스
│   └── device_selector.py # 장치 자동 선택
├── detector/
│   └── tflite.py          # YOLOv8 TFLite 워커
├── gui/
│   └── app_gui.py         # PyQt6 GUI
├── core/
│   ├── state.py           # 카메라 상태 관리
│   └── util.py            # 유틸리티 함수
├── configs/
│   ├── config.yaml        # 보드용 설정
│   ├── config_pc.yaml     # PC용 설정
│   ├── schema.py          # 설정 스키마
│   └── get_cfg.py         # 설정 로더
├── utils/
│   └── capture_loader.py  # 캡처 재생 로더
├── tests/                 # 테스트
└── model/                 # TFLite 모델 및 라벨
```

## 라이센스

Copyright (c) 2025 MomentLab. All Rights Reserved.
