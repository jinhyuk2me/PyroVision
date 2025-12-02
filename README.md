# Vision AI NXP

NXP i.MX8M Plus 기반 듀얼 카메라(RGB + IR) 실시간 화재 감지 시스템

## 개요

이 프로젝트는 NXP i.MX8M Plus EVK 보드에서 동작하는 실시간 비전 AI 시스템입니다.  
RGB 카메라와 IR 열화상 카메라를 동시에 활용하여 화재를 감지하고, NPU 가속을 통해 YOLOv8 객체 탐지를 수행합니다.

### 주요 기능

- **듀얼 카메라 지원**: RGB(IMX219 via MIPI CSI) + IR(PureThermal via USB)
- **NPU 가속 추론**: i.MX8MP NPU를 활용한 YOLOv8 TFLite 모델 추론 (~8 FPS)
- **실시간 화점 탐지**: IR 열화상 데이터 기반 온도 분석을 통한 화점 검출
- **TCP 스트리밍**: 감지 결과를 원격 서버로 실시간 전송
- **런타임 카메라 제어**: 키보드를 통한 회전/반전 실시간 조정

## 시스템 요구사항

### 하드웨어

| 구성 요소 | 사양 |
|-----------|------|
| 보드 | NXP i.MX8M Plus EVK |
| RGB 카메라 | IMX219 (MIPI CSI 연결) |
| IR 카메라 | PureThermal (USB, VID:PID = 1e4e:0100) |

### 소프트웨어

- Python 3.10+
- OpenCV (GStreamer 백엔드 포함)
- TFLite Runtime
- libuvc
- libvx_delegate.so (NPU 델리게이트)
- PyQt6 (GUI 모드 사용 시)

### 의존성 설치 예시

```bash
# 시스템 패키지 (Ubuntu 기준)
sudo apt install python3-opencv python3-pyqt6 libuvc-dev

# Python 패키지
python3 -m pip install --upgrade pip
python3 -m pip install tflite-runtime==2.13.0 pyyaml numpy matplotlib
```

> 보드 환경에서는 BSP에 포함된 OpenCV/GStreamer, NPU delegate(`libvx_delegate.so`)를 사용하고, PC 개발 환경에서는 위 예시처럼 pip/apt로 설치하면 됩니다.

## 프로젝트 구조

```
vision-ai-nxp/
├── app.py                    # 메인 애플리케이션 진입점
├── sender.py                 # TCP 이미지 송신 모듈
├── (optional) docs/pyqt_gui_design.md  # GUI 구성 문서
│
├── camera/                   # 카메라 드라이버
│   ├── rgbcam.py            # RGB 카메라 (V4L2/GStreamer)
│   ├── ircam.py             # IR 열화상 카메라 + 화점탐지
│   └── purethermal/         # PureThermal libuvc 바인딩
│       ├── thermalcamera.py
│       └── uvctypes.py
│
├── detector/                 # 객체 탐지 모듈
│   ├── tflite.py            # YOLOv8 TFLite 추론 워커
│   ├── fire.py              # 화재 판정 로직
│   └── objdet.py            # 객체 탐지 유틸리티
│
├── core/                     # 핵심 유틸리티
│   ├── buffer.py            # DoubleBuffer
│   ├── state.py             # CameraState (카메라 회전/반전 상태)
│   └── util.py              # 공통 유틸리티 함수
│
├── configs/                  # 설정 파일
│   ├── config.yaml          # 메인 설정
│   └── get_cfg.py           # 설정 로더
│
├── model/                    # TFLite 모델 저장소
│   ├── labels.txt           # 클래스 라벨
│   ├── 8n_800_v2/           # YOLOv8n 800px 모델
│   ├── 8n_640_v3/           # YOLOv8n 640px 모델
│   └── ...                  # 기타 해상도/버전 모델
│
└── save/                     # 이미지 저장 디렉토리
    ├── lwir/                # IR 이미지
    └── visible/             # RGB 이미지
```

> 참고: 예전 시각화 도구는 `backup/vis.py`에 남아 있으며, 현재 기본 파이프라인에는 포함되지 않습니다.

## 설정

`configs/config.yaml` 파일에서 시스템 설정을 변경할 수 있습니다:

```yaml
CAMERA:
  IR:
    FPS: 9
    RES: [160, 120]
    SLEEP: 0.11
  RGB_FRONT:
    FPS: 30
    RES: [1920, 1080]
    SLEEP: 0.033

TARGET_RES: [960, 540]              # 전송/표시 해상도

MODEL: /root/vision-ai-nxp/model/8n_800_v2/best_full_integer_quant.tflite
LABEL: /root/vision-ai-nxp/model/labels.txt
DELEGATE: "/usr/lib/libvx_delegate.so"

STATE:
  FIRE:
    NMS: 0.1
    THRESHOLD: 60
    CONFIDENCE: 0.2
    MIN_DUR: 10.0

SERVER:
  IP: '192.168.10.1'
  PORT: 9999
  COMP_RATIO: 70                    # JPEG 압축 품질
```

위 값들은 i.MX8MP EVK에서 RGB/IR 센서를 직접 연결했을 때의 기본 프로파일입니다.  
개발 PC나 다른 보드에서 실행하려면 용도별 설정 파일을 나눠 두는 편이 편리합니다.

### 환경별 설정 프로파일

| 용도 | 파일 | 특징 |
|------|------|------|
| i.MX8MP EVK 기본 | `configs/config.yaml` | 보드 장치 노드(`/dev/video3`, `/dev/video5`)와 VX delegate 경로가 포함됨 |
| 개발 PC(웹캠 + USB IR) | `configs/config_pc.yaml` | 내장/USB 카메라 인덱스, CPU 추론, 로컬 `model/` 경로 사용 |

필요 시 위 파일을 복사해 새 YAML을 만들고 장치 경로·모델·캡처 폴더를 자유롭게 수정하세요.

### CONFIG_PATH로 설정 선택

`configs/get_cfg.py`는 기본적으로 레포지토리 안의 `configs/config.yaml`을 읽습니다.  
다른 프로파일을 쓰려면 `CONFIG_PATH` 환경 변수를 지정하세요.

```bash
# 보드 기본값 (생략해도 동일)
CONFIG_PATH=configs/config.yaml python3 app.py

# 개발 PC 프로파일로 GUI 실행
CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py

# 동일한 프로파일로 캡처 스크립트 실행
CONFIG_PATH=configs/config_pc.yaml python3 capture.py --output ./capture_pc
```

GUI/CLI/capture는 모두 같은 설정 로더를 사용하므로 한 번만 지정하면 됩니다.

### 카메라 장치 및 입력 소스 팁

- `CAMERA.IR.DEVICE`, `CAMERA.RGB_FRONT.DEVICE`에 보드용 `/dev/video*` 경로 또는 PC용 장치 인덱스를 넣습니다.
- `INPUT.RGB.DEVICE`, `INPUT.IR.DEVICE`를 지정하면 FrameSource가 우선적으로 해당 장치를 사용하니, 테스트 중 장치를 빠르게 교체할 수 있습니다.
- `INPUT.*.MODE`를 `video`로 두면 `VIDEO_PATH`에 문자열 또는 리스트를 지정해 플레이리스트를 순차/루프 재생할 수 있습니다.
- `MODE=mock`은 하드웨어 없이도 파이프라인과 GUI를 실행할 수 있는 가상 소스입니다.

## 사용법

### 실행

```bash
# 리포지토리 루트에서 실행 (예: /home/yocto/work/lk_fire)
cd /home/yocto/work/lk_fire
python3 app.py
# GUI 모드 (PyQt6 필요)
APP_MODE=gui python3 app.py
# 또는 python3 app.py --mode gui
```

> 로그 레벨은 `LOG_LEVEL` 환경변수(기본 `INFO`)로 조정할 수 있습니다.  
> 예: `LOG_LEVEL=WARNING python3 app.py`

- **보드 실행 예시**: `CONFIG_PATH=configs/config.yaml python3 app.py`
- **PC 실행 예시**: `CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py`

#### GUI 모드 기능 요약

PyQt6 기반 GUI는 실시간 프리뷰와 런타임 제어판을 제공합니다.

- **입력 제어**: RGB/IR에 대해 `live / video / mock`, 영상 경로, Loop 여부를 수정한 뒤 `Apply Input Settings`로 즉시 반영.
- **캡처 파이프라인**: 출력 경로, 녹화 시간/프레임을 설정해 `Start Capture`를 누르면 같은 설정으로 `capture.py`가 서브프로세스로 실행됩니다.
- **송신·표시 토글**: `Start Sender`, `Start Display` 버튼으로 TCP 전송과 HDMI/윈도우 미리보기를 각각 켜고 끌 수 있습니다.
- **좌표 보정 워크플로우**: Offset/Scale 스핀박스와 `+/-` Nudge 버튼으로 IR↔RGB 정렬을 조정하고, Overlay Preview에서 즉시 결과를 확인합니다.
- **상태 모니터링**: RGB/IR FPS 롤링 플롯, SYNC 경고, 최근 로그 텍스트를 동시에 확인할 수 있습니다.

GUI 구성과 단계별 동작은 `docs/pyqt_gui_design.md`에서 더 자세히 설명되어 있습니다.

#### 환경 변수 오버라이드

입력 소스를 자주 바꿔야 한다면 환경 변수로 빠르게 덮어쓸 수 있습니다.

```
RGB_INPUT_MODE=video RGB_VIDEO_PATH=/data/demo.mp4 RGB_LOOP=false \
IR_INPUT_MODE=mock LOG_LEVEL=DEBUG python3 app.py
```

| 변수 | 설명 |
|------|------|
| `CONFIG_PATH` | 사용할 설정 파일 경로 (`configs/config_pc.yaml` 등) |
| `APP_MODE` | `cli` / `gui` 모드 선택 (`--mode` 인자와 동일) |
| `LOG_LEVEL` | `DEBUG/INFO/WARNING/...` 로거 레벨 |
| `RGB_INPUT_MODE`, `IR_INPUT_MODE` | 각 입력을 `live/video/mock` 중 하나로 지정 |
| `RGB_VIDEO_PATH`, `IR_VIDEO_PATH` | 단일 경로나 `;`로 구분한 다중 경로 |
| `RGB_LOOP`, `IR_LOOP` | `true/false` 문자열로 루프 재생 여부 제어 |
| `RGB_FRAME_INTERVAL_MS`, `IR_FRAME_INTERVAL_MS` | 영상 재생 간격(ms)을 강제 지정 |

### HDMI 디스플레이

- `configs/config.yaml`의 `DISPLAY.ENABLED`를 `true`로 설정하면 보드 HDMI에 로컬 미리보기 창이 표시됩니다.
- 기본 창 제목은 `DISPLAY.WINDOW_NAME`으로 바꿀 수 있으며, `q` 또는 `ESC`로 닫을 수 있습니다.

### IR/RGB 동기화

- `SYNC.ENABLED`를 `true`로 두면 IR과 RGB 검출 프레임의 타임스탬프 차이가 `SYNC.MAX_DIFF_MS`를 넘을 때 해당 패킷 전송을 건너뜁니다.
- 영상 파일/모의 입력 재생 시 두 스트림의 프레임을 최대한 맞춰서 전송하고 싶을 때 사용하세요.

### IR/EO 캡처

- `capture.py`를 실행하면 IR/EO 실시간 스트림을 동기화된 페어로 저장할 수 있습니다.
- `configs/config.yaml`의 `CAPTURE` 섹션에서 출력 경로, 녹화 시간, 허용 시간차 등을 설정하세요.

```bash
cd /root/vision-ai-nxp
python3 capture.py
```

- 다른 프로파일을 쓰려면 `CONFIG_PATH=configs/config_pc.yaml python3 capture.py --output ./capture_pc`처럼 지정하면 됩니다.
- GUI에서 `Start Capture`를 누르면 동일한 명령이 백그라운드에서 실행되어 촬영→테스트 파이프라인을 한 번에 돌릴 수 있습니다.

```yaml
CAPTURE:
  OUTPUT_DIR: "/data/capture/session1"
  DURATION_SEC: 60
  MAX_FRAMES: null
  MAX_DIFF_MS: 80
  SAVE_RGB_VIDEO: true
  SAVE_IR_VIDEO: true
  SAVE_IR_RAW16: true
  RGB_CODEC: "mp4v"
  IR_CODEC: "mp4v"
```

- 결과물로 `rgb.mp4`, `ir_vis.mp4`, `ir16/*.npy`, `metadata.csv`가 생성됩니다.
- 촬영 후에는 `INPUT`을 `video` 모드로 바꿔 방금 저장한 영상을 그대로 재생·테스트할 수 있습니다.

### 저장된 영상/이미지 재생

`INPUT` 섹션을 이용해 실시간 카메라 대신 영상 파일을 소스로 사용할 수 있습니다.

```yaml
INPUT:
  RGB:
    MODE: video                 # live | video | mock
    VIDEO_PATH: ["/data/rgb1.mp4", "/data/rgb2.mp4"]  # 문자열 또는 리스트
    LOOP: true                  # playlist 반복 여부
    FRAME_INTERVAL_MS: 33       # null이면 카메라 설정(SLEEP) 사용
    COLOR: [0, 255, 0]          # mock 모드 전용 (B,G,R)
  IR:
    MODE: video
    VIDEO_PATH: "/data/ir.mp4"
    LOOP: true
    FRAME_INTERVAL_MS: 100
```

- `MODE`가 `video`이면 지정된 `VIDEO_PATH`에서 프레임을 읽어 DoubleBuffer에 공급합니다.
- `VIDEO_PATH`는 단일 파일 경로나 리스트 모두 지원하며, 리스트일 경우 순차 재생합니다.
- `LOOP`를 `true`로 두면 파일 끝에 도달해도 다시 처음부터 재생합니다.
- `FRAME_INTERVAL_MS`를 지정하면 영상 파일 재생 속도를 강제로 맞출 수 있습니다(없으면 기본 SLEEP 사용).
- IR 소스는 8/16bit 영상을 자동으로 16bit RAW로 변환해 기존 hotspot/FireFusion 파이프라인을 그대로 사용할 수 있습니다.
- `MODE=mock`은 고정 패턴을 생성하는 가짜 소스로 테스트/CI에서 유용합니다.
- GUI에서 입력/캡처/좌표를 수정하는 방법은 위의 “GUI 모드 기능 요약” 절을 참고하세요.

### 키보드 제어

실행 중 터미널에서 다음 키를 눌러 카메라 방향을 실시간으로 조정할 수 있습니다.  
조정된 방향은 화면 표시와 화점 탐지 모두에 적용됩니다.

#### IR 카메라

| 키 | 기능 |
|----|------|
| `1` | IR 90도 회전 (시계방향, 0→90→180→270→0) |
| `2` | IR 좌우반전 토글 |
| `3` | IR 상하반전 토글 |

#### RGB 카메라

| 키 | 기능 |
|----|------|
| `4` | RGB 90도 회전 (시계방향) |
| `5` | RGB 좌우반전 토글 |
| `6` | RGB 상하반전 토글 |

#### 공통

| 키 | 기능 |
|----|------|
| `7` | 두 카메라 동시 좌우반전 토글 |
| `8` | 두 카메라 동시 상하반전 토글 |
| `s` | 현재 상태 확인 |
| `h` | 도움말 표시 |
| `q` | 종료 |

#### 카메라 방향 조정 예시

카메라가 90도 회전되어 설치된 경우:
```
[IR] Rotation: 90 degrees      # '1' 키 입력
```

영상이 좌우 반대인 경우:
```
[IR] Horizontal flip: ON       # '2' 키 입력
[RGB] Horizontal flip: ON      # '5' 키 입력
```

현재 상태 확인:
```
[Status] IR:  rotate= 90, flip_h=ON, flip_v=OFF
[Status] RGB: rotate=  0, flip_h=ON, flip_v=OFF
```

### 출력 예시

```
=======================================================
Keyboard Controls:
-------------------------------------------------------
  IR Camera:
    [1] Rotate IR 90 degrees (clockwise)
    [2] Toggle IR horizontal flip (left-right)
    [3] Toggle IR vertical flip (up-down)
-------------------------------------------------------
  RGB Camera:
    [4] Rotate RGB 90 degrees (clockwise)
    [5] Toggle RGB horizontal flip (left-right)
    [6] Toggle RGB vertical flip (up-down)
-------------------------------------------------------
  Both Cameras:
    [7] Toggle BOTH horizontal flip
    [8] Toggle BOTH vertical flip
-------------------------------------------------------
  [s] Show current status
  [h] Show this help message
  [q] Quit application
=======================================================

IRCam - Starting
RGBCam - Starting
Connected to /dev/video5
Resolution - requested: 1920x1080, reported: 1920x1080, actual: 1920x1080
RGB-TFLite - Starting
[DetRGB] VX delegate 로드: /usr/lib/libvx_delegate.so
[DetRGB] TFLite accel=NPU, threads=1
TCP Sender - Starting
[Sender] Connected to 192.168.10.1:9999

[IRCam] Captured 100 frames
[DetRGB] NPU | FPS= 7.88 (target=30.0) | total= 125.1 ms | invoke= 95.3 ms
[Sender] Frame: 82, FPS: 1.42, Mode: DISPLAY, Packet: 25.2KB
  → Images: ['ir', 'rgb_det']
```

## 아키텍처

### 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              i.MX8MP EVK                                │
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│  │  IR Camera   │     │  RGB Camera  │     │   TCP Sender         │   │
│  │ (PureThermal)│     │   (IMX219)   │     │                      │   │
│  └──────┬───────┘     └──────┬───────┘     │  ┌────────────────┐  │   │
│         │                    │             │  │ rgb_det (JPEG) │──┼───┼──→ 항상 전송
│         ▼                    ▼             │  ├────────────────┤  │   │
│  ┌──────────────┐     ┌──────────────┐     │  │ ir (raw)       │──┼───┼──→ 항상 전송
│  │  IRCamera    │     │ FrontRGBCam  │     │  ├────────────────┤  │   │
│  │  - capture   │     │  - capture   │     │  │ rgb (JPEG)     │──┼───┼──→ 저장모드만
│  │  - fire det  │     │              │     │  ├────────────────┤  │   │
│  │  - colormap  │     │              │     │  │ ir16 (raw16)   │──┼───┼──→ 저장모드만
│  └──────┬───────┘     └──────┬───────┘     │  └────────────────┘  │   │
│         │                    │             └──────────────────────┘   │
│         ▼                    ▼                                        │
│    ┌─────────┐          ┌─────────┐                                   │
│    │  d_ir   │          │  d_rgb  │  DoubleBuffer                     │
│    │ d16_ir  │          │         │                                   │
│    └─────────┘          └────┬────┘                                   │
│                              │                                        │
│                              ▼                                        │
│                    ┌──────────────────┐                               │
│                    │  TFLiteWorker    │                               │
│                    │  (YOLOv8 + NPU)  │                               │
│                    │  - letterbox     │                               │
│                    │  - inference     │                               │
│                    │  - NMS           │                               │
│                    │  - draw boxes    │                               │
│                    └────────┬─────────┘                               │
│                             │                                         │
│                             ▼                                         │
│                        ┌─────────┐                                    │
│                        │d_rgb_det│  DoubleBuffer                      │
│                        └─────────┘                                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 탐지 클래스

| ID | 클래스명 | 설명 |
|----|----------|------|
| 0 | smoke | 연기 |
| 1 | fire | 화재 (기본 탐지 대상) |
| 2 | cloud | 구름 |
| 3 | fog | 안개 |
| 4 | light | 빛 |
| 5 | sunlight | 햇빛 |
| 6 | swaying_object | 흔들리는 물체 |
| 7 | swaying_leaves | 흔들리는 나뭇잎 |

## 성능

| 항목 | 수치 |
|------|------|
| RGB 입력 해상도 | 1920x1080 @ 30fps |
| IR 입력 해상도 | 160x120 @ 9fps |
| NPU 추론 속도 | ~8 FPS (YOLOv8n 800px) |
| 추론 지연 시간 | ~95ms (invoke) / ~125ms (total) |
| 전송 FPS | ~1.5 FPS (네트워크 의존) |

## 문제 해결

### USB 장치 인식 안됨

```bash
# USB 장치 확인
lsusb

# IR 카메라 확인 (VID:PID = 1e4e:0100)
lsusb | grep 1e4e
```

### 비디오 장치 확인

```bash
# V4L2 장치 목록
ls -la /dev/video*

# 장치 이름 확인
cat /sys/class/video4linux/*/name
```

### uvc_find_device error

IR 카메라가 연결되지 않았거나 인식되지 않음:
1. USB 케이블 연결 확인
2. `lsusb`로 장치 인식 확인
3. 다른 USB 포트 시도

### Segmentation fault

- IR 카메라 초기화 실패 시 발생 가능
- `camera/ircam.py`의 `capture()` 반환값 확인

## 라이선스

이 프로젝트는 내부 사용 목적으로 개발되었습니다.

## 기여자

- Vision AI Team

## Legacy 정리 내역

- 기존 YOLOv4-tiny ONNX 추론 경로(`detector/objdet.py`, `detector/fire.py`, `darknetonnx/`)는 더 이상 유지되지 않으며, 현재 버전은 YOLOv8 TFLite 파이프라인만 지원합니다.
- `backup/` 디렉터리는 현재 TFLite 비디오 테스트(`backup/test.py`)와 시각화 도구(`backup/vis.py`)만 포함하며, 나머지 실험 스크립트/샘플 이미지는 정리되었습니다.
