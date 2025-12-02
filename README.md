# Vision AI (RGB + IR Fire Detection)

NXP i.MX8M Plus 환경을 우선 대상으로 하는 듀얼 카메라 화재 감지 파이프라인입니다. RGB(객체 탐지)와 IR(화점 탐지)을 동시에 처리하고, TCP로 전송하거나 GUI에서 모니터링/제어할 수 있습니다.

## 핵심 기능
- 듀얼 카메라: RGB(V4L2/GStreamer) + IR(PureThermal, Y16)
- YOLOv8 TFLite 추론: NPU delegate 사용 가능, CPU/XNNPACK 대체 경로
- IR 화점 탐지: RAW16 기반 온도 분석, 파라미터 런타임 조정
- GUI: 탭 기반 설정(Input/Inference/IR Hotspot/Overlay/Capture), 실시간 프리뷰/플롯/로그
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

## 설정
`CONFIG_PATH` 환경변수로 프로파일을 선택합니다.
- `configs/config.yaml`: 보드용(VX delegate 경로 포함)
- `configs/config_pc.yaml`: PC용(CPU 추론, 로컬 모델 경로)

예)
```
CONFIG_PATH=configs/config.yaml python3 app.py              # CLI
CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py
```

입력 소스 빠른 오버라이드:
```
RGB_INPUT_MODE=video RGB_VIDEO_PATH=/data/rgb.mp4 IR_INPUT_MODE=mock python3 app.py
```

## 실행 모드
- 기본: CLI (터미널 단축키로 회전/반전)
- GUI: `APP_MODE=gui` 또는 `--mode gui`

## GUI 안내 (탭)
- Input: RGB/IR 모드(live/video/mock), 경로, Loop, Device 선택(+Browse 버튼)
- Inference: 모델/라벨/Delegate 경로, 클래스 필터(smoke/fire), 적용 시 탐지 워커 재시작
- IR Hotspot: 화점 탐지 on/off, MinTemp, Thr, RawThr, Tau 런타임 적용
- Overlay: IR↔RGB 정렬 Offset/Scale 조정, Nudge 버튼
- Capture: 출력 경로/Duration/MaxFrames 설정, `Start Capture`로 `capture.py` 실행

상단 버튼: IR/RGB 90° 회전, Start/Stop Sender(TCP), Start/Stop Capture.  
프리뷰: RGB/Det/IR/Overlay 4분할, 상태 라벨에 Det/IR/RGB FPS, SYNC 표시.  
플롯: Det/RGB/IR FPS 롤링 그래프. 로그 창: GUI 이벤트/오류 표시.

## 캡처 & 재사용
`python3 capture.py --output ./capture_session [--duration SEC] [--max-frames N] [--save-det]`
- 저장물: `rgb.mp4`, `ir_vis.mp4`, `ir16/*.npy`, `metadata.csv(index,rgb_ts,ir_ts,diff_ms,ir_raw)`, `det.jsonl`(옵션)
- GUI `Start Capture`는 동일 스크립트를 서브프로세스로 실행합니다(현재 --save-det 미전달).
- 로더: `utils/capture_loader.py`
  ```python
  from utils.capture_loader import CaptureLoader
  for item in CaptureLoader("./capture_session"):
      rgb = item["rgb"]; ir = item["ir"]; ir_raw = item["ir_raw"]
  ```

## NPU/Delegate
- `DELEGATE`가 비어 있으면 CPU/XNNPACK 경로, 지정 시 존재하는 `.so`를 로드해 NPU 사용(로드 실패 시 CPU로 폴백).
- i.MX8 보드에서는 BSP에 맞는 delegate 경로를 `DELEGATE`에 설정하세요.

## 자주 겪는 문제
- RGB 장치가 다시 안 열릴 때: 최근 수정으로 stop 시 `VideoCapture.release()` 처리됨. 그래도 안 되면 장치 점유/fuser 확인 후 컨테이너 실행 시 `--device /dev/videoX` 포함 여부 확인.
- IR가 mock→live 전환 후 멈출 때: `IRCamera.stop()`에서 장치 cleanup 추가 완료. 여전히 문제면 `/dev/video*` 인덱스 변동 여부 점검.
- delegate 로드 실패로 CPU로 떨어질 때: `DELEGATE` 경로 존재 여부 확인, 보드용 `.so`로 교체.

## 리포지토리 구조(요약)
- `app.py` 메인 엔트리 (CLI/GUI 공용)
- `camera/` RGB/IR 소스, PureThermal 드라이버
- `detector/tflite.py` YOLOv8 TFLite 워커
- `gui/app_gui.py` PyQt6 GUI
- `capture.py` 캡처 스크립트 (옵션 추론 저장)
- `utils/capture_loader.py` 캡처 재생 로더

