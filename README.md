# PyroVision 🔥

> **Intelligent Fire Detection with Computer Vision**

![sample](https://github.com/Stellar-Moment/PyroVision/blob/refactor/gui-split/asset/fire_detected.gif?raw=true)

듀얼 카메라(RGB/IR) 기반 AI 화재 감지 시스템. RGB는 YOLOv8 TFLite로 화염을 추론하고, IR 열화상 카메라는 hotspot(온도 이상) 감지를 수행하며, 두 결과를 융합(Fusion)하여 정확한 화재 판정을 제공합니다. CLI/GUI/TCP 인터페이스를 통해 다양한 환경에서 활용 가능합니다.

**주요 타겟**: NXP i.MX8M Plus 임베디드 보드
**개발/테스트**: PC 환경에서 mock/video 입력 지원

---

## ✨ 주요 특징

- 🎯 **AI 기반 화염 감지**: YOLOv8 TFLite 모델 (NPU 가속 지원)
- 🌡️ **열화상 분석**: PureThermal IR 카메라로 온도 이상 감지
- 🔗 **센서 융합**: EO-IR 융합으로 오검지 최소화 (Phase 1 완료)
- 🖥️ **다중 인터페이스**: CLI, PyQt6 GUI, TCP 네트워크 전송
- 📹 **캡처/재생**: RGB/IR 비디오, RAW16 데이터, 메타데이터 저장
- ⚡ **NPU 가속**: i.MX8M Plus의 Vivante VIP8000 NPU 활용

---

## 🚀 빠른 시작

### PC 개발/테스트
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CONFIG_PATH=configs/config_pc.yaml python3 app.py
```

### 임베디드 보드 (i.MX8M Plus)
```bash
pip install -r requirements.txt  # 보드에 맞게 opencv/tflite 항목은 조정
CONFIG_PATH=configs/config.yaml python3 app.py
```

### 실행 모드

**GUI 모드**:
```bash
CONFIG_PATH=configs/config_pc.yaml APP_MODE=gui python3 app.py
```

**CLI + Display 모드**:
```bash
CONFIG_PATH=configs/config_pc.yaml python3 app.py
# config에서 DISPLAY.ENABLED: true로 설정
```

**비디오 재생 테스트**:
```bash
RGB_INPUT_MODE=video RGB_VIDEO_PATH=/path/video.mp4 RGB_LOOP=true IR_INPUT_MODE=mock python3 app.py
```

---

## 🎛️ 설정

### 필수 환경변수
- `CONFIG_PATH`: 설정 파일 경로 (기본: `configs/config.yaml`)
- `APP_MODE`: 실행 모드 (`cli` | `gui`)

### 입력 오버라이드
- `RGB_INPUT_MODE`: `live` | `video` | `mock`
- `RGB_VIDEO_PATH`: 비디오 파일 경로
- `IR_INPUT_MODE`: `live` | `video` | `mock`
- `RGB_DEVICE`, `IR_DEVICE`: 장치 경로 또는 인덱스
- `RGB_LOOP`, `IR_LOOP`: 비디오 루프 재생 (`true` | `false`)

### 모델/추론 설정
- `MODEL`: TFLite 모델 경로
- `LABEL`: 라벨 파일 경로
- `DELEGATE`: NPU delegate 라이브러리 (예: `/usr/lib/libvx_delegate.so`)
- `FUSION_VIS_MODE`: 시각화 모드 (`test` | `temp`)

### 해상도 제약
⚠️ **중요**: RGB 해상도는 **너비 16배수, 높이 8배수**여야 합니다.
- ✅ 올바른 예: 640×480, 1280×720
- ❌ 잘못된 예: 960×540

---

## 📹 캡처 & 재생

### 캡처
```bash
python3 capture.py --output ./capture_session [--duration SEC] [--max-frames N] [--save-det]
```

**저장 파일**:
- `rgb.mp4`: RGB 비디오
- `ir_vis.mp4`: IR 가시화 비디오
- `ir16/*.npy`: RAW16 열화상 데이터
- `metadata.csv`: 프레임 메타데이터
- `det.jsonl`: 검출 결과 (옵션)

### 재생
```python
from utils.capture_loader import CaptureLoader

for item in CaptureLoader("./capture_session"):
    rgb_frame = item["rgb"]
    ir_frame = item["ir"]
    ir_raw = item["ir_raw"]
    # 프레임 처리...
```

---

## 🧪 테스트

```bash
pip install -r requirements-dev.txt
pytest
```

**참고**: `sample/fire_sample.mp4`가 없으면 `tests/test_video_sources.py` 일부가 skip됩니다.
`test_fire_fusion.py`는 항상 실행 가능합니다.

---

## 🔧 문제 해결

| 문제 | 해결 방법 |
|------|----------|
| **Delegate 로드 실패** | 경로 확인 후 CPU/XNNPACK으로 자동 폴백 |
| **GStreamer 오류** | 해상도 제약 확인, `gst-inspect-1.0 v4l2src` |
| **IR/RGB 동기화 문제** | `SYNC.ENABLED`, `SYNC.MAX_DIFF_MS` 조정 |

---

## 📚 문서

> 전체 문서 목록은 [**docs/**](docs/) 참조

- **화재 융합 로드맵**: [`docs/FIRE_FUSION_ROADMAP.md`](docs/FIRE_FUSION_ROADMAP.md)
- **GUI 설계**: [`docs/pyqt_gui_design.md`](docs/pyqt_gui_design.md)
- **코드 리팩토링 계획**: [`docs/REFACTORING_ROADMAP.md`](docs/REFACTORING_ROADMAP.md)

---

## 📁 프로젝트 구조

```
pyrovision/
├── app.py                  # 메인 엔트리포인트
├── capture.py              # 데이터 캡처 스크립트
├── receiver.py             # TCP 수신 서버
├── sender.py               # TCP 송신 클라이언트
├── display.py              # CLI 디스플레이
├── camera/                 # 카메라 소스 (RGB/IR/PureThermal)
├── core/                   # 핵심 로직 (융합, 버퍼, 좌표 매핑, 상태)
├── detector/               # TFLite 추론 워커
├── gui/                    # PyQt6 GUI 애플리케이션
├── configs/                # 설정 파일 및 스키마
├── utils/                  # 유틸리티 (캡처 로더 등)
├── tests/                  # 단위/통합 테스트
├── model/                  # TFLite 모델 및 라벨 (대용량)
├── sample/                 # 샘플 영상/이미지
└── docs/                   # 문서 (로드맵, 설계)
```

---

## 🛠️ 기술 스택

- **Language**: Python 3.x
- **AI/ML**: YOLOv8, TFLite Runtime, NPU Delegate (Vivante VIP8000)
- **Computer Vision**: OpenCV, GStreamer
- **GUI**: PyQt6
- **Hardware**: NXP i.MX8M Plus, PureThermal IR Camera
- **Protocols**: V4L2, TCP/IP

---

## 🤝 기여

프로젝트 개선을 위한 기여를 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🔗 링크

- **GitHub**: [Stellar-Moment/PyroVision](https://github.com/Stellar-Moment/PyroVision)
- **Issues**: [Report a Bug](https://github.com/Stellar-Moment/PyroVision/issues)

---

<div align="center">

**Made with 🔥 by PyroVision Team**

</div>
