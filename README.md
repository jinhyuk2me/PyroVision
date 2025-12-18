<div align="center">
  <img src="asset/pyro_banner.png" width="100%">
</div>

## 1. 실행 예시 (Demo)
<div align="center">

| Industrial | Indoor | Outdoor |
| :---: | :---: | :---: |
| <img src="asset/final_industrial_test.gif" height="500"> | <img src="asset/indoor_test.gif" height="500"> | <img src="asset/outdoor_test.gif" height="500"> |

</div>

## 2. 프로젝트 소개 (Project Overview)
**PyroVision**은 NXP i.MX8M Plus 임베디드 보드의 NPU와 멀티미디어 성능을 활용하여 화재를 조기에 감지하고 모니터링하는 시스템입니다.<br>
**RGB-IR Sensor Fusion** 기술을 적용하여 기존 단일 센서 방식의 한계를 극복했습니다.
- **상호 보완 (Robustness)**: RGB로 연기/불꽃을 인식(YOLOv8)하고, IR로 실제 발열을 확인하여 오탐지를 최소화합니다.
- **실시간 데이터 전송**: 탐지된 데이터와 융합 영상은 TCP/IP를 통해 관제 시스템으로 전송됩니다.
- **직관적 제어**: GUI/CLI를 통해 카메라를 제어하고 상황을 실시간으로 모니터링할 수 있습니다.

## 3. 기술 스택 (Tech Stack)
| 분류 | 기술 |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Hardware** | ![NXP](https://img.shields.io/badge/NXP-i.MX8M_Plus-blue?style=flat-square) |
| **Framework** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white) ![GStreamer](https://img.shields.io/badge/GStreamer-E04E39?style=flat-square) |
| **AI/ML** | ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square) ![TensorFlow Lite](https://img.shields.io/badge/TFLite-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |

## 4. 리포지토리 구조

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
