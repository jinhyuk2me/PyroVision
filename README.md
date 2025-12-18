<div align="center">
  <img src="asset/pyro_banner.png" width="100%">
</div>

## 1. Demo
<div align="center">

| Industrial | Indoor | Outdoor |
| :---: | :---: | :---: |
| <img src="asset/final_industrial_test.gif" height="500"> | <img src="asset/indoor_test.gif" height="500"> | <img src="asset/outdoor_test.gif" height="500"> |

</div>

## 2. Project Overview
**PyroVision** is an early fire detection and monitoring system that leverages the NPU and multimedia capabilities of the NXP i.MX8M Plus embedded board.<br>
It applies **RGB-IR Sensor Fusion** technology to overcome the limitations of traditional single-sensor approaches.
- **Robustness**: RGB detects smoke/flames (YOLOv8) while IR verifies actual heat, minimizing false positives.
- **Real-time Data Transmission**: Detected data and fused video are transmitted to the control system via TCP/IP.
- **Intuitive Control**: Monitor and control cameras in real-time through GUI/CLI.

## 3. Tech Stack
| Category | Technology |
| :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Hardware** | ![NXP](https://img.shields.io/badge/NXP-i.MX8M_Plus-blue?style=flat-square) |
| **Framework** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) ![PyQt6](https://img.shields.io/badge/PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white) ![GStreamer](https://img.shields.io/badge/GStreamer-E04E39?style=flat-square) |
| **AI/ML** | ![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square) ![TensorFlow Lite](https://img.shields.io/badge/TFLite-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) |

## 4. Repository Structure

```
pyro_vision/
├── app.py                  # Main entry point
├── capture.py              # Capture script
├── receiver.py             # TCP receiving server
├── sender.py               # TCP transmission module
├── camera/                 # Camera sources
│   ├── rgbcam.py          # RGB camera
│   ├── ircam.py           # IR camera interface
│   ├── purethermal/       # PureThermal driver
│   ├── frame_source.py    # Frame source base class
│   └── device_selector.py # Automatic device selection
├── detector/
│   └── tflite.py          # YOLOv8 TFLite worker
├── gui/
│   └── app_gui.py         # PyQt6 GUI
├── core/
│   ├── state.py           # Camera state management
│   └── util.py            # Utility functions
├── configs/
│   ├── config.yaml        # Board configuration
│   ├── config_pc.yaml     # PC configuration
│   ├── schema.py          # Configuration schema
│   └── get_cfg.py         # Configuration loader
├── utils/
│   └── capture_loader.py  # Capture playback loader
├── tests/                 # Tests
└── model/                 # TFLite models and labels
```
