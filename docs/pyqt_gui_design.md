% PYQT GUI 설계서

# PyQt GUI 설계서

## 1. 목표
- CLI 기반 제어/모니터링을 PyQt GUI로 확장하여 현장 운용/디버깅 효율을 높인다.
- HDMI 출력/버퍼 구조를 그대로 활용하되, PyQt 창에서 상태 확인과 제어를 한 번에 수행한다.
- GUI 모드는 선택 옵션으로 제공하고, CLI 모드와 동일한 파이프라인을 공유한다.

## 2. 주요 화면 구성안

### 2.1 메인 창 레이아웃
```
┌─────────────────────────────────────────────────────────┐
│ [상단 상태바]                                           │
│  - 시스템 상태: IR 준비됨/미준비, RGB 준비됨/미준비    │
│  - FPS/지연: IR FPS, RGB FPS, 추론 FPS, 송신 FPS        │
│  - SYNC 상태: 허용시간차 내/초과                         │
├─────────────────────────────────────────────────────────┤
│ [좌측 영상 패널]              │ [우측 영상 패널]         │
│  - RGB 검출 (YOLO 결과)       │  - IR (컬러맵 or RAW)     │
│  - 버튼으로 표시 모드 전환    │  - Hotspot 정보 오버레이  │
├─────────────────────────────────────────────────────────┤
│ [하단 제어 패널]                                        │
│  1) 입력 소스/모드 선택: live/video/mock, 경로 지정      │
│  2) 카메라 조작: 회전/플립 버튼, 캡처 시작/중지         │
│  3) 송신/디스플레이 토글: Sender ON/OFF, Display ON/OFF │
│  4) 로그 뷰어 탭: 최근 경고/오류 표시                  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 패널 설명
1. **상단 상태바**
   - QLabel/QFrame로 IR/RGB 준비 상태, FPS, SYNC 상태, 현재 모드 등을 표시
   - 준비 상태는 `FrameSource`에서 emit하는 시그널 또는 첫 프레임 수신 이벤트 기반
2. **영상 패널**
   - `QLabel` + `QPixmap` 형태로 표시하거나 `QOpenGLWidget` 이용 (성능 고려)
   - RGB/IR 각각 독립 갱신, overlay는 OpenCV → QImage 변환 후 draw
   - FPS가 낮아질 경우 경고 표시
3. **제어 패널**
   - 입력 모드: ComboBox + 경로 입력 (line edit) + 파일 선택 버튼
   - 회전/플립: `camera_state`와 연동하는 push button
   - 캡처 제어: `capture.py`를 서브 프로세스로 실행하는 버튼과 진행 상태 표시
   - 송신/디스플레이 토글: `sender`/`display_loop` 스레드의 활성/비활성 제어
   - 로그 탭: QPlainTextEdit에 주요 로그(경고/오류)를 append

## 3. 아키텍처

### 3.1 프로세스 흐름
- GUI 모드에서는 `QApplication`을 생성하고, `AppController`가 기존 파이프라인(카메라/검출/송신)을 관리
- 각 백그라운드 스레드는 기존 `run()` 로직을 그대로 사용하고, GUI와는 `Queue` or `Signal`로 상태만 공유
- `DoubleBuffer`에서 읽은 프레임을 GUI가 주기적으로 pull (`QTimer` 30~60ms)

### 3.2 주요 클래스
- `AppController`: 파이프라인 초기화, 설정 로딩, GUI와의 인터페이스
- `MainWindow(QMainWindow)`: 앞서 설명한 패널들로 구성, controller에 명령 전달
- `StatusModel`: IR/RGB/송신/추론 상태를 보관, GUI에서 바인딩
- `LogHandler(logging.Handler)`: Python logging을 Qt signal로 전달

### 3.3 이벤트 처리
- 키보드 단축키는 GUI에 통합 (예: Ctrl+1 → IR 90° 회전)
- 파일 선택 → 설정 업데이트 → controller가 FrameSource 재시작
- 송신/디스플레이 버튼 → 해당 스레드 토글 (start/stop)
- 캡처 버튼 → `capture.py` 실행 (subprocess) 후 진행 상황 모니터링

## 4. PyQt 구성 요소
| 기능 | 위젯/모듈 | 비고 |
|------|-----------|------|
| 메인 윈도우 | QMainWindow + QWidget layout | DockWidget 또는 Splitter |
| 영상 표시 | QLabel + QPixmap (또는 QGraphicsView) | OpenCV → QImage 변환 |
| 상태 표시 | QLabel, QProgressBar | 색상으로 상태 표현 |
| 로그 표시 | QPlainTextEdit | logging handler 연결 |
| 입력 제어 | QComboBox, QLineEdit, QPushButton | 모드/경로 선택 |
| 캡처 제어 | QPushButton, QProgressBar | subprocess 관리 |
| 송신/디스플레이 | QPushButton (toggle) | 상태 LED 표시 |

## 5. UI 상태 관리
- `AppController`는 `@pyqtSlot`으로 GUI 액션을 받아, 기존 카메라 state 변경/설정 반영
- 상태 업데이트는 Qt `Signal` 사용  
  예: `frame_ready`(rgb_frame, ir_frame), `status_changed`(dict), `log_received`(str)

## 6. 구현 계획
1. **1단계**: PyQt GUI 기본 틀/상태바/로그 뷰 + RGB 미리보기
2. **2단계**: IR 표시 + 제어 버튼 연동(회전/플립)
3. **3단계**: 입력 모드 전환 UI + Sender/Display 토글
4. **4단계**: 캡처 제어/상태 표시
5. **5단계**: 테스트/정리 (mock 입력으로 CI 테스트)

## 7. 의존성/빌드
- PyQt6 (이미 설치 완료)  
- Qt Designer (선택) → 보드에서는 어려울 수 있으니 PC에서 디자인 후 `.ui` 배포  
- GUI 모드 옵션: `python3 app.py --mode gui` 혹은 `APP_MODE=gui`

## 8. 남은 숙제
- FrameSource 헬스체크 상태를 GUI에 전달하기 위한 공통 인터페이스 정리
- PyQt GUI와 HDMI 디스플레이 중복 사용 시 자원 충돌 검토
- 보드 성능에 따라 QOpenGLWidget 사용 여부 결정 (필요 시 fallback)
