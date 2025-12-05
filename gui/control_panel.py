import glob
import logging
import os
from typing import Optional, List, Dict, Any, Callable

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QGridLayout,
    QTabWidget,
    QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal, Qt


logger = logging.getLogger(__name__)


def _list_video_devices() -> List[str]:
    devices = []
    for path in glob.glob("/dev/video*"):
        if os.access(path, os.R_OK):
            devices.append(path)
    def _key(p: str) -> Any:
        try:
            return int("".join(ch for ch in p if ch.isdigit()))
        except Exception:
            return p
    return sorted(devices, key=_key)


class ControlPanel(QWidget):
    """카메라/탐지/좌표/캡처 제어 패널"""

    inputChanged = pyqtSignal(dict, dict)          # rgb_cfg, ir_cfg
    senderToggled = pyqtSignal(bool)
    displayToggled = pyqtSignal(bool)
    rotateIr = pyqtSignal()
    rotateRgb = pyqtSignal()
    irParamChanged = pyqtSignal(dict)
    detectorParamChanged = pyqtSignal(dict)
    coordChanged = pyqtSignal(dict)
    labelScaleChanged = pyqtSignal(float, bool)    # delta, reset
    captureToggled = pyqtSignal(bool, dict)        # start?, cfg
    visModeChanged = pyqtSignal(str)

    def __init__(
        self,
        controller: Optional[Any] = None,
        camera_state: Optional[Any] = None,
        log_fn: Optional[Callable[[str], None]] = None,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.camera_state = camera_state
        self.log_fn = log_fn

        self._build_ui()
        self._sync_from_controller()

    # === UI 빌드 ===
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._build_input_tab(), "Input")
        tabs.addTab(self._build_infer_tab(), "Inference")
        tabs.addTab(self._build_ir_tab(), "IR Hotspot")
        tabs.addTab(self._build_overlay_tab(), "Overlay")
        tabs.addTab(self._build_capture_tab(), "Capture")

        layout.addWidget(tabs)
        layout.addLayout(self._build_actions())
        layout.addStretch()

    def _build_input_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.rgb_mode_combo = QComboBox()
        self.rgb_mode_combo.addItems(["live", "video", "mock"])
        self.rgb_device_combo = QComboBox()
        self.rgb_device_combo.setEditable(True)
        self.rgb_device_combo.addItems([""] + _list_video_devices())
        self.rgb_path_edit = QLineEdit()
        self.rgb_loop_chk = QCheckBox("Loop")
        self.rgb_loop_chk.setChecked(True)

        self.ir_mode_combo = QComboBox()
        self.ir_mode_combo.addItems(["live", "video", "mock"])
        self.ir_device_combo = QComboBox()
        self.ir_device_combo.setEditable(True)
        self.ir_device_combo.addItems([""] + _list_video_devices())
        self.ir_path_edit = QLineEdit()
        self.ir_loop_chk = QCheckBox("Loop")
        self.ir_loop_chk.setChecked(True)

        form = QGridLayout()
        form.setAlignment(Qt.AlignmentFlag.AlignTop)
        form.addWidget(QLabel("RGB Mode"), 0, 0)
        form.addWidget(self.rgb_mode_combo, 0, 1)
        form.addWidget(QLabel("RGB Device"), 1, 0)
        form.addWidget(self.rgb_device_combo, 1, 1)
        form.addWidget(QLabel("RGB Path(s)"), 2, 0)
        form.addWidget(self.rgb_path_edit, 2, 1)
        form.addWidget(self.rgb_loop_chk, 3, 1)

        form.addWidget(QLabel("IR Mode"), 4, 0)
        form.addWidget(self.ir_mode_combo, 4, 1)
        form.addWidget(QLabel("IR Device"), 5, 0)
        form.addWidget(self.ir_device_combo, 5, 1)
        form.addWidget(QLabel("IR Path(s)"), 6, 0)
        form.addWidget(self.ir_path_edit, 6, 1)
        form.addWidget(self.ir_loop_chk, 7, 1)

        self.dev_refresh_btn = QPushButton("Refresh Devices")
        self.dev_refresh_btn.clicked.connect(self.refresh_devices)
        self.apply_input_btn = QPushButton("Apply Input")
        self.apply_input_btn.clicked.connect(self.on_apply_inputs)
        form.addWidget(self.dev_refresh_btn, 8, 0)
        form.addWidget(self.apply_input_btn, 8, 1)

        layout.addLayout(form)
        layout.addStretch()
        return widget

    def _build_infer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QGridLayout()
        form.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_edit = QLineEdit()
        self.label_edit = QLineEdit()
        self.delegate_edit = QLineEdit()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.15)
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["test", "prod", "off"])
        self.vis_mode_combo.currentTextChanged.connect(self._on_vis_mode_change)
        self.cls_smoke_chk = QCheckBox("Smoke")
        self.cls_fire_chk = QCheckBox("Fire")
        self.cls_fire_chk.setChecked(True)
        self.apply_infer_btn = QPushButton("Apply Detector")
        self.apply_infer_btn.clicked.connect(self.on_apply_infer)

        form.addWidget(QLabel("Model"), 0, 0)
        form.addWidget(self.model_edit, 0, 1)
        form.addWidget(QLabel("Label"), 1, 0)
        form.addWidget(self.label_edit, 1, 1)
        form.addWidget(QLabel("Delegate"), 2, 0)
        form.addWidget(self.delegate_edit, 2, 1)
        form.addWidget(QLabel("Conf Thr"), 3, 0)
        form.addWidget(self.conf_spin, 3, 1)
        form.addWidget(QLabel("Fusion Vis Mode"), 4, 0)
        form.addWidget(self.vis_mode_combo, 4, 1)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Classes"))
        class_row.addWidget(self.cls_smoke_chk)
        class_row.addWidget(self.cls_fire_chk)
        class_row.addStretch()

        layout.addLayout(form)
        layout.addLayout(class_row)
        layout.addWidget(self.apply_infer_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        return widget

    def _build_ir_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ir_fire_chk = QCheckBox("Fire Detection")
        self.ir_fire_chk.setChecked(True)
        self.ir_fire_min = QDoubleSpinBox()
        self.ir_fire_min.setRange(-50.0, 400.0)
        self.ir_fire_min.setDecimals(1)
        self.ir_fire_thr = QDoubleSpinBox()
        self.ir_fire_thr.setRange(0.0, 200.0)
        self.ir_fire_thr.setDecimals(1)
        self.ir_fire_raw = QDoubleSpinBox()
        self.ir_fire_raw.setRange(0.0, 200.0)
        self.ir_fire_raw.setDecimals(1)
        self.ir_tau = QDoubleSpinBox()
        self.ir_tau.setRange(0.1, 1.0)
        self.ir_tau.setDecimals(3)
        self.ir_tau.setSingleStep(0.01)
        self.apply_ir_btn = QPushButton("Apply IR Hotspot")
        self.apply_ir_btn.clicked.connect(self.on_apply_ir)

        layout.addWidget(self.ir_fire_chk, 0, 0, 1, 2)
        layout.addWidget(QLabel("MinTemp(C)"), 1, 0)
        layout.addWidget(self.ir_fire_min, 1, 1)
        layout.addWidget(QLabel("Thr(C)"), 2, 0)
        layout.addWidget(self.ir_fire_thr, 2, 1)
        layout.addWidget(QLabel("RawThr(C)"), 3, 0)
        layout.addWidget(self.ir_fire_raw, 3, 1)
        layout.addWidget(QLabel("Tau"), 4, 0)
        layout.addWidget(self.ir_tau, 4, 1)
        layout.addWidget(self.apply_ir_btn, 5, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        return widget

    def _build_overlay_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.offset_x_spin = QDoubleSpinBox()
        self.offset_x_spin.setRange(-4000.0, 4000.0)
        self.offset_x_spin.setDecimals(2)
        self.offset_y_spin = QDoubleSpinBox()
        self.offset_y_spin.setRange(-4000.0, 4000.0)
        self.offset_y_spin.setDecimals(2)
        self.offset_step_spin = QDoubleSpinBox()
        self.offset_step_spin.setRange(0.1, 100.0)
        self.offset_step_spin.setValue(5.0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 20.0)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSingleStep(0.05)
        self.scale_step_spin = QDoubleSpinBox()
        self.scale_step_spin.setRange(0.001, 1.0)
        self.scale_step_spin.setValue(0.05)
        self.apply_coord_btn = QPushButton("Apply Coord")
        self.apply_coord_btn.clicked.connect(self.on_apply_coord)

        layout.addWidget(QLabel("Offset X"), 0, 0)
        layout.addWidget(self.offset_x_spin, 0, 1)
        layout.addWidget(QLabel("Offset Y"), 1, 0)
        layout.addWidget(self.offset_y_spin, 1, 1)
        layout.addWidget(QLabel("Offset Step"), 2, 0)
        layout.addWidget(self.offset_step_spin, 2, 1)

        nudge = QHBoxLayout()
        for text, dx, dy in [("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)]:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, dx=dx, dy=dy: self.nudge_offset(dx, dy))
            nudge.addWidget(btn)
        layout.addLayout(nudge, 3, 0, 1, 2)

        layout.addWidget(QLabel("Scale"), 4, 0)
        layout.addWidget(self.scale_spin, 4, 1)
        layout.addWidget(QLabel("Scale Step"), 5, 0)
        layout.addWidget(self.scale_step_spin, 5, 1)

        scale_row = QHBoxLayout()
        btn_minus = QPushButton("Scale -")
        btn_minus.clicked.connect(lambda: self.nudge_scale(-self.scale_step_spin.value()))
        btn_plus = QPushButton("Scale +")
        btn_plus.clicked.connect(lambda: self.nudge_scale(self.scale_step_spin.value()))
        scale_row.addWidget(btn_minus)
        scale_row.addWidget(btn_plus)
        layout.addLayout(scale_row, 6, 0, 1, 2)

        self.apply_coord_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.apply_coord_btn, 7, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

        label_row = QHBoxLayout()
        self.label_scale_step = QDoubleSpinBox()
        self.label_scale_step.setRange(0.05, 1.0)
        self.label_scale_step.setSingleStep(0.05)
        self.label_scale_step.setValue(0.1)
        label_row.addWidget(QLabel("Label Scale Step"))
        label_row.addWidget(self.label_scale_step)
        reset_btn = QPushButton("Reset Label Scale")
        reset_btn.clicked.connect(lambda: self.labelScaleChanged.emit(0.0, True))
        down_btn = QPushButton("Label Scale -")
        down_btn.clicked.connect(lambda: self.labelScaleChanged.emit(-self.label_scale_step.value(), False))
        up_btn = QPushButton("Label Scale +")
        up_btn.clicked.connect(lambda: self.labelScaleChanged.emit(self.label_scale_step.value(), False))
        label_row.addWidget(down_btn)
        label_row.addWidget(up_btn)
        label_row.addWidget(reset_btn)
        layout.addLayout(label_row, 8, 0, 1, 2)
        return widget

    def _build_capture_tab(self):
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.capture_output_edit = QLineEdit("./capture_session")
        self.capture_duration_spin = QDoubleSpinBox()
        self.capture_duration_spin.setSuffix(" s")
        self.capture_duration_spin.setRange(0, 10000)
        self.capture_max_spin = QSpinBox()
        self.capture_max_spin.setRange(0, 100000)
        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.setCheckable(True)
        self.capture_btn.toggled.connect(self.on_capture_toggle)

        layout.addWidget(QLabel("Output Dir"), 0, 0)
        layout.addWidget(self.capture_output_edit, 0, 1)
        layout.addWidget(QLabel("Duration"), 1, 0)
        layout.addWidget(self.capture_duration_spin, 1, 1)
        layout.addWidget(QLabel("Max Frames"), 2, 0)
        layout.addWidget(self.capture_max_spin, 2, 1)
        layout.addWidget(self.capture_btn, 3, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
        return widget

    def _build_actions(self):
        layout = QHBoxLayout()
        self.sender_btn = QPushButton("Start Sender")
        self.sender_btn.setCheckable(True)
        self.sender_btn.toggled.connect(self.senderToggled)

        self.display_btn = QPushButton("Start Display")
        self.display_btn.setCheckable(True)
        self.display_btn.toggled.connect(self.displayToggled)

        rotate_ir_btn = QPushButton("Rotate IR 90°")
        rotate_ir_btn.clicked.connect(self.rotateIr)
        rotate_rgb_btn = QPushButton("Rotate RGB 90°")
        rotate_rgb_btn.clicked.connect(self.rotateRgb)

        layout.addWidget(self.sender_btn)
        layout.addWidget(self.display_btn)
        layout.addWidget(rotate_ir_btn)
        layout.addWidget(rotate_rgb_btn)
        layout.addStretch()
        return layout

    # === 이벤트 핸들러 ===
    def refresh_devices(self):
        current_rgb = self.rgb_device_combo.currentText().strip()
        current_ir = self.ir_device_combo.currentText().strip()
        devices = [""] + _list_video_devices()
        self.rgb_device_combo.clear()
        self.rgb_device_combo.addItems(devices)
        self.rgb_device_combo.setCurrentText(current_rgb)
        self.ir_device_combo.clear()
        self.ir_device_combo.addItems(devices)
        self.ir_device_combo.setCurrentText(current_ir)
        if self.log_fn:
            self.log_fn("Device list refreshed")

    def _parse_paths(self, text):
        parts = [p.strip() for p in text.split(';') if p.strip()]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return parts

    def on_apply_inputs(self) -> None:
        rgb_cfg = {
            'MODE': self.rgb_mode_combo.currentText().lower(),
            'LOOP': self.rgb_loop_chk.isChecked(),
            'DEVICE': self.rgb_device_combo.currentText().strip(),
        }
        ir_cfg = {
            'MODE': self.ir_mode_combo.currentText().lower(),
            'LOOP': self.ir_loop_chk.isChecked(),
            'DEVICE': self.ir_device_combo.currentText().strip(),
        }
        rgb_paths = self._parse_paths(self.rgb_path_edit.text())
        ir_paths = self._parse_paths(self.ir_path_edit.text())
        if rgb_cfg['MODE'] == 'video' and not rgb_paths:
            if self.log_fn:
                self.log_fn("RGB video mode requires path(s)")
            return
        if ir_cfg['MODE'] == 'video' and not ir_paths:
            if self.log_fn:
                self.log_fn("IR video mode requires path(s)")
            return
        if rgb_paths:
            rgb_cfg['VIDEO_PATH'] = rgb_paths
        if ir_paths:
            ir_cfg['VIDEO_PATH'] = ir_paths
        if rgb_cfg['MODE'] != 'live':
            rgb_cfg['DEVICE'] = ""
        if ir_cfg['MODE'] != 'live':
            ir_cfg['DEVICE'] = ""
        self.inputChanged.emit(rgb_cfg, ir_cfg)

    def on_apply_infer(self) -> None:
        allowed = []
        if self.cls_smoke_chk.isChecked():
            allowed.append(0)
        if self.cls_fire_chk.isChecked():
            allowed.append(1)
        if not allowed:
            allowed = None
        params = {
            'MODEL': self.model_edit.text().strip(),
            'LABEL': self.label_edit.text().strip(),
            'DELEGATE': self.delegate_edit.text().strip(),
            'ALLOWED_CLASSES': allowed,
            'CONF_THR': self.conf_spin.value(),
            'VIS_MODE': self.vis_mode_combo.currentText().lower(),
        }
        self.detectorParamChanged.emit(params)

    def on_apply_ir(self) -> None:
        params = {
            'FIRE_DETECTION': self.ir_fire_chk.isChecked(),
            'FIRE_MIN_TEMP': self.ir_fire_min.value(),
            'FIRE_THR': self.ir_fire_thr.value(),
            'FIRE_RAW_THR': self.ir_fire_raw.value(),
            'TAU': self.ir_tau.value(),
        }
        self.irParamChanged.emit(params)

    def on_apply_coord(self) -> None:
        params = {
            'offset_x': self.offset_x_spin.value(),
            'offset_y': self.offset_y_spin.value(),
            'scale': self.scale_spin.value(),
        }
        self.coordChanged.emit(params)

    def nudge_offset(self, dx: float, dy: float) -> None:
        step = self.offset_step_spin.value()
        self.offset_x_spin.setValue(self.offset_x_spin.value() + dx * step)
        self.offset_y_spin.setValue(self.offset_y_spin.value() + dy * step)
        self.on_apply_coord()

    def nudge_scale(self, delta: float) -> None:
        new_scale = max(0.1, self.scale_spin.value() + delta)
        self.scale_spin.setValue(new_scale)
        self.on_apply_coord()

    def on_capture_toggle(self, checked: bool) -> None:
        cfg = {
            'OUTPUT_DIR': self.capture_output_edit.text().strip(),
            'DURATION_SEC': self.capture_duration_spin.value(),
            'MAX_FRAMES': self.capture_max_spin.value(),
        }
        self.capture_btn.setText("Stop Capture" if checked else "Start Capture")
        self.captureToggled.emit(checked, cfg)

    def _on_vis_mode_change(self, text):
        self.visModeChanged.emit((text or "test").lower())

    # === 외부 동기화 ===
    def _sync_from_controller(self):
        if not self.controller:
            return
        rgb_cfg, ir_cfg = self.controller.get_input_cfg()
        self.rgb_mode_combo.setCurrentText(str(rgb_cfg.get('MODE', 'live')).lower())
        self.rgb_device_combo.setCurrentText(str(rgb_cfg.get('DEVICE', "")))
        self.rgb_loop_chk.setChecked(bool(rgb_cfg.get('LOOP', True)))
        if rgb_cfg.get('VIDEO_PATH'):
            self.rgb_path_edit.setText(";".join(rgb_cfg['VIDEO_PATH']) if isinstance(rgb_cfg['VIDEO_PATH'], (list, tuple)) else str(rgb_cfg['VIDEO_PATH']))

        self.ir_mode_combo.setCurrentText(str(ir_cfg.get('MODE', 'live')).lower())
        self.ir_device_combo.setCurrentText(str(ir_cfg.get('DEVICE', "")))
        self.ir_loop_chk.setChecked(bool(ir_cfg.get('LOOP', True)))
        if ir_cfg.get('VIDEO_PATH'):
            self.ir_path_edit.setText(";".join(ir_cfg['VIDEO_PATH']) if isinstance(ir_cfg['VIDEO_PATH'], (list, tuple)) else str(ir_cfg['VIDEO_PATH']))

        det_cfg = self.controller.get_detector_cfg()
        self.model_edit.setText(det_cfg.get('MODEL', ""))
        self.label_edit.setText(det_cfg.get('LABEL', ""))
        self.delegate_edit.setText(det_cfg.get('DELEGATE', ""))
        self.conf_spin.setValue(float(det_cfg.get('CONF_THR', det_cfg.get('CONF_THRESHOLD', 0.15))))
        vis_mode = os.getenv("FUSION_VIS_MODE", det_cfg.get('VIS_MODE', "test")).lower()
        self.vis_mode_combo.setCurrentText(vis_mode)
        allowed = det_cfg.get('ALLOWED_CLASSES') or []
        self.cls_smoke_chk.setChecked(0 in allowed if allowed else False)
        self.cls_fire_chk.setChecked((not allowed) or (1 in allowed))

        ir_cfg_full = getattr(self.controller, "ir_cfg", {})
        self.ir_fire_chk.setChecked(bool(ir_cfg_full.get('FIRE_DETECTION', True)))
        self.ir_fire_min.setValue(ir_cfg_full.get('FIRE_MIN_TEMP', 80))
        self.ir_fire_thr.setValue(ir_cfg_full.get('FIRE_THR', 20))
        self.ir_fire_raw.setValue(ir_cfg_full.get('FIRE_RAW_THR', 5))
        self.ir_tau.setValue(ir_cfg_full.get('TAU', 0.95))

        coord = self.controller.get_coord_cfg()
        self.set_coord_params(coord)

        capture_cfg = self.controller.get_capture_cfg()
        self.capture_output_edit.setText(capture_cfg.get('OUTPUT_DIR', "./capture_session"))
        self.capture_duration_spin.setValue(capture_cfg.get('DURATION_SEC') or 0)
        self.capture_max_spin.setValue(capture_cfg.get('MAX_FRAMES') or 0)

    def set_coord_params(self, params: dict) -> None:
        if not params:
            return
        self.offset_x_spin.blockSignals(True)
        self.offset_y_spin.blockSignals(True)
        self.scale_spin.blockSignals(True)
        try:
            if 'offset_x' in params:
                self.offset_x_spin.setValue(params.get('offset_x', 0.0))
            if 'offset_y' in params:
                self.offset_y_spin.setValue(params.get('offset_y', 0.0))
            if params.get('scale') is not None:
                self.scale_spin.setValue(params.get('scale', 1.0))
        finally:
            self.offset_x_spin.blockSignals(False)
            self.offset_y_spin.blockSignals(False)
            self.scale_spin.blockSignals(False)

    def set_sender_state(self, running: bool) -> None:
        self.sender_btn.blockSignals(True)
        self.sender_btn.setChecked(running)
        self.sender_btn.setText("Stop Sender" if running else "Start Sender")
        self.sender_btn.blockSignals(False)

    def set_display_state(self, running: bool) -> None:
        self.display_btn.blockSignals(True)
        self.display_btn.setChecked(running)
        self.display_btn.setText("Stop Display" if running else "Start Display")
        self.display_btn.blockSignals(False)

    def set_capture_state(self, running: bool) -> None:
        self.capture_btn.blockSignals(True)
        self.capture_btn.setChecked(running)
        self.capture_btn.setText("Stop Capture" if running else "Start Capture")
        self.capture_btn.blockSignals(False)
