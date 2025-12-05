import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Any

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent

from gui.control_panel import ControlPanel
from gui.monitor_panel import MonitorPanel
from gui.frame_updater import FrameUpdater
from gui.constants import (
    PREVIEW_MIN_WIDTH,
    PREVIEW_MIN_HEIGHT,
    TIMER_INTERVAL_MS,
    VIDEO_GRID_RGB_ROW,
    VIDEO_GRID_RGB_COL,
    VIDEO_GRID_DET_ROW,
    VIDEO_GRID_DET_COL,
    VIDEO_GRID_IR_ROW,
    VIDEO_GRID_IR_COL,
    VIDEO_GRID_OVERLAY_ROW,
    VIDEO_GRID_OVERLAY_COL,
)


logger = logging.getLogger(__name__)


def _preview_label(title: str) -> QLabel:
    lbl = QLabel(title)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("background: #111; color: #eee;")
    lbl.setMinimumSize(PREVIEW_MIN_WIDTH, PREVIEW_MIN_HEIGHT)
    lbl.setWordWrap(True)
    return lbl


class MainWindow(QMainWindow):
    def __init__(
        self,
        buffers: Dict[str, Any],
        camera_state: Any,
        controller: Optional[Any],
        log_handler: Optional[Any] = None
    ) -> None:
        super().__init__()
        self.buffers = buffers
        self.camera_state = camera_state
        self.controller = controller
        self.log_handler = log_handler
        self.capture_process = None
        self.fusion_vis_mode = os.getenv("FUSION_VIS_MODE", "test").lower()

        self.central = QWidget()
        self.setCentralWidget(self.central)
        root_layout = QVBoxLayout()
        self.central.setLayout(root_layout)

        # === Video area ===
        self.rgb_label = _preview_label("RGB Preview")
        self.det_label = _preview_label("Det Preview")
        self.ir_label = _preview_label("IR Preview")
        self.overlay_label = _preview_label("Overlay Preview")

        self.rgb_info = QLabel("-")
        self.det_info = QLabel("-")
        self.ir_info = QLabel("-")
        self.overlay_info = QLabel("-")

        grid = QGridLayout()
        grid.addWidget(self.rgb_label, VIDEO_GRID_RGB_ROW, VIDEO_GRID_RGB_COL)
        grid.addWidget(self.det_label, VIDEO_GRID_DET_ROW, VIDEO_GRID_DET_COL)
        grid.addWidget(self.ir_label, VIDEO_GRID_IR_ROW, VIDEO_GRID_IR_COL)
        grid.addWidget(self.overlay_label, VIDEO_GRID_OVERLAY_ROW, VIDEO_GRID_OVERLAY_COL)
        grid.addWidget(self.rgb_info, 2, 0)
        grid.addWidget(self.det_info, 2, 1)
        grid.addWidget(self.ir_info, 3, 0)
        grid.addWidget(self.overlay_info, 3, 1)
        video_widget = QWidget()
        video_widget.setLayout(grid)

        # === Control / Monitor panels ===
        self.control_panel = ControlPanel(controller=self.controller, camera_state=self.camera_state, log_fn=self.append_log)
        self.monitor_panel = MonitorPanel()

        # Compose splitters
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.addWidget(video_widget)
        h_split.addWidget(self.control_panel)
        h_split.setStretchFactor(0, 3)
        h_split.setStretchFactor(1, 1)

        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.addWidget(h_split)
        v_split.addWidget(self.monitor_panel)
        v_split.setStretchFactor(0, 4)
        v_split.setStretchFactor(1, 1)

        root_layout.addWidget(v_split)

        # FrameUpdater wiring
        labels: Dict[str, QLabel] = {
            'rgb_label': self.rgb_label,
            'det_label': self.det_label,
            'ir_label': self.ir_label,
            'overlay_label': self.overlay_label,
            'rgb_info': self.rgb_info,
            'det_info': self.det_info,
            'ir_info': self.ir_info,
            'overlay_info': self.overlay_info,
            'status_label': self.monitor_panel.status_label,
            'fusion_info': self.monitor_panel.fusion_info,
        }
        plots = self.monitor_panel.get_plots()
        self.frame_updater = FrameUpdater(
            buffers,
            controller,
            controller.cfg if controller else {},
            labels,
            plots,
            controller.get_sync_cfg() if controller else {},
        )
        self.frame_updater.set_fire_fusion(getattr(controller, 'fire_fusion', None))
        self.frame_updater.fusion_vis_mode = self.fusion_vis_mode
        self.frame_updater.set_coord_sync_handler(self._sync_coord_ui)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.frame_updater.update_frames)
        self.timer.start(TIMER_INTERVAL_MS)

        # Signal wiring
        self.control_panel.senderToggled.connect(self._on_sender_toggled)
        self.control_panel.displayToggled.connect(self._on_display_toggled)
        self.control_panel.rotateIr.connect(self._on_rotate_ir)
        self.control_panel.rotateRgb.connect(self._on_rotate_rgb)
        self.control_panel.inputChanged.connect(self._on_input_changed)
        self.control_panel.detectorParamChanged.connect(self._on_detector_params)
        self.control_panel.irParamChanged.connect(self._on_ir_params)
        self.control_panel.coordChanged.connect(self._on_coord_changed)
        self.control_panel.labelScaleChanged.connect(self._on_label_scale)
        self.control_panel.captureToggled.connect(self._on_capture_toggled)
        self.control_panel.visModeChanged.connect(self._on_vis_mode_changed)

        # 초기 버튼 상태 동기화
        if self.controller:
            self.control_panel.set_sender_state(self.controller.sender_running())
            self.control_panel.set_display_state(self.controller.display_running())

    def _on_sender_toggled(self, checked: bool) -> None:
        if not self.controller:
            return
        if checked:
            ok = self.controller.start_sender()
            if not ok:
                self.append_log("Failed to start sender")
                self.control_panel.set_sender_state(False)
                return
            self.append_log("Sender started")
        else:
            self.controller.stop_sender()
            self.append_log("Sender stopped")
        self.control_panel.set_sender_state(self.controller.sender_running())

    def _on_display_toggled(self, checked: bool) -> None:
        if not self.controller:
            return
        if checked:
            ok = self.controller.start_display()
            if not ok:
                self.append_log("Failed to start display")
                self.control_panel.set_display_state(False)
                return
            self.append_log("Display started")
        else:
            self.controller.stop_display()
            self.append_log("Display stopped")
        self.control_panel.set_display_state(self.controller.display_running())

    def _on_rotate_ir(self) -> None:
        if not self.camera_state:
            return
        angle = self.camera_state.rotate_ir_cw()
        self.append_log(f"IR rotation: {angle}°")

    def _on_rotate_rgb(self) -> None:
        if not self.camera_state:
            return
        angle = self.camera_state.rotate_rgb_cw()
        self.append_log(f"RGB rotation: {angle}°")

    def _on_input_changed(self, rgb_cfg: dict, ir_cfg: dict) -> None:
        if not self.controller:
            return
        merged_rgb = dict(getattr(self.controller, "rgb_input_cfg", {}))
        merged_ir = dict(getattr(self.controller, "ir_input_cfg", {}))
        merged_rgb.update(rgb_cfg or {})
        merged_ir.update(ir_cfg or {})
        try:
            self.controller.restart_sources(rgb_input_cfg=merged_rgb, ir_input_cfg=merged_ir)
            self.append_log(f"Input applied: RGB={merged_rgb.get('DEVICE','')} IR={merged_ir.get('DEVICE','')}")
        except Exception as exc:
            self.append_log(f"Failed to apply input: {exc}")
            logger.exception("Input apply failed: %s", exc)

    def _on_detector_params(self, params: dict) -> None:
        if not self.controller:
            return
        vis_mode = params.get('VIS_MODE', self.fusion_vis_mode)
        self._on_vis_mode_changed(vis_mode)
        allowed = params.get('ALLOWED_CLASSES')
        try:
            self.controller.update_detector_cfg(
                model_path=params.get('MODEL'),
                label_path=params.get('LABEL'),
                delegate=params.get('DELEGATE'),
                allowed_classes=allowed,
                conf_thr=params.get('CONF_THR'),
                use_npu=bool(params.get('DELEGATE')),
                restart=True,
            )
            self.append_log(f"RGB detector applied (vis_mode={self.fusion_vis_mode})")
        except Exception as exc:
            self.append_log(f"Failed to apply RGB detector: {exc}")
            logger.exception("Detector apply failed: %s", exc)

    def _on_ir_params(self, params: dict) -> None:
        if not self.controller:
            return
        try:
            self.controller.update_ir_fire_cfg(
                fire_enabled=params.get('FIRE_DETECTION'),
                min_temp=params.get('FIRE_MIN_TEMP'),
                thr=params.get('FIRE_THR'),
                raw_thr=params.get('FIRE_RAW_THR'),
                tau=params.get('TAU'),
            )
            self.append_log("IR hotspot config applied")
        except Exception as exc:
            self.append_log(f"Failed to apply IR hotspot: {exc}")
            logger.exception("IR hotspot apply failed: %s", exc)

    def _on_coord_changed(self, params: dict) -> None:
        if not self.controller:
            return
        self.controller.set_coord_cfg(params)
        self._sync_coord_ui(params)
        self.append_log(f"Coord updated: {params}")

    def _on_label_scale(self, delta: float, reset: bool) -> None:
        if not self.controller:
            return
        if reset:
            new_scale = self.controller.reset_label_scale()
        else:
            new_scale = self.controller.adjust_label_scale(delta)
        if new_scale is not None:
            self.append_log(f"Label scale → {new_scale:.2f}x")

    def _on_capture_toggled(self, start: bool, cfg: dict) -> None:
        if start:
            self._start_capture(cfg)
        else:
            self._stop_capture()

    def _start_capture(self, cfg: dict) -> None:
        capture_script = Path(__file__).resolve().parents[1] / "capture.py"
        if not capture_script.exists():
            self.append_log("capture.py not found")
            self.control_panel.set_capture_state(False)
            return
        try:
            exe = sys.executable or "python3"
            cmd = [exe, str(capture_script)]
            output_dir = cfg.get('OUTPUT_DIR')
            if output_dir:
                cmd += ["--output", output_dir]
            duration = cfg.get('DURATION_SEC') or 0
            if duration > 0:
                cmd += ["--duration", f"{duration}"]
            max_frames = cfg.get('MAX_FRAMES') or 0
            if max_frames > 0:
                cmd += ["--max-frames", str(max_frames)]
            self.capture_process = subprocess.Popen(cmd)
            self.append_log("Capture started")
        except Exception as exc:
            self.append_log(f"Capture start failed: {exc}")
            logger.exception("Capture start failed: %s", exc)
            self.control_panel.set_capture_state(False)

    def _stop_capture(self) -> None:
        if self.capture_process:
            try:
                self.capture_process.terminate()
            except Exception:
                pass
            self.capture_process = None
        self.append_log("Capture stopped")
        self.control_panel.set_capture_state(False)

    def _on_vis_mode_changed(self, mode: str) -> None:
        self.fusion_vis_mode = (mode or "test").lower()
        os.environ["FUSION_VIS_MODE"] = self.fusion_vis_mode
        if self.frame_updater:
            self.frame_updater.fusion_vis_mode = self.fusion_vis_mode

    def _sync_coord_ui(self, params: Optional[dict] = None) -> None:
        if not self.controller:
            return
        params = params or self.controller.get_coord_cfg()
        self.control_panel.set_coord_params(params)

    def append_log(self, text: str) -> None:
        if hasattr(self.monitor_panel, "append_log"):
            self.monitor_panel.append_log(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.controller and self.controller.sender_running():
            self.controller.stop_sender()
        if self.controller:
            self.controller.stop_sources()
        if self.log_handler:
            logging.getLogger().removeHandler(self.log_handler)
        super().closeEvent(event)
