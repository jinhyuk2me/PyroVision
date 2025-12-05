import logging
import time
from collections import deque
from typing import Any, Dict, Optional, Callable

import cv2
from PyQt6.QtCore import Qt

from core.coord_mapper import CoordMapper
from core.fire_fusion import prepare_fusion_for_output
from core.state import DEFAULT_LABEL_SCALE
from core.util import ts_to_epoch_ms
from gui.utils import cv_to_qpixmap, calc_fps, build_overlay
from gui.constants import FPS_HISTORY_SIZE


logger = logging.getLogger(__name__)


class FrameUpdater:
    def __init__(
        self,
        buffers: Dict[str, Any],
        controller: Optional[Any],
        config: Any,
        labels: Dict[str, Any],
        plots: Optional[Dict[str, Any]],
        sync_cfg: Optional[Dict[str, Any]]
    ) -> None:
        self.buffers = buffers
        self.controller = controller
        self.config = config
        self.labels = labels
        self.plots = plots or {}
        self.sync_cfg = sync_cfg or {}
        self.rgb_ts_history = deque(maxlen=FPS_HISTORY_SIZE)
        self.det_ts_history = deque(maxlen=FPS_HISTORY_SIZE)
        self.ir_ts_history = deque(maxlen=FPS_HISTORY_SIZE)
        self._last_det_ts: Optional[str] = None
        self._coord_auto_set: bool = False
        self.ir_size = tuple(self.controller.ir_cfg.get('RES', (160, 120))) if self.controller else (160, 120)
        self.rgb_size = tuple(getattr(self.config, "TARGET_RES", (960, 540)))
        self.fire_fusion: Optional[Any] = None
        self.coord_sync_cb: Optional[Callable] = None

    def set_fire_fusion(self, fusion: Optional[Any]) -> None:
        self.fire_fusion = fusion

    def set_coord_sync_handler(self, callback: Optional[Callable]) -> None:
        """자동 스케일 계산 시 UI를 갱신하기 위한 콜백"""
        self.coord_sync_cb = callback

    def update_frames(self) -> None:
        det_item = self.buffers['rgb_det'].read()
        rgb_item = self.buffers['rgb'].read()
        ir_item = self.buffers['ir'].read()

        det_frame = det_item[0] if det_item else None
        det_meta = det_item[2] if det_item and len(det_item) > 2 else None
        det_count = len(det_meta) if det_meta else 0
        det_ts_str = det_item[1] if det_item else None
        ir_meta = ir_item[2] if ir_item and len(ir_item) > 2 else None
        ir_hotspots = ir_item[3] if ir_item and len(ir_item) > 3 else []
        ir_max = None
        ir_min = None
        if ir_meta and isinstance(ir_meta, dict):
            ir_max = ir_meta.get('temp_corrected', ir_meta.get('temp_raw'))
            ir_min = ir_meta.get('min_temp', None)
        rgb_frame = rgb_item[0] if rgb_item else None
        ir_frame = ir_item[0] if ir_item else None
        t_det = ts_to_epoch_ms(det_ts_str) if det_ts_str else None
        t_rgb = ts_to_epoch_ms(rgb_item[1]) if rgb_item else None
        t_ir = ts_to_epoch_ms(ir_item[1]) if ir_item else None

        if det_item and det_ts_str != self._last_det_ts:
            self._last_det_ts = det_ts_str
            self.det_ts_history.append(time.time() * 1000.0)
        if t_rgb:
            self.rgb_ts_history.append(t_rgb)
        if t_ir:
            self.ir_ts_history.append(t_ir)

        if self.controller and not self._coord_auto_set and rgb_frame is not None and ir_frame is not None:
            coord = self.controller.get_coord_cfg()
            if coord.get('scale') is None:
                try:
                    auto_scale = min(rgb_frame.shape[1] / ir_frame.shape[1], rgb_frame.shape[0] / ir_frame.shape[0])
                    if auto_scale > 0:
                        updated = dict(coord, scale=auto_scale)
                        self.controller.set_coord_cfg(updated)
                        self._coord_auto_set = True
                        logger.info("Coord scale auto-set: %.3f", auto_scale)
                        if self.coord_sync_cb:
                            self.coord_sync_cb(updated)
                except Exception as e:
                    logger.debug("Auto-scale calculation failed: %s", e, exc_info=True)

        vis_mode = getattr(self, "fusion_vis_mode", "test")
        annotated_det = det_frame.copy() if det_frame is not None else None

        # FPS/정보 표시
        if rgb_frame is not None and self.labels.get('rgb_label'):
            pix = cv_to_qpixmap(rgb_frame)
            if pix:
                self.labels['rgb_label'].setPixmap(pix.scaled(
                    self.labels['rgb_label'].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            rgb_dev = "-"
            if self.controller:
                rgb_cfg, _ = self.controller.get_input_cfg()
                rgb_dev = rgb_cfg.get('DEVICE', "-")
            model_name = getattr(self.config, "MODEL", "-") if self.config else "-"
            if self.labels.get('rgb_info'):
                self.labels['rgb_info'].setText(
                    f"RGB {rgb_frame.shape[1]}x{rgb_frame.shape[0]} | fps~{calc_fps(self.rgb_ts_history):.1f} | dev={rgb_dev} | model={model_name}"
                )

        if ir_frame is not None and self.labels.get('ir_label'):
            pix = cv_to_qpixmap(ir_frame)
            if pix:
                self.labels['ir_label'].setPixmap(pix.scaled(
                    self.labels['ir_label'].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            ir_dev = "-"
            if self.controller:
                _, ir_cfg = self.controller.get_input_cfg()
                ir_dev = ir_cfg.get('DEVICE', "-")
            if self.labels.get('ir_info'):
                self.labels['ir_info'].setText(
                    f"IR {ir_frame.shape[1]}x{ir_frame.shape[0]} | fps~{calc_fps(self.ir_ts_history):.1f} | dev={ir_dev}"
                    + (f" | min={ir_min:.1f}C" if ir_min is not None else "")
                    + (f" | max={ir_max:.1f}C" if ir_max is not None else "")
                )

        # Fusion
        fusion_info = "-"
        if det_meta and annotated_det is not None and self.fire_fusion:
            if self.controller:
                params = self.controller.get_coord_cfg()
                self.fire_fusion.coord_mapper = CoordMapper(
                    ir_size=self.ir_size,
                    rgb_size=self.rgb_size,
                    offset_x=params.get('offset_x', 0.0),
                    offset_y=params.get('offset_y', 0.0),
                    scale=params.get('scale'),
                )
            if not isinstance(ir_hotspots, list):
                ir_hotspots = []
            eo_bboxes = [d for d in det_meta if len(d) >= 6]
            fusion = self.fire_fusion.fuse(ir_hotspots, eo_bboxes)
            label_scale = self.controller.get_label_scale() if self.controller else DEFAULT_LABEL_SCALE
            fusion_payload, annotated_det = prepare_fusion_for_output(
                fusion,
                det_frame=annotated_det,
                vis_mode=vis_mode,
                label_scale=label_scale,
                default_label_scale=DEFAULT_LABEL_SCALE,
                json_safe=False,
            )
            if fusion_payload:
                fusion_info = (
                    f"{fusion_payload.get('status','NO_FIRE')} | "
                    f"conf={fusion_payload.get('confidence',0.0):.2f} | "
                    f"ir_hotspot={len(ir_hotspots)} | eo={len(eo_bboxes)}"
                )

        # Detection view 업데이트 (Fusion 후 결과 사용)
        if annotated_det is not None and self.labels.get('det_label'):
            pix = cv_to_qpixmap(annotated_det)
            if pix:
                self.labels['det_label'].setPixmap(pix.scaled(
                    self.labels['det_label'].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            model_name = getattr(self.config, "MODEL", "-") if self.config else "-"
            if self.labels.get('det_info'):
                self.labels['det_info'].setText(
                    f"Det {annotated_det.shape[1]}x{annotated_det.shape[0]} | det={det_count} | model={model_name}"
                )

        # Overlay
        overlay_frame = build_overlay(rgb_frame, ir_frame, self.controller.get_coord_cfg() if self.controller else {})
        if overlay_frame is not None and self.labels.get('overlay_label'):
            pix = cv_to_qpixmap(overlay_frame)
            if pix:
                self.labels['overlay_label'].setPixmap(pix.scaled(
                    self.labels['overlay_label'].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            if self.labels.get('overlay_info'):
                coord = self.controller.get_coord_cfg() if self.controller else {}
                self.labels['overlay_info'].setText(
                    f"Overlay offset=({coord.get('offset_x',0):.1f},{coord.get('offset_y',0):.1f}) scale={coord.get('scale','auto')}"
                )

        # Status / plots
        sender_state = "Sender: ON" if self.controller and self.controller.sender_running() else "Sender: OFF"
        max_diff = self.sync_cfg.get('MAX_DIFF_MS', 120)
        if t_det and t_ir:
            diff = abs(t_det - t_ir)
            sync_state = f"SYNC: {'OK' if diff <= max_diff else f'WARN ({diff:.0f}ms)'}"
        else:
            sync_state = "SYNC: N/A"
        det_fps = calc_fps(self.det_ts_history)
        rgb_fps = calc_fps(self.rgb_ts_history)
        ir_fps = calc_fps(self.ir_ts_history)
        if self.labels.get('status_label'):
            ts_det = det_item[1] if det_item else "-"
            ts_rgb = rgb_item[1] if rgb_item else "-"
            ts_ir = ir_item[1] if ir_item else "-"
            line1 = f"{sender_state} | {sync_state} | MaxDiff={max_diff}ms"
            line2 = f"Det {det_fps:.1f} FPS | IR {ir_fps:.1f} FPS | RGB {rgb_fps:.1f} FPS"
            line3 = f"TS det={ts_det} | rgb={ts_rgb} | ir={ts_ir}"
            self.labels['status_label'].setText("\n".join([line1, line2, line3]))
        if self.plots.get('det_plot'):
            self.plots['det_plot'].update_value(det_fps)
        if self.plots.get('rgb_plot'):
            self.plots['rgb_plot'].update_value(rgb_fps)
        if self.plots.get('ir_plot'):
            self.plots['ir_plot'].update_value(ir_fps)

        # Fusion info text
        if self.labels.get('fusion_info'):
            self.labels['fusion_info'].setText(fusion_info)
