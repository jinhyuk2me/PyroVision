import logging
import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap

from core.coord_mapper import CoordMapper

logger = logging.getLogger(__name__)


def cv_to_qpixmap(frame):
    if frame is None:
        return None
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    frame_rgb = frame[:, :, ::-1].copy()
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def calc_fps(history_ms):
    if len(history_ms) < 2:
        return 0.0
    duration = history_ms[-1] - history_ms[0]
    if duration <= 0:
        return 0.0
    return (len(history_ms) - 1) * 1000.0 / duration


def overlay_text(frame, lines, margin=6, line_height=None, color=(255, 255, 255), font_scale=None):
    if frame is None:
        return None
    out = frame.copy()
    h, w = out.shape[:2]
    min_dim = min(h, w)
    base_dim = 720.0
    base_scale = 0.5
    scale = font_scale if font_scale is not None else max(0.25, min(0.9, (min_dim / base_dim) * base_scale))
    lh = line_height if line_height is not None else max(12, int(18 * (scale / base_scale)))
    y = margin + lh
    for line in lines:
        cv2.putText(out, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += lh
    return out


def build_overlay(rgb_frame, ir_frame, params):
    if rgb_frame is None or ir_frame is None:
        return None
    if rgb_frame.size == 0 or ir_frame.size == 0:
        return None
    # 채널 정규화
    if rgb_frame.ndim == 2:
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2BGR)
    elif rgb_frame.shape[2] == 4:
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGRA2BGR)
    if ir_frame.ndim == 2:
        ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
    elif ir_frame.shape[2] == 4:
        ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGRA2BGR)

    rgb_h, rgb_w = rgb_frame.shape[:2]
    ir_h, ir_w = ir_frame.shape[:2]
    mapper = CoordMapper(
        ir_size=(ir_w, ir_h),
        rgb_size=(rgb_w, rgb_h),
        offset_x=params.get('offset_x', 0.0),
        offset_y=params.get('offset_y', 0.0),
        scale=params.get('scale'),
    )
    scale = mapper.scale
    target_w = max(1, int(ir_w * scale))
    target_h = max(1, int(ir_h * scale))
    ir_resized = cv2.resize(ir_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    overlay = rgb_frame.copy()
    x = int(mapper.base_offset_x + mapper.offset_x)
    y = int(mapper.base_offset_y + mapper.offset_y)

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(rgb_w, x + target_w)
    y1 = min(rgb_h, y + target_h)

    if x0 >= x1 or y0 >= y1:
        return overlay

    ir_x0 = x0 - x
    ir_y0 = y0 - y
    ir_x1 = ir_x0 + (x1 - x0)
    ir_y1 = ir_y0 + (y1 - y0)

    alpha = 0.4
    roi_rgb = overlay[y0:y1, x0:x1]
    roi_ir = ir_resized[ir_y0:ir_y1, ir_x0:ir_x1]
    cv2.addWeighted(roi_ir, alpha, roi_rgb, 1 - alpha, 0, dst=roi_rgb)
    return overlay
