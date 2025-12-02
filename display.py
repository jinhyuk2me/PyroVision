import time
import logging
import cv2
import numpy as np


logger = logging.getLogger(__name__)


def _extract_frame(item):
    if not item:
        return None
    frame = item[0]
    return frame


def display_loop(d_rgb, d_ir, d_rgb_det, window_name="Vision AI Display",
                 target_res=None, refresh_interval=0.03):
    """
    로컬 HDMI 출력용 디스플레이 루프.
    - RGB 검출 프레임을 우선 표시하고, IR 프레임이 있으면 오른쪽에 함께 배치
    - 'q' / ESC / 창 닫기로 종료
    """
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        logger.error("Display window creation failed: %s", e)
        return

    while True:
        rgb_item = d_rgb_det.read() or d_rgb.read()
        ir_item = d_ir.read()

        rgb_frame = _extract_frame(rgb_item)
        ir_frame = _extract_frame(ir_item)

        if rgb_frame is None and ir_frame is None:
            time.sleep(refresh_interval)
            continue

        composed = _compose_frame(rgb_frame, ir_frame, target_res)
        try:
            cv2.imshow(window_name, composed)
        except cv2.error as e:
            logger.error("Display error: %s", e)
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            logger.info("Display window closed via key press")
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Display window closed")
            break

        time.sleep(refresh_interval)

    cv2.destroyWindow(window_name)


def _compose_frame(rgb_frame, ir_frame, target_res):
    frames = []

    if rgb_frame is not None:
        frames.append(_resize_to(rgb_frame, target_res))

    if ir_frame is not None:
        frames.append(_resize_to(ir_frame, target_res, match=frames[0] if frames else None))

    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if len(frames) == 1:
        return frames[0]

    # 높이를 맞춰 가로로 붙임
    h = min(frame.shape[0] for frame in frames)
    resized = [
        cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    return np.hstack(resized)


def _resize_to(frame, target_res, match=None):
    if frame is None:
        return None
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if match is not None:
        target_size = (match.shape[1], match.shape[0])
    elif target_res:
        target_size = (target_res[0], target_res[1])
    else:
        target_size = frame.shape[1], frame.shape[0]

    if (frame.shape[1], frame.shape[0]) == target_size:
        return frame
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
