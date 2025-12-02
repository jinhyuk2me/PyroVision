import cv2
import numpy as np
import logging
import time


logger = logging.getLogger(__name__)


class VideoThermalCamera:
    """
    ThermalCamera 호환 비디오 소스.
    8/16bit 영상 파일을 불러와 RAW16 프레임으로 변환한다.
    """

    def __init__(self, path, loop=True, target_size=(160, 120), frame_interval=None):
        if isinstance(path, str):
            paths = [path]
        elif isinstance(path, (list, tuple)):
            paths = list(path)
        else:
            raise ValueError("VideoThermalCamera path must be str or list")

        if not paths:
            raise ValueError("VideoThermalCamera requires at least one path")

        self.paths = paths
        self.loop_playlist = loop
        self.path_idx = 0
        self.width, self.height = target_size
        self.frame_interval = frame_interval
        self.cap = None
        self._open_current()

    def _open_current(self):
        if self.cap:
            self.cap.release()
        current = self.paths[self.path_idx]
        self.cap = cv2.VideoCapture(current)
        if not self.cap.isOpened():
            raise RuntimeError(f"VideoThermalCamera - cannot open {current}")

    def _advance(self):
        self.path_idx += 1
        if self.path_idx >= len(self.paths):
            if self.loop_playlist:
                self.path_idx = 0
            else:
                return False
        self._open_current()
        return True

    def _read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        if self._advance():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def capture(self):
        frame = self._read_frame()
        if frame is None:
            return None

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        raw16 = cv2.normalize(gray, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        if self.frame_interval:
            time.sleep(self.frame_interval)
        return raw16

    def stop(self):
        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.cleanup()
