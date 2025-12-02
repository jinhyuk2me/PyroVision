import cv2
import time
import threading
from datetime import datetime

from core.util import dyn_sleep
from core.state import camera_state
from camera.frame_source import FrameSource


class VideoRGBCamera(FrameSource):
    def __init__(self, cfg, d_buffer, paths, loop=True, frame_interval=None):
        super().__init__("VideoRGBCamera")
        if not paths:
            raise ValueError("VideoRGBCamera requires at least one path")
        if isinstance(paths, str):
            paths = [paths]

        self.paths = paths
        self.loop_playlist = loop
        self.path_idx = 0
        self.cap = None
        self._open_current()

        self.loop_file = True
        self.last_ts = None
        self.sleep = frame_interval if frame_interval is not None else cfg['SLEEP']
        self.d_buffer = d_buffer
        self.stop_event = threading.Event()

    def _open_current(self):
        if self.cap:
            self.cap.release()
        current_path = self.paths[self.path_idx]
        self.cap = cv2.VideoCapture(current_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"VideoRGBCamera - cannot open {current_path}")

    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            if self.cap and self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0:
                self.path_idx += 1
                if self.path_idx >= len(self.paths):
                    if self.loop_playlist:
                        self.path_idx = 0
                    else:
                        return None, None
                self._open_current()
                ret, frame = self.cap.read()
                if not ret:
                    return None, None
            else:
                return None, None

        rotate = camera_state.rotate_rgb
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if camera_state.flip_h_rgb:
            frame = cv2.flip(frame, 1)
        if camera_state.flip_v_rgb:
            frame = cv2.flip(frame, 0)

        ts = datetime.now().strftime("%y%m%d%H%M%S%f")[:-4]
        return frame, ts

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self.thread

    def _loop(self):
        while not self.stop_event.is_set():
            s_time = time.time()
            frame, ts = self.capture()
            if frame is None:
                dyn_sleep(s_time, self.sleep)
                continue
            if self.last_ts == ts:
                dyn_sleep(s_time, self.sleep)
                continue

            self.d_buffer.write((frame, ts))
            self.last_ts = ts
            dyn_sleep(s_time, self.sleep)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
