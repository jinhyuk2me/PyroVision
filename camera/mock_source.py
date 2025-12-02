import time
import numpy as np
import threading
from datetime import datetime

from core.util import dyn_sleep
from camera.frame_source import FrameSource


class MockRGBCamera(FrameSource):
    """
    고정/변형 패턴을 생성하는 RGB 가짜 카메라.
    테스트 및 데모용.
    """

    def __init__(self, cfg, d_buffer, color=(0, 255, 0), frame_interval=None):
        super().__init__("MockRGBCamera")
        self.size = cfg['RES']
        self.sleep = frame_interval if frame_interval is not None else cfg['SLEEP']
        self.d_buffer = d_buffer
        self.last_ts = None
        self.color = color
        self.counter = 0
        self.stop_event = threading.Event()

    def _gen_frame(self):
        w, h = self.size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = self.color
        thickness = 20
        offset = (self.counter * 10) % w
        frame[:, offset:offset + thickness] = (0, 0, 255)
        self.counter += 1
        return frame

    def capture(self):
        frame = self._gen_frame()
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


class MockThermalCamera:
    """
    ThermalCamera 인터페이스 호환 가짜 소스 (IRCamera에서 사용).
    """

    def __init__(self, size=(160, 120), frame_interval=None):
        self.width, self.height = size
        self.sleep = frame_interval
        self.counter = 0

    def capture(self):
        pattern = np.indices((self.height, self.width)).sum(axis=0)
        pattern = (pattern + (self.counter * 50)) % 65535
        frame = pattern.astype(np.uint16)
        self.counter += 1
        if self.sleep:
            time.sleep(self.sleep)
        return frame
