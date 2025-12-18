import os
import cv2
import time
import subprocess
import threading
from datetime import datetime

from core.util import dyn_sleep
from core.state import camera_state
from camera.frame_source import FrameSource
from camera.device_selector import CameraDeviceSelector


def _log(msg):
    print(f"[RGBCamera] {msg}")


def _open_capture(dev_path, size, fps):
    """YUYV GStreamer 우선, 실패 시 V4L2(YUYV→NV12)."""
    _log(f"Opening video device: {dev_path}")
    w, h = size
    dev_num = None
    try:
        if isinstance(dev_path, str) and dev_path.startswith("/dev/video"):
            dev_num = int(dev_path.replace("/dev/video", ""))
    except Exception:
        dev_num = None

    # GStreamer 파이프라인(YUY2)
    gst_pipeline = (
        f"v4l2src device={dev_path} io-mode=2 ! "
        f"video/x-raw,format=YUY2,width={w},height={h} ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    _log("GStreamer failed, trying V4L2...")

    def _try_fourcc(fourcc):
        cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
        return cap

    cap = _try_fourcc('YUYV')
    if cap.isOpened():
        return cap
    _log("V4L2 YUYV failed, trying NV12...")
    cap = _try_fourcc('NV12')
    if cap.isOpened():
        return cap

    # 문자열 경로가 실패하면 정수 인덱스로도 시도
    if dev_num is not None:
        _log(f"Trying numeric device index: {dev_num}")
        cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class RGBCamera(FrameSource):
    def __init__(self, cfg, d_buffer):
        super().__init__("RGBCamera")
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.last_ts = None

        self.fps = cfg['FPS']
        self.size = cfg['RES']
        self.sleep = cfg['SLEEP']
        self.device_override = cfg.get('DEVICE_OVERRIDE')
        self._auto_selector = CameraDeviceSelector(
            target_size=cfg.get('RES', (640, 480)),
            min_width=max(320, cfg.get('RES', [0])[0]),
        )

        self.d_buffer = d_buffer
        self.init_cam()

    def __del__(self):
        if self.cap:
            self.cap.release()

    def _gather_candidates(self):
        """우선순위대로 시도할 디바이스 후보 목록을 만든다."""
        candidates = []
        if self.device_override is not None:
            candidates.append(self.device_override)

        try:
            default_dev = self._get_device()
            if default_dev:
                candidates.append(default_dev)
        except NotImplementedError:
            pass
        except Exception as e:
            _log(f"Default device lookup failed: {e}")

        auto_candidate = None
        try:
            auto_candidate = self._auto_selector.choose()
        except Exception as e:
            _log(f"Auto device selection failed: {e}")
        if auto_candidate:
            candidates.append(auto_candidate)

        # 중복 제거(순서 유지)
        unique = []
        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            unique.append(c)
            seen.add(c)
        return unique

    def _normalize_device(self, device):
        """심볼릭 링크를 실제 /dev/videoN 경로로 해석."""
        if isinstance(device, int):
            return f"/dev/video{device}"
        if isinstance(device, str):
            try:
                return os.path.realpath(device)
            except Exception:
                return device
        return str(device)

    def _retry_with_udev(self, dev_path):
        try:
            subprocess.run(
                ["udevadm", "trigger"],
                check=False,
                capture_output=True
            )
            _log("Ran 'udevadm trigger', retrying capture...")
            time.sleep(2.0)
        except Exception as e:
            _log(f"udevadm trigger failed: {e}")
        return _open_capture(dev_path, self.size, self.fps)

    def _print_cap_info(self, device, frame):
        req_w, req_h = self.size
        rep_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rep_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rep_fps = self.cap.get(cv2.CAP_PROP_FPS)
        act_h, act_w = frame.shape[:2]

        _log(f"Connected to {device}")
        _log(
            f"Resolution - requested: {req_w}x{req_h}, "
            f"reported: {rep_w}x{rep_h}, actual: {act_w}x{act_h}"
        )
        _log(f"FPS - requested: {self.fps}, reported: {rep_fps:.2f}")

    def init_cam(self):
        candidates = self._gather_candidates()

        if not candidates:
            _log("No RGB video device configured or detected.")
            return

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        active_device = None
        tried = []
        for candidate in candidates:
            dev_path = self._normalize_device(candidate)
            tried.append(dev_path)
            self.cap = _open_capture(dev_path, self.size, self.fps)
            if not self.cap.isOpened():
                self.cap = self._retry_with_udev(dev_path)
            if self.cap.isOpened():
                active_device = dev_path
                break

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._print_cap_info(active_device or candidates[0], frame)
            else:
                _log(f"Device opened but failed to read frame from {active_device}")
        else:
            unique_tried = ", ".join(dict.fromkeys(tried))
            _log(f"Failed to open RGB device(s): {unique_tried}")
            
    def _get_device(self):
        raise NotImplementedError("_get_device() must be implemented in subclass")
    
    def capture(self):
        ret, frame = self.cap.read() if self.cap else (False, None)
        if not ret or frame is None:
            if not hasattr(self, '_cap_fail_count'):
                self._cap_fail_count = 0
            self._cap_fail_count += 1
            if self._cap_fail_count % 100 == 1:
                _log(f"Capture failed (count: {self._cap_fail_count})")
            return None, None
        
        if frame.size == 0 or frame.ndim < 2 or frame.shape[0] < 2 or frame.shape[1] < 2:
            return None, None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
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
        frame_count = 0
        while not self.stop_event.is_set():
            s_time = time.time()

            frame, ts = self.capture()
            if frame is None:
                dyn_sleep(s_time, self.sleep); continue
            if self.last_ts == ts:
                dyn_sleep(s_time, self.sleep); continue

            self.d_buffer.write((frame, ts))
            frame_count += 1
            if frame_count % 100 == 0:
                _log(f"Captured {frame_count} frames")
        
            self.last_ts = ts
            dyn_sleep(s_time, self.sleep)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class FrontRGBCamera(RGBCamera):
    def _get_device(self):
        for dev in ("/dev/pyro_rgb_cam", "/dev/video5", "/dev/video3"):
            if os.path.exists(dev):
                return dev
