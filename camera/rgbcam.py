import os
import cv2
import time
import subprocess
import threading

from datetime import datetime
from core.util import dyn_sleep
from core.state import camera_state
from camera.frame_source import FrameSource


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

        self.d_buffer = d_buffer

        self.init_cam()

    def __del__(self):
        if self.cap:
            self.cap.release()

    def _print_cap_info(self, device, frame):
        req_w, req_h = self.size
        rep_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rep_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rep_fps = self.cap.get(cv2.CAP_PROP_FPS)
        act_h, act_w = frame.shape[:2]

        print(f"Connected to {device}")
        print(
            f"Resolution - requested: {req_w}x{req_h}, "
            f"reported: {rep_w}x{rep_h}, actual: {act_w}x{act_h}"
        )
        print(f"FPS - requested: {self.fps}, reported: {rep_fps:.2f}")

    def init_cam(self):
        device_override = getattr(self, 'device_override', None)
        device = device_override if device_override is not None else self._get_device()
        dev_path = device
        dev_num = None
        if isinstance(device, int):
            dev_num = device
            dev_path = f"/dev/video{device}"
        elif isinstance(device, str) and device.startswith("/dev/video"):
            try:
                dev_num = int(device.split("video")[-1])
            except Exception:
                dev_num = None
        
        def _open_cap():
            print(f"Opening video device: {dev_path} (index: {dev_num})")
            
            gst_pipeline = (
                f"v4l2src device={dev_path} ! "
                f"video/x-raw,format=NV12,width={self.size[0]},height={self.size[1]},framerate={self.fps}/1 ! "
                "videoconvert ! appsink"
            )
            
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print("GStreamer backend failed, trying V4L2 with device index...")
                cap = cv2.VideoCapture(dev_num if dev_num is not None else dev_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'NV12'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            return cap
        
        self.cap = _open_cap()
        
        if not self.cap.isOpened():
            try:
                subprocess.run(
                    ["sudo", "udevadm", "trigger"],
                    check=False,
                    capture_output=True
                )
                print("Ran 'udevadm trigger', retrying capture...")
                time.sleep(2.0)
                self.cap = _open_cap()
            except Exception as e:
                print(f"udevadm trigger failed: {e}")

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._print_cap_info(dev_path or self._get_device(), frame)
            else:
                print(f"Device opened but failed to read frame from {dev_path}")
        else:
            print(f"Failed to open device {dev_path}")
            
    def _get_device(self):
        raise NotImplementedError("_get_device() must be implemented in subclass")
    
    def capture(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        
        # 프레임 유효성 검사 (너무 작은 해상도/채널 오류 방지)
        if frame.size == 0 or frame.ndim < 2 or frame.shape[0] < 2 or frame.shape[1] < 2:
            return None, None
        # 1채널 또는 4채널을 BGR 3채널로 변환
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            return None, None
        
        # === 방향 조정 (키보드로 제어) ===
        # 회전 적용
        rotate = camera_state.rotate_rgb
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 좌우반전
        if camera_state.flip_h_rgb:
            frame = cv2.flip(frame, 1)
        
        # 상하반전
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
                dyn_sleep(s_time, self.sleep); continue

            if self.last_ts == ts:
                dyn_sleep(s_time, self.sleep); continue

            self.d_buffer.write((frame, ts))
        
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
        for dev in ("/dev/video3", "/dev/video5"):
            if os.path.exists(dev):
                return dev
