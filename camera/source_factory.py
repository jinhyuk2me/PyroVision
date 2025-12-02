import glob
import os
import re
import shutil
import subprocess

from camera.rgbcam import FrontRGBCamera
from camera.rgb_video import VideoRGBCamera
from camera.ircam import IRCamera
from camera.purethermal.video_thermal import VideoThermalCamera
from camera.purethermal.thermalcamera import ThermalCamera
from camera.mock_source import MockRGBCamera, MockThermalCamera


def _parse_paths(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value:
        if ';' in value:
            return [p.strip() for p in value.split(';') if p.strip()]
        return value
    return None


def _parse_interval(mode_cfg, key='FRAME_INTERVAL_MS', default=None):
    interval_ms = mode_cfg.get(key)
    if interval_ms is None:
        return default
    try:
        return max(0.0, float(interval_ms) / 1000.0)
    except Exception:
        return default


def _list_video_devices():
    """읽기 가능한 /dev/video* 경로 나열"""
    devices = []
    for path in glob.glob("/dev/video*"):
        if os.access(path, os.R_OK):
            devices.append(path)
    devices.sort(key=lambda p: int(re.sub(r"\D", "", p) or 0))
    return devices


def _probe_device_max_resolution(device):
    """v4l2-ctl로 지원 최대 해상도 추정 (실패 시 None)"""
    v4l2_cmd = shutil.which("v4l2-ctl") or "/usr/bin/v4l2-ctl"
    if os.path.exists(v4l2_cmd):
        try:
            res = subprocess.run(
                [v4l2_cmd, "--device", device, "--list-formats-ext"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            sizes = re.findall(r"Size:\s+Discrete\s+(\d+)x(\d+)", res.stdout or "")
            if sizes:
                sizes_int = [(int(w), int(h)) for w, h in sizes]
                sizes_int.sort(key=lambda wh: wh[0] * wh[1], reverse=True)
                return sizes_int[0]
        except Exception:
            pass

    # v4l2-ctl을 사용 못할 때는 프레임을 직접 읽어 추정
    try:
        import cv2  # 지연 로드
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            try:
                dev_num = int(re.sub(r"\D", "", str(device)) or -1)
                if dev_num >= 0:
                    cap = cv2.VideoCapture(dev_num, cv2.CAP_V4L2)
            except Exception:
                pass
        if not cap.isOpened():
            cap.release()
            return None
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.ndim >= 2:
            h, w = frame.shape[:2]
            return w, h
    except Exception:
        return None


def _auto_select_device(target_size=(640, 480), min_width=320):
    """목표 해상도에 가장 근접한 비디오 장치 자동 선택"""
    target_w = target_size[0] if target_size else 640
    devices = _list_video_devices()
    best = None
    best_score = -1
    for dev in devices:
        res = _probe_device_max_resolution(dev)
        if not res:
            continue
        w, h = res
        if w < min_width:
            continue
        score = w * h
        if score > best_score:
            best_score = score
            best = dev
    if best:
        return best
    # 해상도 정보를 못 얻어도 장치가 있으면 첫 번째로 fallback
    if devices:
        return devices[0]
    return None


def create_rgb_source(rgb_cfg, mode_cfg, buffer):
    mode = str(mode_cfg.get('MODE', 'live') or 'live').lower()
    frame_interval = _parse_interval(mode_cfg, default=rgb_cfg.get('SLEEP'))
    if mode == 'video':
        paths = _parse_paths(mode_cfg.get('VIDEO_PATH'))
        if not paths:
            raise ValueError("RGB video mode requires VIDEO_PATH")
        loop = mode_cfg.get('LOOP', True)
        return VideoRGBCamera(rgb_cfg, buffer, paths, loop=loop, frame_interval=frame_interval)
    if mode == 'mock':
        color = tuple(mode_cfg.get('COLOR', (0, 255, 0)))
        return MockRGBCamera(rgb_cfg, buffer, color=color, frame_interval=frame_interval)
    if mode != 'live':
        raise ValueError(f"Unsupported RGB input mode: {mode}")

    device = mode_cfg.get('DEVICE', rgb_cfg.get('DEVICE'))
    if device == "":
        device = None
    if device is None or str(device).lower() == "auto":
        auto_dev = _auto_select_device(target_size=rgb_cfg.get('RES'), min_width=max(320, rgb_cfg.get('RES', [0])[0]))
        if auto_dev:
            device = auto_dev
    if device is None:
        device = rgb_cfg.get('DEVICE')
    if device is not None:
        cfg = dict(rgb_cfg)
        cfg['DEVICE_OVERRIDE'] = device
        return FrontRGBCamera(cfg, buffer)
    return FrontRGBCamera(rgb_cfg, buffer)


def create_ir_source(ir_cfg, mode_cfg, ir_buffer, d16_buffer):
    mode = str(mode_cfg.get('MODE', 'live') or 'live').lower()
    cam_impl = None
    frame_interval = _parse_interval(mode_cfg)
    if mode == 'video':
        paths = _parse_paths(mode_cfg.get('VIDEO_PATH'))
        if not paths:
            raise ValueError("IR video mode requires VIDEO_PATH")
        loop = mode_cfg.get('LOOP', True)
        target_size = tuple(ir_cfg['RES'])
        cam_impl = VideoThermalCamera(paths, loop=loop, target_size=target_size, frame_interval=frame_interval)
    elif mode == 'mock':
        target_size = tuple(ir_cfg['RES'])
        cam_impl = MockThermalCamera(size=target_size, frame_interval=frame_interval)
    elif mode == 'live':
        device = mode_cfg.get('DEVICE', ir_cfg.get('DEVICE'))
        if device == "":
            device = None
        if device:
            cam_impl = ThermalCamera(device_path=device)
    else:
        raise ValueError(f"Unsupported IR input mode: {mode}")
    return IRCamera(ir_cfg, ir_buffer, d16_buffer, cam_impl=cam_impl)
