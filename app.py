import cv2
import sys
import time
import select
import threading
import termios
import tty
import os
import logging
import argparse

# from vis import visualize
from sender import send_images
from configs.get_cfg import get_cfg, ConfigError

from camera.source_factory import create_rgb_source, create_ir_source
from detector.tflite import TFLiteWorker
from core.buffer import DoubleBuffer
from core.state import camera_state
from display import display_loop

logger = logging.getLogger(__name__)


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("tflite_runtime").setLevel(logging.WARNING)


def _apply_input_overrides(name, cfg):
    prefix = name.upper()
    mode = os.getenv(f"{prefix}_INPUT_MODE")
    if mode:
        cfg['MODE'] = mode
    video_path = os.getenv(f"{prefix}_VIDEO_PATH")
    if video_path:
        if ';' in video_path:
            cfg['VIDEO_PATH'] = [p.strip() for p in video_path.split(';') if p.strip()]
        else:
            cfg['VIDEO_PATH'] = video_path
    loop = os.getenv(f"{prefix}_LOOP")
    if loop:
        cfg['LOOP'] = loop.lower() in ('1', 'true', 'yes', 'on')
    interval = os.getenv(f"{prefix}_FRAME_INTERVAL_MS")
    if interval:
        cfg['FRAME_INTERVAL_MS'] = int(interval)


def setup_keyboard():
    """터미널을 raw 모드로 설정 (키 입력 즉시 감지)"""
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return old_settings
    except:
        return None


def restore_keyboard(old_settings):
    """터미널 설정 복원"""
    if old_settings:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except:
            pass


def check_keyboard():
    """
    키보드 입력 확인 (논블로킹)
    Returns: 입력된 키 또는 None
    """
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except:
        pass
    return None


def print_help():
    """키보드 단축키 도움말 출력"""
    print("\n" + "=" * 55)
    print("Keyboard Controls:")
    print("-" * 55)
    print("  IR Camera:")
    print("    [1] Rotate IR 90 degrees (clockwise)")
    print("    [2] Toggle IR horizontal flip (left-right)")
    print("    [3] Toggle IR vertical flip (up-down)")
    print("-" * 55)
    print("  RGB Camera:")
    print("    [4] Rotate RGB 90 degrees (clockwise)")
    print("    [5] Toggle RGB horizontal flip (left-right)")
    print("    [6] Toggle RGB vertical flip (up-down)")
    print("-" * 55)
    print("  Both Cameras:")
    print("    [7] Toggle BOTH horizontal flip")
    print("    [8] Toggle BOTH vertical flip")
    print("-" * 55)
    print("  [s] Show current status")
    print("  [h] Show this help message")
    print("  [q] Quit application")
    print("=" * 55 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Vision AI Application")
    parser.add_argument("--mode", choices=["cli", "gui"], default=None,
                        help="실행 모드 선택 (cli | gui), 기본값은 APP_MODE 또는 cli")
    return parser.parse_args()

def _normalize_coord_cfg(params):
    """CONFIG의 COORD 키(대문자/소문자)를 런타임에서 쓰는 소문자 키로 정규화"""
    cfg = dict(params or {})
    def _get(key, default=None):
        val = cfg.get(key)
        if val is None:
            val = cfg.get(key.upper(), default)
        return val if val is not None else default
    out = {
        'offset_x': float(_get('offset_x', 0.0)),
        'offset_y': float(_get('offset_y', 0.0)),
        'scale': _get('scale', None),
    }
    if out['scale'] is not None:
        try:
            out['scale'] = float(out['scale'])
        except Exception:
            out['scale'] = None
    return out


class CoordState:
    def __init__(self, params=None):
        self._lock = threading.Lock()
        self._params = dict(params or {'offset_x': 0.0, 'offset_y': 0.0, 'scale': None})
        self._version = 0

    def get(self):
        with self._lock:
            return dict(self._params), self._version

    def update(self, **kwargs):
        with self._lock:
            self._params.update({k: v for k, v in kwargs.items() if v is not None})
            self._version += 1


class RuntimeController:
    """
    런타임 파이프라인을 묶어 관리하는 컨트롤러.
    - 카메라 소스/탐지 워커 시작·정지
    - 송신(TCP)/디스플레이 제어
    - 좌표/캡처 설정 공유
    """
    def __init__(self, buffers, server, sync_cfg, display_cfg, target_res, coord_cfg, capture_cfg, cfg=None):
        self.buffers = buffers
        self.server = server
        self.sync_cfg = sync_cfg
        self.display_cfg = display_cfg
        self.target_res = target_res
        self.coord_state = CoordState(_normalize_coord_cfg(coord_cfg))
        self.capture_cfg = capture_cfg or {}
        self.cfg = cfg or {}
        self.sender_thread = None
        self.sender_stop = threading.Event()
        self.display_thread = None
        self.display_enabled = False
        self.rgb_source = None
        self.ir_source = None
        self.rgb_cfg = None
        self.ir_cfg = None
        self.rgb_input_cfg = None
        self.ir_input_cfg = None
        self.detector_worker = None
        self.detector_cfg = {}
        self._threads = {}

    def _start_thread(self, name, target, args=(), kwargs=None):
        if name in self._threads and self._threads[name].is_alive():
            return False
        t = threading.Thread(target=target, args=args, kwargs=kwargs or {}, daemon=True)
        self._threads[name] = t
        t.start()
        return True

    def _stop_thread(self, name, stop_event=None, join_timeout=2.0):
        t = self._threads.get(name)
        if not t:
            return False
        if stop_event:
            stop_event.set()
        t.join(timeout=join_timeout)
        self._threads.pop(name, None)
        return True

    def start_sender(self):
        self.sender_stop.clear()
        kwargs = {
            "host": self.server['IP'],
            "port": self.server['PORT'],
            "jpeg_quality": self.server.get('COMP_RATIO', 70),
            "sync_cfg": self.sync_cfg,
            "stop_event": self.sender_stop,
            "coord_state": self.coord_state,
        }
        return self._start_thread(
            "sender",
            target=send_images,
            args=(
                self.buffers['rgb'],
                self.buffers['ir'],
                self.buffers['ir16'],
                self.buffers['rgb_det'],
            ),
            kwargs=kwargs,
        )

    def stop_sender(self):
        return self._stop_thread("sender", stop_event=self.sender_stop)

    def sender_running(self):
        t = self._threads.get("sender")
        return t is not None and t.is_alive()

    def start_display(self):
        if self.display_enabled:
            return False
        self.display_enabled = True
        window_name = self.display_cfg.get('WINDOW_NAME', "Vision AI Display")
        return self._start_thread(
            "display",
            target=display_loop,
            args=(self.buffers['rgb'], self.buffers['ir'], self.buffers['rgb_det']),
            kwargs={"window_name": window_name, "target_res": self.target_res},
        )

    def stop_display(self):
        self.display_enabled = False
        return self._stop_thread("display")

    def display_running(self):
        t = self._threads.get("display")
        return self.display_enabled and t is not None and t.is_alive()

    def set_sources(self, rgb_source, ir_source, rgb_cfg, ir_cfg, rgb_input_cfg, ir_input_cfg):
        self.rgb_source = rgb_source
        self.ir_source = ir_source
        self.rgb_cfg = rgb_cfg
        self.ir_cfg = ir_cfg
        self.rgb_input_cfg = dict(rgb_input_cfg)
        self.ir_input_cfg = dict(ir_input_cfg)

    def set_detector(self, worker, det_cfg):
        self.detector_worker = worker
        self.detector_cfg = dict(det_cfg or {})

    def stop_sources(self):
        if self.rgb_source:
            try:
                self.rgb_source.stop()
            except Exception as exc:
                logger.warning("RGB source stop failed: %s", exc)
        if self.ir_source:
            try:
                self.ir_source.stop()
            except Exception as exc:
                logger.warning("IR source stop failed: %s", exc)

    def restart_sources(self, rgb_input_cfg=None, ir_input_cfg=None):
        if rgb_input_cfg:
            self.rgb_input_cfg = dict(rgb_input_cfg)
        if ir_input_cfg:
            self.ir_input_cfg = dict(ir_input_cfg)
        self.stop_sources()
        self.rgb_source = create_rgb_source(self.rgb_cfg, self.rgb_input_cfg, self.buffers['rgb'])
        self.ir_source = create_ir_source(self.ir_cfg, self.ir_input_cfg, self.buffers['ir'], self.buffers['ir16'])
        self.rgb_source.start()
        self.ir_source.start()
        return True

    def restart_ir_source(self):
        """IR 소스만 재시작 (RGB는 유지)"""
        if self.ir_source:
            try:
                self.ir_source.stop()
            except Exception:
                pass
        self.ir_source = create_ir_source(self.ir_cfg, self.ir_input_cfg, self.buffers['ir'], self.buffers['ir16'])
        self.ir_source.start()
        return True

    def get_input_cfg(self):
        return dict(self.rgb_input_cfg or {}), dict(self.ir_input_cfg or {})

    def get_coord_cfg(self):
        params, _ = self.coord_state.get()
        return params

    def set_coord_cfg(self, params):
        self.coord_state.update(**params)

    def get_sync_cfg(self):
        return dict(self.sync_cfg or {})

    def get_capture_cfg(self):
        return dict(self.capture_cfg or {})

    def update_ir_fire_cfg(self, fire_enabled=None, min_temp=None, thr=None, raw_thr=None, tau=None, restart=False):
        """IR 화점 탐지 관련 설정 업데이트. 기본은 런타임 적용, 필요 시 restart=True로 재시작"""
        ir = dict(self.ir_cfg or {})
        if 'FIRE_DETECTION' in ir and fire_enabled is not None:
            ir['FIRE_DETECTION'] = bool(fire_enabled)
        if min_temp is not None:
            ir['FIRE_MIN_TEMP'] = float(min_temp)
        if thr is not None:
            ir['FIRE_THR'] = float(thr)
        if raw_thr is not None:
            ir['FIRE_RAW_THR'] = float(raw_thr)
        if tau is not None:
            ir['TAU'] = float(tau)
        self.ir_cfg.update(ir)

        # 런타임 적용 지원 시 바로 반영
        if self.ir_source and hasattr(self.ir_source, "update_fire_params"):
            try:
                self.ir_source.update_fire_params(
                    fire_detection=fire_enabled,
                    min_temp=min_temp,
                    thr=thr,
                    raw_thr=raw_thr,
                    tau=tau,
                )
                if not restart:
                    return True
            except Exception:
                pass

        if restart:
            return self.restart_ir_source()
        return True

    def get_detector_cfg(self):
        return dict(self.detector_cfg or {})

    def restart_detector(self):
        """현재 설정으로 탐지 워커 재시작"""
        if self.detector_worker:
            try:
                self.detector_worker.stop()
                self.detector_worker.join(timeout=2.0)
            except Exception:
                pass
        cfg = self.detector_cfg or {}
        model_path = cfg.get('MODEL')
        labels_path = cfg.get('LABEL')
        delegate = cfg.get('DELEGATE')
        allowed = cfg.get('ALLOWED_CLASSES')
        use_npu = bool(cfg.get('USE_NPU', False))
        cpu_threads = cfg.get('CPU_THREADS', 1)
        conf_thr = float(cfg.get('CONF_THR', cfg.get('CONF_THRESHOLD', 0.15)))
        name = cfg.get('NAME', "DetRGB")
        new_worker = TFLiteWorker(
            model_path=model_path,
            labels_path=labels_path,
            input_buf=self.buffers['rgb'],
            output_buf=self.buffers['rgb_det'],
            allowed_class_ids=allowed,
            use_npu=use_npu,
            delegate_lib=delegate,
            cpu_threads=cpu_threads,
            target_fps=self.rgb_cfg.get('FPS', 30),
            target_res=self.target_res,
            conf_thr=conf_thr,
            name=name
        )
        new_worker.start()
        self.detector_worker = new_worker
        return True

    def update_detector_cfg(self, model_path=None, label_path=None, delegate=None, allowed_classes=None, use_npu=None, cpu_threads=None, conf_thr=None, restart=True):
        cfg = dict(self.detector_cfg or {})
        if model_path:
            cfg['MODEL'] = model_path
        if label_path:
            cfg['LABEL'] = label_path
        if delegate is not None:
            cfg['DELEGATE'] = delegate
        if allowed_classes is not None:
            cfg['ALLOWED_CLASSES'] = allowed_classes if allowed_classes else None
        if use_npu is not None:
            cfg['USE_NPU'] = bool(use_npu)
        if cpu_threads is not None:
            cfg['CPU_THREADS'] = int(cpu_threads)
        if conf_thr is not None:
            cfg['CONF_THR'] = float(conf_thr)
        self.detector_cfg = cfg
        if restart:
            return self.restart_detector()
        return True

    def status(self):
        """송신/디스플레이/소스/탐지기 상태를 요약"""
        return {
            "sender": self.sender_running(),
            "display": self.display_running(),
            "rgb_source": getattr(self.rgb_source, "thread", None) is not None and getattr(self.rgb_source.thread, "is_alive", lambda: False)(),
            "ir_source": getattr(self.ir_source, "thread", None) is not None and getattr(self.ir_source.thread, "is_alive", lambda: False)(),
            "detector": self.detector_worker is not None and self.detector_worker.is_alive(),
        }


def _load_config():
    cfg = get_cfg()
    if cfg is None:
        raise RuntimeError("Config is empty or invalid")
    return cfg


def _build_buffers():
    d16_ir, d_ir = DoubleBuffer(), DoubleBuffer()
    d_rgb, d_rgb_det = DoubleBuffer(), DoubleBuffer()
    return {
        'rgb': d_rgb,
        'rgb_det': d_rgb_det,
        'ir': d_ir,
        'ir16': d16_ir,
    }


def _start_sources(ir_cfg, ir_input_cfg, rgb_cfg, rgb_input_cfg, buffers):
    logger.info("IR source - Starting (%s)", ir_input_cfg.get('MODE', 'live'))
    ir_source = create_ir_source(ir_cfg, ir_input_cfg, buffers['ir'], buffers['ir16'])
    ir_source.start()

    logger.info("RGB source - Starting (%s)", rgb_input_cfg.get('MODE', 'live'))
    rgb_source = create_rgb_source(rgb_cfg, rgb_input_cfg, buffers['rgb'])
    rgb_source.start()

    return rgb_source, ir_source


def _start_detector(cfg, rgb_cfg, buffers, delegate, model, label):
    rgb_det_cfg = {
        'MODEL': model,
        'LABEL': label,
        'DELEGATE': delegate,
        'ALLOWED_CLASSES': [1],
        'USE_NPU': True,
        'CPU_THREADS': 1,
        'CONF_THR': float(getattr(cfg, 'CONF_THR', getattr(cfg, 'CONF_THRESHOLD', 0.15))),
        'NAME': "DetRGB",
    }
    worker = TFLiteWorker(
        model_path=model,
        labels_path=label,
        input_buf=buffers['rgb'],
        output_buf=buffers['rgb_det'],
        allowed_class_ids=rgb_det_cfg['ALLOWED_CLASSES'],
        use_npu=rgb_det_cfg['USE_NPU'],
        delegate_lib=delegate,
        cpu_threads=rgb_det_cfg['CPU_THREADS'],
        target_fps=rgb_cfg['FPS'],
        target_res=tuple(getattr(cfg, 'TARGET_RES', (rgb_cfg.get('RES', [0, 0])[0], rgb_cfg.get('RES', [0, 0])[1]))),
        conf_thr=rgb_det_cfg['CONF_THR'],
        name=rgb_det_cfg['NAME'],
    )
    worker.start()
    return worker, rgb_det_cfg


def _init_pipeline(gui_mode=False):
    cfg = _load_config()
    model = cfg.MODEL
    label = cfg.LABEL
    server = cfg.SERVER
    delegate = cfg.DELEGATE
    ir_cfg = cfg.CAMERA_IR.__dict__
    rgb_cfg = cfg.CAMERA_RGB_FRONT.__dict__
    state = cfg.STATE
    target_res = tuple(cfg.TARGET_RES)
    display_cfg = cfg.DISPLAY
    sync_cfg = cfg.SYNC
    input_cfg = cfg.INPUT

    rgb_input_cfg = dict(input_cfg.get('RGB', {})) if isinstance(input_cfg, dict) else {}
    ir_input_cfg = dict(input_cfg.get('IR', {})) if isinstance(input_cfg, dict) else {}
    _apply_input_overrides("RGB", rgb_input_cfg)
    _apply_input_overrides("IR", ir_input_cfg)

    display_enabled = False
    display_window = "Vision AI Display"
    if isinstance(display_cfg, dict):
        display_enabled = display_cfg.get('ENABLED', False)
        display_window = display_cfg.get('WINDOW_NAME', display_window)
    else:
        display_enabled = bool(display_cfg)
    if gui_mode:
        display_enabled = False

    buffers = _build_buffers()

    try:
        rgb_source, ir_source = _start_sources(ir_cfg, ir_input_cfg, rgb_cfg, rgb_input_cfg, buffers)
    except Exception as e:
        logger.exception("Camera source start failed: %s", e)
        raise

    try:
        rgb_det, rgb_det_cfg = _start_detector(cfg, rgb_cfg, buffers, delegate, model, label)
    except Exception as e:
        logger.exception("RGB-TFLite - Start failed: %s", e)
        raise

    coord_cfg = cfg.COORD
    capture_cfg = cfg.CAPTURE
    controller = RuntimeController(
        buffers,
        server,
        sync_cfg,
        display_cfg if isinstance(display_cfg, dict) else {},
        target_res,
        coord_cfg,
        capture_cfg,
        cfg=cfg
    )
    controller.set_sources(rgb_source, ir_source, rgb_cfg, ir_cfg, rgb_input_cfg, ir_input_cfg)
    if rgb_det:
        controller.set_detector(rgb_det, rgb_det_cfg)

    return {
        'cfg': cfg,
        'controller': controller,
        'display_enabled': display_enabled,
        'display_window': display_window,
        'gui_mode': gui_mode,
    }


def _run_cli(ctx):
    controller = ctx['controller']
    display_enabled = ctx['display_enabled']
    gui_mode = ctx['gui_mode']

    if not gui_mode:
        try:
            logger.info("TCP Sender - Starting")
            controller.start_sender()
        except Exception as e:
            logger.exception("TCP Sender - Start failed: %s", e)

    if display_enabled:
        try:
            logger.info("Display - Starting")
            controller.start_display()
        except Exception as e:
            logger.exception("Display - Start failed: %s", e)

    old_settings = setup_keyboard()
    print_help()
    try:
        while True:
            key = check_keyboard()

            if key == '1':
                angle = camera_state.rotate_ir_cw()
                logger.info("[IR] Rotation: %s degrees", angle)
            elif key == '2':
                state = camera_state.toggle_flip_h_ir()
                logger.info("[IR] Horizontal flip: %s", "ON" if state else "OFF")
            elif key == '3':
                state = camera_state.toggle_flip_v_ir()
                logger.info("[IR] Vertical flip: %s", "ON" if state else "OFF")
            elif key == '4':
                angle = camera_state.rotate_rgb_cw()
                logger.info("[RGB] Rotation: %s degrees", angle)
            elif key == '5':
                state = camera_state.toggle_flip_h_rgb()
                logger.info("[RGB] Horizontal flip: %s", "ON" if state else "OFF")
            elif key == '6':
                state = camera_state.toggle_flip_v_rgb()
                logger.info("[RGB] Vertical flip: %s", "ON" if state else "OFF")
            elif key == '7':
                state = camera_state.toggle_flip_h_both()
                logger.info("[BOTH] Horizontal flip: %s", "ON" if state else "OFF")
            elif key == '8':
                state = camera_state.toggle_flip_v_both()
                logger.info("[BOTH] Vertical flip: %s", "ON" if state else "OFF")
            elif key == 's':
                status = camera_state.get_status()
                ir = status['ir']
                rgb = status['rgb']
                logger.info(
                    "[Status] IR rotate=%3d flip_h=%s flip_v=%s",
                    ir['rotate'], "ON" if ir['flip_h'] else "OFF", "ON" if ir['flip_v'] else "OFF"
                )
                logger.info(
                    "[Status] RGB rotate=%3d flip_h=%s flip_v=%s",
                    rgb['rotate'], "ON" if rgb['flip_h'] else "OFF", "ON" if rgb['flip_v'] else "OFF"
                )
            elif key == 'h':
                print_help()
            elif key == 'q':
                logger.info("Shutting down...")
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if controller:
            controller.stop_sources()
        restore_keyboard(old_settings)


def _run_gui(ctx):
    try:
        from gui.app_gui import run_gui
    except ImportError as e:
        logger.error("GUI mode requested but PyQt6 not available: %s", e)
        sys.exit(1)

    run_gui(
        ctx['controller'].buffers,
        camera_state,
        ctx['controller'],
    )


def main():
    args = parse_args()
    env_mode = os.getenv("APP_MODE", "cli").lower()
    selected_mode = (args.mode or env_mode).lower()
    gui_mode = selected_mode == "gui"

    setup_logging()
    cv2.ocl.setUseOpenCL(True)

    try:
        ctx = _init_pipeline(gui_mode=gui_mode)
    except ConfigError as e:
        logger.error("Config error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline init failed: %s", e)
        sys.exit(1)

    if gui_mode:
        _run_gui(ctx)
    else:
        _run_cli(ctx)


if __name__ == "__main__":
    main()
