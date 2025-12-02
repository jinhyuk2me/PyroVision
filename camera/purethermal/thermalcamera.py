"""
PureThermal/FLIR Lepton Thermal Camera Module

V4L2 기반 구현 - libuvc 대신 커널 UVC 드라이버 사용
v4l2-ctl을 통한 스트리밍 캡처
"""
import os
import fcntl
import numpy as np
import subprocess
import logging

from queue import Queue
from threading import Thread, Event

# USB reset ioctl
USBDEVFS_RESET = 21780
logger = logging.getLogger(__name__)


class ThermalCamera:
    """
    V4L2 기반 열화상 카메라 클래스
    
    PureThermal 보드를 통해 FLIR Lepton 센서에서 Y16 raw 데이터를 캡처합니다.
    커널 uvcvideo 드라이버를 사용하여 안정적으로 동작합니다.
    """
    
    # PureThermal USB VID/PID
    USB_VID = "1e4e"
    USB_PID = "0100"
    
    def __init__(self, device_path=None, reset_on_init=True):
        """
        열화상 카메라 초기화
        
        Args:
            device_path: 비디오 장치 경로 (None이면 자동 탐지)
            reset_on_init: 초기화 시 USB 리셋 수행 여부
        """
        self.BUF_SIZE = 4
        self.streaming = False
        self.q = Queue(self.BUF_SIZE)
        
        self.capture_thread = None
        self.stop_event = Event()
        self.proc = None
        
        self.width = 160
        self.height = 120
        
        # USB 리셋 후 장치 탐지
        if reset_on_init:
            self._reset_usb_device()
        
        self.device_path = device_path or self._find_thermal_device()
        
        if self.device_path:
            logger.info("[ThermalCam] Using %s", self.device_path)
        else:
            logger.warning("PureThermal device not found")
    
    def _find_usb_device_path(self):
        """PureThermal USB 장치 경로 찾기 (/dev/bus/usb/XXX/YYY)"""
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if self.USB_VID in line:
                    # "Bus 003 Device 006: ID 1e4e:0100 ..." 파싱
                    parts = line.split()
                    if len(parts) >= 4:
                        bus = parts[1]
                        device = parts[3].rstrip(':')
                        return f"/dev/bus/usb/{bus}/{device}"
        except:
            pass
        return None
    
    def _reset_usb_device(self):
        """USB 장치 리셋으로 카메라 상태 복구"""
        usb_path = self._find_usb_device_path()
        if not usb_path:
            return False
        
        try:
            fd = os.open(usb_path, os.O_WRONLY)
            fcntl.ioctl(fd, USBDEVFS_RESET, 0)
            os.close(fd)
            logger.info("[ThermalCam] USB reset: %s", usb_path)
            
            # 장치 재인식 대기
            import time
            time.sleep(2)
            return True
        except Exception as e:
            logger.error("[ThermalCam] USB reset failed: %s", e)
            return False
    
    def _find_thermal_device(self):
        """PureThermal 장치 자동 탐지"""
        try:
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True, text=True, timeout=5
            )
            
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'PureThermal' in line:
                    # 다음 줄에서 /dev/videoX 찾기
                    for j in range(i+1, min(i+4, len(lines))):
                        if '/dev/video' in lines[j]:
                            device = lines[j].strip()
                            logger.info("[ThermalCam] Found PureThermal at %s", device)
                            return device
        except Exception as e:
            logger.error("[ThermalCam] Device search failed: %s", e)
        
        return None
    
    def _set_format(self):
        """v4l2-ctl을 사용하여 Y16 포맷 설정"""
        try:
            subprocess.run([
                'v4l2-ctl', '-d', self.device_path,
                '--set-fmt-video', f'width={self.width},height={self.height},pixelformat=Y16 '
            ], capture_output=True, timeout=5)
        except Exception as e:
            logger.error("[ThermalCam] Set format failed: %s", e)
    
    def _capture_loop(self):
        """백그라운드 캡처 루프 - v4l2-ctl 스트리밍 사용"""
        import select
        
        try:
            # Y16 포맷 설정
            self._set_format()
            
            # 스트리밍 프로세스 시작
            self.proc = subprocess.Popen(
                ['v4l2-ctl', '-d', self.device_path,
                 '--stream-mmap', '--stream-count=0', '--stream-to=-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 2
            )
            
            frame_size = self.width * self.height * 2
            
            while not self.stop_event.is_set():
                # 데이터 읽기
                data = self.proc.stdout.read(frame_size)
                
                if len(data) == frame_size:
                    frame = np.frombuffer(data, dtype=np.uint16).reshape(self.height, self.width)
                    if not self.q.full():
                        self.q.put(frame.copy())
                elif len(data) == 0:
                    break
                    
        except Exception as e:
            logger.exception("[ThermalCam] Capture error: %s", e)
        finally:
            if hasattr(self, 'proc') and self.proc:
                self.proc.terminate()
                self.proc = None
    
    def capture(self):
        """
        단일 프레임 캡처
        
        Returns:
            numpy.ndarray: Y16 raw 데이터 (160x120, uint16) 또는 None
        """
        if not self.device_path:
            return None
        
        if not self.streaming:
            self.streaming = True
            self.stop_event.clear()
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        
        try:
            return self.q.get(timeout=1)
        except:
            return None
    
    def cleanup(self):
        """리소스 정리"""
        self.stop_event.set()
        
        # 스트리밍 프로세스 종료
        if hasattr(self, 'proc') and self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except:
                pass
            self.proc = None
        
        # 캡처 스레드 종료 대기
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        self.streaming = False
    
    def __del__(self):
        self.cleanup()
    
    # FFC 관련 기능 (V4L2에서는 지원되지 않음 - 더미 구현)
    def performffc(self):
        """FFC(Flat Field Correction) 수행 - V4L2에서는 미지원"""
        pass
    
    def print_shutter_info(self):
        """셔터 정보 출력 - V4L2에서는 미지원"""
        pass
    
    def setmanualffc(self):
        """수동 FFC 모드 설정 - V4L2에서는 미지원"""
        pass
    
    def setautoffc(self):
        """자동 FFC 모드 설정 - V4L2에서는 미지원"""
        pass
