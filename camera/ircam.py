"""
IR 열화상 카메라 모듈 (PureThermal/FLIR Lepton)

주요 기능:
- USB를 통한 열화상 카메라 데이터 캡처
- RAW16 온도 데이터를 컬러맵 이미지로 변환
- 온도 기반 화점(hotspot) 탐지
- 런타임 방향 조정 (회전/반전)

센서 스펙:
- 해상도: 160 x 120
- 최대 FPS: 9 (하드웨어 제한)
- 출력: 16bit RAW 온도 데이터 (단위: 0.01 Kelvin)
"""

import cv2
import time
import numpy as np
import logging
import threading

from datetime import datetime
from core.util import dyn_sleep
from core.state import camera_state
from camera.frame_source import FrameSource
from .purethermal.thermalcamera import ThermalCamera


def detect_fire(data, min_val, tau=0.95, thr=20, raw_thr=5, window_size=10, delta_thr=10):
    """
    온도 기반 화점 탐지 알고리즘
    
    IR 센서의 RAW16 온도 데이터를 분석하여 화점(고온점)을 탐지합니다.
    대기 투과율 보정을 적용하여 실제 온도를 추정합니다.
    
    Args:
        data: RAW16 온도 데이터 (단위: 0.01 Kelvin, shape: 160x120 flatten)
        min_val: 화점으로 판정할 최소 온도 (섭씨)
        tau: 대기 투과율 (0~1, 기본값 0.95 = 95% 투과, 실내용)
             - 실내/근거리: 0.95~1.0 (대기 흡수 적음)
             - 야외/장거리: 0.3~0.7 (대기 흡수 큼)
        thr: 보정 후 온도에서 (최고 - 평균) 임계값 (섭씨)
        raw_thr: 보정 전 온도에서 (최고 - 평균) 임계값 (섭씨)
        window_size: 스캔 윈도우 크기 (픽셀)
        delta_thr: hotspot 확장 시 온도 차이 허용 범위 (섭씨)
    
    Returns:
        tuple: (detected: bool, bboxes: list or None, hotspots: list)
               - detected: 화점 탐지 여부
               - bboxes: 탐지된 화점들의 바운딩 박스 [(x, y, w, h), ...]
               - hotspots: 화점 좌표 및 온도 [(x, y, temp_corrected, temp_raw), ...]
    
    알고리즘 흐름:
        1. 온도 변환: RAW16 → Kelvin → 섭씨
        2. 대기 투과율 보정 적용
        3. 윈도우 기반 스캔으로 hotspot 후보 탐색
        4. Hotspot 주변 영역 확장 (비슷한 온도 픽셀 포함)
        5. Contour 추출 및 BBox 생성
    """
    try:
        # ===== 1단계: 온도 변환 =====
        # RAW16 데이터: 0.01 Kelvin 단위 (예: 30000 = 300.00K)
        T_scene_K = data / 100  # Kelvin으로 변환
        
        # 상수 정의
        T_atm_K = 295.15   # 대기 온도 (약 22도C)
        T_0C_K = 273.15    # 절대 영도 기준 (0도C = 273.15K)
        
        # ===== 2단계: 대기 투과율 보정 =====
        # 공식: T_corrected = (T_measured - T_atm) / tau + T_atm
        # 원리: 대기가 적외선을 흡수하므로, 측정된 온도는 실제보다 낮음
        #       tau가 낮을수록 (대기 흡수가 클수록) 보정값이 커짐
        T_corrected_K = (T_scene_K - T_atm_K) / tau + T_atm_K
        
        # 섭씨로 변환하고 2D 배열로 reshape
        temper_raw = (T_scene_K - T_0C_K).reshape(120, 160)      # 보정 전 온도 (섭씨)
        temper = (T_corrected_K - T_0C_K).reshape(120, 160)      # 보정 후 온도 (섭씨)

        # ===== 3단계: 윈도우 기반 Hotspot 탐색 =====
        hotspots = []  # 탐지된 hotspot 리스트: [(x, y, temp, raw_temp), ...]
        h, w = temper.shape  # 120 x 160
        mask = np.zeros((h, w), dtype=np.uint8)  # 화점 마스크

        # 이미지를 window_size x window_size 블록으로 나눠서 스캔
        for y in range(0, h - window_size, window_size):
            for x in range(0, w - window_size, window_size):
                # 현재 윈도우 추출
                window_raw = temper_raw[y:y + window_size, x:x + window_size]
                window = temper[y:y + window_size, x:x + window_size]
                
                # 윈도우 내 통계
                max_temp = np.max(window)           # 보정 후 최고 온도
                mean_temp = np.mean(window)         # 보정 후 평균 온도
                max_temp_raw = np.max(window_raw)   # 보정 전 최고 온도
                mean_temp_raw = np.mean(window_raw) # 보정 전 평균 온도
                
                # ===== Hotspot 판정 조건 (모두 충족해야 함) =====
                # 1) 보정 전: 최고온도가 평균보다 raw_thr 이상 높음
                # 2) 보정 후: 최고온도가 평균보다 thr 이상 높음
                # 3) 보정 후: 최고온도가 min_val 이상
                if (max_temp_raw >= mean_temp_raw + raw_thr and
                    max_temp >= mean_temp + thr and
                    max_temp >= min_val):
                    # 윈도우 내에서 최고 온도 픽셀 위치 찾기
                    max_idx = np.unravel_index(np.argmax(window), window.shape)
                    cx, cy = x + max_idx[1], y + max_idx[0]  # 전체 이미지 좌표로 변환
                    
                    # Hotspot 정보 저장
                    hotspots.append((cx, cy, max_temp, max_temp_raw))
                    mask[cy, cx] = 255  # 마스크에 표시
        
        # Hotspot이 없으면 종료
        if len(hotspots) == 0:
            return False, None, []

        # ===== 4단계: Hotspot 주변 영역 확장 =====
        # 각 hotspot 주변에서 비슷한 온도를 가진 픽셀을 마스크에 추가
        for (hx, hy, h_temp, _) in hotspots:
            # Hotspot 주변 5픽셀 범위 ROI
            y_min, y_max = max(0, hy-5), min(h, hy+5)
            x_min, x_max = max(0, hx-5), min(w, hx+5)
            roi = temper[y_min:y_max, x_min:x_max]
            
            # Hotspot 온도와 delta_thr 이내의 픽셀을 마스크에 추가
            roi_mask = np.abs(roi - h_temp) <= delta_thr
            mask[y_min:y_max, x_min:x_max] |= roi_mask.astype(np.uint8) * 255

        # ===== 5단계: Contour 추출 및 BBox 생성 =====
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
        
            # 각 contour가 실제 hotspot을 포함하는지 확인
            for hx, hy, h_temp, h_temp_raw in hotspots:
                if x <= hx <= x + w and y <= hy <= y + h:
                    # 면적이 크면 온도 조건 추가 검증, 작으면 그냥 포함
                    # h_temp = 보정 온도 기준으로 판정
                    if area > 10 and h_temp >= min_val:
                        bboxes.append((x, y, w, h))
                    elif area <= 10:
                        bboxes.append((x, y, w, h))

        if bboxes:
            return (True, bboxes, hotspots)
        else:
            return (False, None, [])
    
    except Exception:
        # 오류 발생 시 탐지 실패로 처리
        return False, None, []


def draw_bbox(frame, datas):
    """
    화점 탐지 결과를 프레임에 그리기
    
    Args:
        frame: BGR 컬러맵 이미지 (cv2 형식)
        datas: 바운딩 박스 리스트 [(x, y, w, h), ...]
    
    Returns:
        frame: 박스가 그려진 이미지
    
    Note:
        - 빨간색 박스 (BGR: 0, 0, 255)
        - 두께 1픽셀
    """
    for bbox in datas:
        y, x, h, w = bbox  # 주의: detect_fire에서 (x, y, w, h) 순서로 저장됨
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return frame


logger = logging.getLogger(__name__)


class IRCamera(FrameSource):
    """
    IR 열화상 카메라 클래스
    
    PureThermal 보드를 통해 FLIR Lepton 센서 데이터를 캡처하고,
    실시간으로 처리하여 DoubleBuffer에 저장합니다.
    
    출력 데이터:
        1. d_buffer: 컬러맵 이미지 (8bit BGR, 화면 표시용)
        2. d16_buffer: RAW16 온도 데이터 (16bit, 분석/저장용)
    
    Attributes:
        cam: ThermalCamera 인스턴스 (libuvc 기반)
        fire_detection_enabled: 화점 탐지 활성화 여부
        fire_min_temp: 화점 탐지 최소 온도 임계값
        cur_det: 현재 프레임의 화점 탐지 결과
    """
    
    def __init__(self, cfg, d_buffer, d16_buffer, cam_impl=None):
        super().__init__("IRCamera")
        """
        IR 카메라 초기화
        
        Args:
            cfg: 설정 딕셔너리
                - FPS: 목표 프레임레이트 (최대 9)
                - RES: 출력 해상도 [width, height]
                - SLEEP: 루프 슬립 시간 (초)
                - TAU: 대기 투과율 (기본: 0.95, 실내용)
                - FIRE_DETECTION: 화점 탐지 활성화 (기본: True)
                - FIRE_MIN_TEMP: 화점 최소 온도 (기본: 80도C)
                - FIRE_THR: 보정 온도 임계값 (기본: 20)
                - FIRE_RAW_THR: raw 온도 임계값 (기본: 5)
            d_buffer: 컬러맵 이미지 출력 버퍼 (DoubleBuffer)
            d16_buffer: RAW16 데이터 출력 버퍼 (DoubleBuffer)
        """
        # USB를 통해 PureThermal 카메라 초기화 (libuvc 사용)
        self.cam = cam_impl or ThermalCamera()
        self.thread = None
        self.stop_event = threading.Event()
        self.last_ts = None  # 마지막 타임스탬프 (중복 프레임 방지용)

        # 설정 로드
        self.fps = cfg['FPS']      # 목표 FPS (하드웨어 한계: 9)
        self.size = cfg['RES']     # 출력 해상도 [width, height]
        self.sleep = cfg['SLEEP']  # 루프당 최소 슬립 시간

        # 출력 버퍼 참조 저장
        self.d_buffer = d_buffer      # 컬러맵 이미지 버퍼
        self.d16_buffer = d16_buffer  # RAW16 온도 데이터 버퍼
        
        # 화점 탐지 설정
        self.fire_detection_enabled = cfg.get('FIRE_DETECTION', True)
        self.fire_min_temp = cfg.get('FIRE_MIN_TEMP', 80)  # 최소 온도 (섭씨)
        self.tau = cfg.get('TAU', 0.95)  # 대기 투과율 (실내: 0.95)
        self.fire_thr = cfg.get('FIRE_THR', 20)  # 보정 온도 임계값
        self.fire_raw_thr = cfg.get('FIRE_RAW_THR', 5)  # raw 온도 임계값
        self.cur_det = False  # 현재 프레임 탐지 결과
        self.hotspots = []    # 현재 프레임의 hotspot 리스트
        
        # 최고 온도 정보 (매 프레임 업데이트)
        self.max_temp_info = None

    def update_fire_params(self, fire_detection=None, min_temp=None, thr=None, raw_thr=None, tau=None):
        """런타임에 화점 탐지 파라미터를 업데이트"""
        if fire_detection is not None:
            self.fire_detection_enabled = bool(fire_detection)
        if min_temp is not None:
            self.fire_min_temp = float(min_temp)
        if thr is not None:
            self.fire_thr = float(thr)
        if raw_thr is not None:
            self.fire_raw_thr = float(raw_thr)
        if tau is not None:
            self.tau = float(tau)


    def _get_max_temp_info(self, raw16, tau=None):
        """
        RAW16 데이터에서 최고 온도 지점 정보 추출
        
        Args:
            raw16: 16bit 온도 데이터 (0.01 Kelvin 단위)
            tau: 대기 투과율 (보정 계수)
        
        Returns:
            dict: {
                'x': int,              # 최고온도 x좌표 (픽셀)
                'y': int,              # 최고온도 y좌표 (픽셀)
                'temp_raw': float,     # 보정 전 온도 (섭씨)
                'temp_corrected': float # 보정 후 온도 (섭씨)
            }
        """
        try:
            # config에서 tau 사용 (인자로 전달되지 않은 경우)
            if tau is None:
                tau = self.tau
            
            # 온도 변환 상수
            T_atm_K = 295.15   # 대기 온도 (약 22도C)
            T_0C_K = 273.15    # 0도C in Kelvin
            
            # RAW16 → Kelvin → 섭씨
            T_scene_K = raw16.astype(np.float32) / 100.0
            
            # 대기 투과율 보정
            T_corrected_K = (T_scene_K - T_atm_K) / tau + T_atm_K
            
            # 섭씨 변환
            temp_raw = T_scene_K - T_0C_K
            temp_corrected = T_corrected_K - T_0C_K
            
            # 최고/최저 온도 위치 찾기 (보정 후 온도 기준)
            max_idx = np.unravel_index(np.argmax(temp_corrected), temp_corrected.shape)
            min_idx = np.unravel_index(np.argmin(temp_corrected), temp_corrected.shape)
            y, x = int(max_idx[0]), int(max_idx[1])
            y_min, x_min = int(min_idx[0]), int(min_idx[1])
            
            return {
                'x': x,
                'y': y,
                'min_temp': round(float(temp_corrected[y_min, x_min]), 2),
                'temp_raw': round(float(temp_raw[y, x]), 2),
                'temp_corrected': round(float(temp_corrected[y, x]), 2)
            }
        except Exception:
            return None


    def capture(self):
        """
        단일 프레임 캡처 및 처리
        
        처리 순서:
            1. RAW16 데이터 캡처 (libuvc)
            2. 정규화 및 컬러맵 적용 (PLASMA)
            3. 방향 조정 (회전/반전) - 런타임 키보드 제어
            4. 화점 탐지 (옵션)
            5. 탐지 결과 시각화 (박스 그리기)
            6. 출력 해상도로 리사이즈
        
        Returns:
            tuple: (raw16, frame, timestamp, max_temp_info, hotspots)
                - raw16: 16bit 온도 데이터 (분석/저장용)
                - frame: 8bit BGR 컬러맵 이미지 (화면 표시용)
                - timestamp: 캡처 시각 문자열 (YYMMDDHHMMSSff)
                - max_temp_info: 최고 온도 정보 dict
                - hotspots: 화점 리스트 [(x, y, temp, raw_temp), ...]
            
            캡처 실패 시: (None, None, None, None, [])
        """
        # ===== 1. RAW16 데이터 캡처 =====
        raw16 = self.cam.capture()
        if raw16 is None:
            return None, None, None, None, []

        # 타임스탬프 생성 (밀리초 2자리까지)
        ts = datetime.now().strftime("%y%m%d%H%M%S%f")[:-4]
        
        # ===== 2. 정규화 및 컬러맵 적용 =====
        # RAW16 → 0~65535 범위로 정규화 (대비 향상)
        norm = cv2.normalize(raw16, None, 0, 65535, cv2.NORM_MINMAX)
        # 16bit → 8bit 변환 (상위 8비트 사용)
        gray8 = (norm >> 8).astype(np.uint8)
        # 그레이스케일 → 컬러맵 (PLASMA: 보라-노랑 계열, 열화상에 적합)
        frame = cv2.applyColorMap(gray8, cv2.COLORMAP_PLASMA)
        
        # ===== 3. 방향 조정 (키보드로 실시간 제어) =====
        # camera_state는 싱글톤으로 app.py에서 키보드 입력으로 변경됨
        
        # 회전 적용 (0, 90, 180, 270도)
        rotate = camera_state.rotate_ir
        if rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            raw16 = cv2.rotate(raw16, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            raw16 = cv2.rotate(raw16, cv2.ROTATE_180)
        elif rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            raw16 = cv2.rotate(raw16, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # 좌우반전 (horizontal flip)
        if camera_state.flip_h_ir:
            frame = cv2.flip(frame, 1)  # 1 = 좌우반전
            raw16 = cv2.flip(raw16, 1)
        
        # 상하반전 (vertical flip)
        if camera_state.flip_v_ir:
            frame = cv2.flip(frame, 0)  # 0 = 상하반전
            raw16 = cv2.flip(raw16, 0)
        
        # ===== 4. 최고 온도 지점 추출 =====
        self.max_temp_info = self._get_max_temp_info(raw16)
        
        # ===== 5. 화점 탐지 =====
        # 방향 조정이 완료된 raw16으로 탐지 수행
        # (좌표가 최종 출력 이미지와 일치하도록)
        datas = None
        self.hotspots = []
        if self.fire_detection_enabled:
            self.cur_det, datas, self.hotspots = detect_fire(
                raw16, self.fire_min_temp, 
                tau=self.tau, thr=self.fire_thr, raw_thr=self.fire_raw_thr
            )
            
            # 디버깅용 로그: 임계값, 최대 온도, 탐지 여부
            max_temp = None
            if self.max_temp_info and 'temp_corrected' in self.max_temp_info:
                max_temp = self.max_temp_info['temp_corrected']
            exceeded = max_temp is not None and max_temp >= self.fire_min_temp
            bbox_count = len(datas) if datas else 0
            logger.debug(
                "[IR DET] thr=%.1fC max=%s exceeded=%s fire_detected=%s bbox=%d",
                self.fire_min_temp,
                max_temp if max_temp is not None else "N/A",
                "YES" if exceeded else "NO",
                "YES" if self.cur_det else "NO",
                bbox_count,
            )
        
        # ===== 6. 탐지 결과 시각화 =====
        if self.fire_detection_enabled and self.cur_det and datas is not None:
            frame = draw_bbox(frame, datas)
        
        # ===== 7. 출력 해상도로 리사이즈 =====
        # config의 RES 설정에 맞춰 리사이즈
        frame = cv2.resize(frame, (self.size[0], self.size[1]), interpolation=cv2.INTER_AREA)
        
        # hotspots 정보 포함하여 반환
        return raw16, frame, ts, self.max_temp_info, self.hotspots


    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self.thread

    def _loop(self):
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                s_time = time.time()  # 루프 시작 시간

                # 프레임 캡처
                raw16, frame, ts, max_temp_info, hotspots = self.capture()
                
                # 캡처 실패 시 스킵
                if frame is None:
                    dyn_sleep(s_time, self.sleep)
                    continue

                # 중복 프레임 스킵 (같은 타임스탬프면 새 프레임 아님)
                if self.last_ts == ts:
                    dyn_sleep(s_time, self.sleep)
                    continue

                # 버퍼에 데이터 저장 (tuple: (data, timestamp, max_temp_info, hotspots))
                self.d16_buffer.write((raw16, ts, max_temp_info, hotspots))  # RAW16 + 최고온도 + hotspots
                self.d_buffer.write((frame, ts, max_temp_info, hotspots))    # 컬러맵 + 최고온도 + hotspots
                self.last_ts = ts

                # 프레임 카운트 및 로그
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info("[IRCam] Captured %d frames", frame_count)

                # FPS 유지를 위한 동적 슬립
                dyn_sleep(s_time, self.sleep)
                
            except Exception as e:
                import traceback
                logger.exception("[IRCam] Error in capture loop: %s", e)
                time.sleep(0.1)  # 에러 시 잠시 대기
    

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        try:
            if hasattr(self, "cam") and self.cam:
                self.cam.cleanup()
        except Exception:
            pass
