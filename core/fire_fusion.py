"""
EO-IR 화재 감지 융합 모듈

IR 열화상 카메라의 hotspot 감지 결과와 EO(RGB) 카메라의 화염 감지 결과를
융합하여 최종 화재 판정을 수행합니다.

Phase 1: IR 게이트키퍼 방식
- IR hotspot이 없으면 화재 아님 (EO 결과 무시)
- IR + EO 모두 감지 시 높은 신뢰도
- IR만 감지 시 중간 신뢰도
"""

import logging
from .coord_mapper import CoordMapper, point_in_bbox


# 신뢰도 상수
CONFIDENCE_HIGH = 0.95      # IR + EO 매칭
CONFIDENCE_MEDIUM = 0.70    # IR만 감지
CONFIDENCE_LOW = 0.30       # EO만 감지 (게이트키핑됨)
CONFIDENCE_NONE = 0.0       # 미감지

# 판정 결과 상수
FIRE_CONFIRMED = 'CONFIRMED'      # IR + EO 확정 화재
FIRE_IR_ONLY = 'IR_ONLY'          # IR만 감지 (의심)
FIRE_FILTERED = 'FILTERED'        # EO만 감지 (게이트키핑됨)
NO_FIRE = 'NO_FIRE'               # 화재 아님

# bbox 색상 (BGR)
COLOR_CONFIRMED = (0, 0, 255)     # 빨강: 확정 화재
COLOR_IR_ONLY = (0, 165, 255)     # 주황: IR만 감지
COLOR_FILTERED = (0, 255, 255)    # 노랑: EO만 감지 (필터링됨)
COLOR_NO_FIRE = (128, 128, 128)   # 회색: 미감지


logger = logging.getLogger(__name__)


class FireFusion:
    """
    EO-IR 화재 감지 융합 클래스
    
    Phase 1 로직:
    - IR hotspot이 없으면 무조건 NOT FIRE
    - IR hotspot이 EO bbox 안에 있으면 CONFIRMED
    - IR hotspot만 있으면 IR_ONLY
    """
    
    def __init__(self, ir_size=(160, 120), rgb_size=(960, 540),
                 offset_x=0, offset_y=0, scale=None):
        """
        융합 모듈 초기화
        
        Args:
            ir_size: IR 이미지 크기
            rgb_size: RGB 이미지 크기
            offset_x, offset_y: 캘리브레이션 오프셋
            scale: 스케일 팩터
        """
        self.coord_mapper = CoordMapper(ir_size, rgb_size, offset_x, offset_y, scale)
        
        # 마지막 융합 결과
        self.last_result = None
    
    def fuse(self, ir_hotspots, eo_fire_bboxes):
        """
        IR hotspot과 EO fire bbox를 융합하여 최종 화재 판정
        
        Args:
            ir_hotspots: IR 화점 리스트 [(x, y, temp_corrected, temp_raw), ...]
            eo_fire_bboxes: EO 화염 bbox 리스트 [(x, y, w, h, confidence), ...]
            
        Returns:
            dict: {
                'fire_detected': bool,
                'confidence': float,
                'status': str,
                'details': list of detection details,
                'eo_annotations': list of (bbox, color, label) for drawing
            }
        """
        details = []
        eo_annotations = []  # EO 프레임에 그릴 bbox 정보
        
        # ===== 게이트키퍼: IR hotspot 체크 =====
        if not ir_hotspots or len(ir_hotspots) == 0:
            # IR 감지 없음 → EO 결과 무시
            for eo_bbox in (eo_fire_bboxes or []):
                # EO bbox를 필터링된 것으로 표시 (노란색)
                bbox = eo_bbox[:4] if len(eo_bbox) >= 4 else eo_bbox
                eo_conf = eo_bbox[4] if len(eo_bbox) > 4 else 0.0
                eo_annotations.append({
                    'bbox': bbox,
                    'color': COLOR_FILTERED,
                    'label': f'FILTERED ({eo_conf:.0%})',
                    'status': FIRE_FILTERED
                })
            
            self.last_result = {
                'fire_detected': False,
                'confidence': CONFIDENCE_NONE,
                'status': NO_FIRE,
                'reason': 'NO_IR_HOTSPOT',
                'details': details,
                'eo_annotations': eo_annotations
            }
            return self.last_result
        
        # ===== IR hotspot 있음: EO와 매칭 확인 =====
        confirmed_fires = []
        ir_only_fires = []
        matched_eo_indices = set()
        
        for hotspot in ir_hotspots:
            ir_x, ir_y = hotspot[0], hotspot[1]
            temp = hotspot[2] if len(hotspot) > 2 else 0
            
            # IR 좌표를 RGB 좌표로 변환
            rgb_x, rgb_y = self.coord_mapper.ir_to_rgb(ir_x, ir_y)
            
            # EO bbox와 매칭 확인
            matched = False
            for i, eo_bbox in enumerate(eo_fire_bboxes or []):
                bbox = eo_bbox[:4] if len(eo_bbox) >= 4 else eo_bbox
                eo_conf = eo_bbox[4] if len(eo_bbox) > 4 else 0.0
                
                if point_in_bbox(rgb_x, rgb_y, bbox):
                    # IR + EO 매칭 → 확정 화재
                    confirmed_fires.append({
                        'ir_pos': (ir_x, ir_y),
                        'rgb_pos': (rgb_x, rgb_y),
                        'temp': temp,
                        'eo_bbox': bbox,
                        'eo_conf': eo_conf,
                        'confidence': CONFIDENCE_HIGH,
                        'status': FIRE_CONFIRMED
                    })
                    matched_eo_indices.add(i)
                    matched = True
                    break
            
            if not matched:
                # IR만 감지
                ir_only_fires.append({
                    'ir_pos': (ir_x, ir_y),
                    'rgb_pos': (rgb_x, rgb_y),
                    'temp': temp,
                    'confidence': CONFIDENCE_MEDIUM,
                    'status': FIRE_IR_ONLY
                })
        
        # ===== Phase1 fallback: 좌표 매핑이 없어도 IR이 임계 초과하면 EO bbox 전부 확정 처리 =====
        fallback_confirmed = False
        if ir_hotspots and not confirmed_fires and (eo_fire_bboxes or []):
            fallback_confirmed = True
            # 가장 뜨거운 hotspot 사용
            ref_hotspot = max(ir_hotspots, key=lambda h: h[2] if len(h) > 2 else 0)
            ref_temp = ref_hotspot[2] if len(ref_hotspot) > 2 else 0
            rgb_ref = self.coord_mapper.ir_to_rgb(ref_hotspot[0], ref_hotspot[1])
            for i, eo_bbox in enumerate(eo_fire_bboxes):
                bbox = eo_bbox[:4] if len(eo_bbox) >= 4 else eo_bbox
                eo_conf = eo_bbox[4] if len(eo_bbox) > 4 else 0.0
                confirmed_fires.append({
                    'ir_pos': (ref_hotspot[0], ref_hotspot[1]),
                    'rgb_pos': rgb_ref,
                    'temp': ref_temp,
                    'eo_bbox': bbox,
                    'eo_conf': eo_conf,
                    'confidence': CONFIDENCE_HIGH,
                    'status': FIRE_CONFIRMED
                })
                matched_eo_indices.add(i)
        
        # EO annotations 생성
        for i, eo_bbox in enumerate(eo_fire_bboxes or []):
            bbox = eo_bbox[:4] if len(eo_bbox) >= 4 else eo_bbox
            eo_conf = eo_bbox[4] if len(eo_bbox) > 4 else 0.0
            
            if i in matched_eo_indices:
                # 확정 화재 (빨간색)
                # 매칭된 온도 찾기
                matched_temp = None
                for cf in confirmed_fires:
                    if cf['eo_bbox'] == bbox:
                        matched_temp = cf['temp']
                        break
                temp_str = f'{matched_temp:.0f}C' if matched_temp else ''
                eo_annotations.append({
                    'bbox': bbox,
                    'color': COLOR_CONFIRMED,
                    'label': f'FIRE {temp_str} ({eo_conf:.0%})',
                    'status': FIRE_CONFIRMED
                })
            else:
                # 필터링됨 (노란색)
                eo_annotations.append({
                    'bbox': bbox,
                    'color': COLOR_FILTERED,
                    'label': f'FILTERED ({eo_conf:.0%})',
                    'status': FIRE_FILTERED
                })
        
        # IR만 감지된 위치에 대한 annotation (캘리브레이션 전까지 비활성화)
        # TODO: Phase 2에서 키보드 캘리브레이션 후 활성화
        # for ir_fire in ir_only_fires:
        #     rgb_x, rgb_y = ir_fire['rgb_pos']
        #     temp = ir_fire['temp']
        #     marker_size = 30
        #     eo_annotations.append({
        #         'bbox': (rgb_x - marker_size/2, rgb_y - marker_size/2, marker_size, marker_size),
        #         'color': COLOR_IR_ONLY,
        #         'label': f'IR {temp:.0f}C',
        #         'status': FIRE_IR_ONLY
        #     })
        
        # 최종 결과 결정
        details = confirmed_fires + ir_only_fires
        
        if confirmed_fires:
            max_conf = max(f['confidence'] for f in confirmed_fires)
            status = FIRE_CONFIRMED
        elif ir_only_fires:
            max_conf = max(f['confidence'] for f in ir_only_fires)
            status = FIRE_IR_ONLY
        else:
            max_conf = CONFIDENCE_NONE
            status = NO_FIRE
        
        self.last_result = {
            'fire_detected': len(details) > 0,
            'confidence': max_conf,
            'status': status,
            'confirmed_count': len(confirmed_fires),
            'ir_only_count': len(ir_only_fires),
            'details': details,
            'eo_annotations': eo_annotations
        }
        
        filtered_count = max(0, len(eo_fire_bboxes or []) - len(matched_eo_indices))
        logger.debug(
            "[FUSION] fire_detected=%s status=%s confirmed=%d ir_only=%d filtered=%d confidence=%.2f",
            "YES" if self.last_result['fire_detected'] else "NO",
            status,
            len(confirmed_fires),
            len(ir_only_fires),
            filtered_count,
            max_conf,
        )
        
        return self.last_result
    
    def adjust_offset(self, dx, dy):
        """좌표 오프셋 조정"""
        self.coord_mapper.adjust_offset(dx, dy)
    
    def adjust_scale(self, ds):
        """스케일 조정"""
        self.coord_mapper.adjust_scale(ds)
    
    def get_calibration(self):
        """현재 캘리브레이션 파라미터 반환"""
        return self.coord_mapper.get_params()


def draw_fire_annotations(frame, annotations):
    """
    화재 감지 결과를 프레임에 그리기
    
    Args:
        frame: BGR 이미지 (numpy array)
        annotations: eo_annotations 리스트
        
    Returns:
        frame: annotation이 그려진 이미지
    """
    import cv2
    
    for ann in annotations:
        bbox = ann['bbox']
        color = ann['color']
        label = ann['label']
        status = ann['status']
        
        x, y, w, h = [int(v) for v in bbox]
        
        # bbox 그리기
        thickness = 3 if status == FIRE_CONFIRMED else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # 라벨 그리기
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_size, _ = cv2.getTextSize(label, font, font_scale, 2)
        
        # 라벨 배경
        cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                     (x + label_size[0] + 10, y), color, -1)
        
        # 라벨 텍스트
        cv2.putText(frame, label, (x + 5, y - 5), font, font_scale, 
                   (255, 255, 255), 2)
    
    return frame
