"""
IR-RGB 좌표 변환 모듈

IR 카메라 (160x120)와 RGB 카메라 (960x540) 간의 좌표 변환을 수행합니다.
"""


class CoordMapper:
    """
    IR-RGB 좌표 매핑 클래스
    
    비율 유지 스케일링 + 중심 정렬 방식을 사용합니다.
    """
    
    def __init__(self, ir_size=(160, 120), rgb_size=(960, 540), 
                 offset_x=0, offset_y=0, scale=None):
        """
        좌표 매퍼 초기화
        
        Args:
            ir_size: IR 이미지 크기 (width, height)
            rgb_size: RGB 이미지 크기 (width, height)
            offset_x: X축 오프셋 (캘리브레이션용)
            offset_y: Y축 오프셋 (캘리브레이션용)
            scale: 스케일 팩터 (None이면 자동 계산)
        """
        self.ir_w, self.ir_h = ir_size
        self.rgb_w, self.rgb_h = rgb_size
        
        # 스케일 계산 (비율 유지, 작은 쪽 기준)
        if scale is None:
            self.scale = min(self.rgb_w / self.ir_w, self.rgb_h / self.ir_h)
        else:
            self.scale = scale
        
        # 중심 정렬을 위한 기본 오프셋
        self.base_offset_x = (self.rgb_w - self.ir_w * self.scale) / 2
        self.base_offset_y = (self.rgb_h - self.ir_h * self.scale) / 2
        
        # 캘리브레이션 오프셋
        self.offset_x = offset_x
        self.offset_y = offset_y
    
    def ir_to_rgb(self, ir_x, ir_y):
        """
        IR 좌표를 RGB 좌표로 변환
        
        Args:
            ir_x: IR 이미지에서의 X 좌표
            ir_y: IR 이미지에서의 Y 좌표
            
        Returns:
            tuple: (rgb_x, rgb_y)
        """
        rgb_x = ir_x * self.scale + self.base_offset_x + self.offset_x
        rgb_y = ir_y * self.scale + self.base_offset_y + self.offset_y
        return rgb_x, rgb_y
    
    def rgb_to_ir(self, rgb_x, rgb_y):
        """
        RGB 좌표를 IR 좌표로 변환
        
        Args:
            rgb_x: RGB 이미지에서의 X 좌표
            rgb_y: RGB 이미지에서의 Y 좌표
            
        Returns:
            tuple: (ir_x, ir_y)
        """
        ir_x = (rgb_x - self.base_offset_x - self.offset_x) / self.scale
        ir_y = (rgb_y - self.base_offset_y - self.offset_y) / self.scale
        return ir_x, ir_y
    
    def ir_bbox_to_rgb(self, ir_bbox):
        """
        IR bbox를 RGB bbox로 변환
        
        Args:
            ir_bbox: (x, y, w, h) IR 좌표계
            
        Returns:
            tuple: (x, y, w, h) RGB 좌표계
        """
        x, y, w, h = ir_bbox
        rgb_x, rgb_y = self.ir_to_rgb(x, y)
        rgb_w = w * self.scale
        rgb_h = h * self.scale
        return (rgb_x, rgb_y, rgb_w, rgb_h)
    
    def adjust_offset(self, dx, dy):
        """
        오프셋 조정 (실시간 캘리브레이션용)
        
        Args:
            dx: X축 오프셋 변화량
            dy: Y축 오프셋 변화량
        """
        self.offset_x += dx
        self.offset_y += dy
    
    def adjust_scale(self, ds):
        """
        스케일 조정 (실시간 캘리브레이션용)
        
        Args:
            ds: 스케일 변화량
        """
        self.scale = max(0.1, self.scale + ds)
        # 오프셋 재계산
        self.base_offset_x = (self.rgb_w - self.ir_w * self.scale) / 2
        self.base_offset_y = (self.rgb_h - self.ir_h * self.scale) / 2
    
    def get_params(self):
        """
        현재 파라미터 반환 (저장용)
        
        Returns:
            dict: {offset_x, offset_y, scale}
        """
        return {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'scale': self.scale
        }
    
    def __repr__(self):
        return (f"CoordMapper(scale={self.scale:.2f}, "
                f"offset=({self.offset_x:.1f}, {self.offset_y:.1f}))")


def point_in_bbox(x, y, bbox):
    """
    점이 bbox 안에 있는지 확인
    
    Args:
        x, y: 점 좌표
        bbox: (bx, by, bw, bh)
        
    Returns:
        bool: bbox 내부에 있으면 True
    """
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def bbox_iou(bbox1, bbox2):
    """
    두 bbox의 IoU (Intersection over Union) 계산
    
    Args:
        bbox1, bbox2: (x, y, w, h)
        
    Returns:
        float: IoU 값 (0~1)
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 교집합 영역
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # 합집합 영역
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

