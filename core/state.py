# core/state.py
import threading


class CameraState:
    """
    카메라 방향 조정 상태를 관리하는 싱글톤 클래스
    스레드 안전하게 flip/rotate 상태를 공유함
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_state()
        return cls._instance
    
    def _init_state(self):
        self._state_lock = threading.Lock()
        # IR 카메라 상태
        self._flip_h_ir = False      # 좌우반전
        self._flip_v_ir = False      # 상하반전
        self._rotate_ir = 0          # 회전 (0, 90, 180, 270)
        # RGB 카메라 상태
        self._flip_h_rgb = False     # 좌우반전
        self._flip_v_rgb = False     # 상하반전
        self._rotate_rgb = 0         # 회전 (0, 90, 180, 270)
    
    # === IR 카메라 ===
    @property
    def flip_h_ir(self):
        with self._state_lock:
            return self._flip_h_ir
    
    @property
    def flip_v_ir(self):
        with self._state_lock:
            return self._flip_v_ir
    
    @property
    def rotate_ir(self):
        with self._state_lock:
            return self._rotate_ir
    
    def toggle_flip_h_ir(self):
        with self._state_lock:
            self._flip_h_ir = not self._flip_h_ir
            return self._flip_h_ir
    
    def toggle_flip_v_ir(self):
        with self._state_lock:
            self._flip_v_ir = not self._flip_v_ir
            return self._flip_v_ir
    
    def rotate_ir_cw(self):
        """IR 카메라 시계방향 90도 회전"""
        with self._state_lock:
            self._rotate_ir = (self._rotate_ir + 90) % 360
            return self._rotate_ir
    
    # === RGB 카메라 ===
    @property
    def flip_h_rgb(self):
        with self._state_lock:
            return self._flip_h_rgb
    
    @property
    def flip_v_rgb(self):
        with self._state_lock:
            return self._flip_v_rgb
    
    @property
    def rotate_rgb(self):
        with self._state_lock:
            return self._rotate_rgb
    
    def toggle_flip_h_rgb(self):
        with self._state_lock:
            self._flip_h_rgb = not self._flip_h_rgb
            return self._flip_h_rgb
    
    def toggle_flip_v_rgb(self):
        with self._state_lock:
            self._flip_v_rgb = not self._flip_v_rgb
            return self._flip_v_rgb
    
    def rotate_rgb_cw(self):
        """RGB 카메라 시계방향 90도 회전"""
        with self._state_lock:
            self._rotate_rgb = (self._rotate_rgb + 90) % 360
            return self._rotate_rgb
    
    # === 공통 ===
    def toggle_flip_h_both(self):
        with self._state_lock:
            new_state = not self._flip_h_ir
            self._flip_h_ir = new_state
            self._flip_h_rgb = new_state
            return new_state
    
    def toggle_flip_v_both(self):
        with self._state_lock:
            new_state = not self._flip_v_ir
            self._flip_v_ir = new_state
            self._flip_v_rgb = new_state
            return new_state
    
    def get_status(self):
        with self._state_lock:
            return {
                'ir': {
                    'flip_h': self._flip_h_ir,
                    'flip_v': self._flip_v_ir,
                    'rotate': self._rotate_ir
                },
                'rgb': {
                    'flip_h': self._flip_h_rgb,
                    'flip_v': self._flip_v_rgb,
                    'rotate': self._rotate_rgb
                }
            }


# 전역 인스턴스
camera_state = CameraState()
