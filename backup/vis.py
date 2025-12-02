# visualize.py
import os
import cv2
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from configs.get_cfg import get_cfg

# ===== Config 로드 =====
cfg = get_cfg()
TARGET_W, TARGET_H = cfg['TARGET_RES']  # RGB 출력 크기
IR_W, IR_H = cfg['CAMERA']['IR']['RES']  # IR 원본 크기 (160, 120)

# ===== 저장 경로/상태 =====
VISIBLE_DIR = "save/visible"
LWIR_DIR    = "save/lwir"
os.makedirs(VISIBLE_DIR, exist_ok=True)
os.makedirs(LWIR_DIR, exist_ok=True)

saving = False  # 's'로 시작, 'e'로 중지

# ===== 비동기 저장용 ThreadPoolExecutor =====
save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SaveWorker")

# ===== 유틸 =====
def _save_single_frame(file_path, frame, is_16bit=False):
    """단일 프레임을 파일로 저장 (워커 스레드에서 실행)"""
    try:
        if is_16bit:
            ok = cv2.imwrite(file_path, frame.astype(np.uint16))
        else:
            ok = cv2.imwrite(file_path, frame)
        
        if ok:
            print(f"[SAVED] {'IR' if is_16bit else 'RGB'} → {file_path}")
        else:
            print(f"[ERROR] Save failed: {file_path}")
    except Exception as e:
        print(f"[ERROR] Exception during save: {file_path}, {e}")

def save_frames(rgb_item, ir16_item):
    """원본 RGB와 16bit IR 프레임을 비동기로 저장"""
    if rgb_item is None or ir16_item is None:
        print("No frames to save.")
        return
    rgb_frame, ts_rgb = rgb_item
    ir16_frame, ts_ir = ir16_item

    rgb_file = os.path.join(VISIBLE_DIR, f"{ts_rgb}.jpg")
    ir_file  = os.path.join(LWIR_DIR,    f"{ts_ir}.tiff")

    # 비동기 저장 (메인 스레드 블로킹 방지)
    save_executor.submit(_save_single_frame, rgb_file, rgb_frame, False)
    save_executor.submit(_save_single_frame, ir_file, ir16_frame, True)

def put_label(frame, text, pos,
              font=cv2.FONT_HERSHEY_DUPLEX, scale=0.5,
              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = pos
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 4), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), font, scale, text_color, 1, cv2.LINE_AA)

# ===== 레이아웃/스타일 =====
GAP = 8  # 패널 사이 간격
BG_COLOR = (32, 32, 32)

def _fit_to_target(img, w, h, interp=cv2.INTER_AREA):
    """이미지가 (w,h)가 아니면 리사이즈, 맞으면 그대로 반환."""
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=interp)

# ===== 메인 시각화 =====
def visualize(d_rgb, d_ir, d16_ir, d_rgb_det):
    """
    화면 표시: 2패널
      - 좌: LWIR (160x120 원본을 TARGET_RES 크기 캔버스 중앙 배치)
      - 우: Visible FRONT (RGB 감지, TARGET_RES 크기)
    저장: 's' 시작 / 'e' 중지 (원본 RGB + 16bit IR)
    종료: 'q'
    """
    global saving

    win_name = "LK ROBOTICS Inc."
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # 2패널: LWIR(패딩 추가) + RGB Detection (모두 TARGET_RES)
    canvas_w = TARGET_W * 2 + GAP
    canvas_h = TARGET_H

    while True:
        # --- 표시용 프레임 획득 ---
        ir_item      = d_ir.read()       # LWIR 원본
        rgb_det_item = d_rgb_det.read()  # RGB 감지 결과

        # --- LWIR 패널 (원본을 TARGET_RES 캔버스 중앙 배치) ---
        ir_src = ir_item[0] if ir_item else None
        ir_disp = _fit_to_target(ir_src, IR_W, IR_H, cv2.INTER_NEAREST)
        
        # IR을 TARGET_RES 크기 캔버스 중앙에 배치 (패딩 추가)
        ir_padded = np.full((TARGET_H, TARGET_W, 3), BG_COLOR, dtype=np.uint8)
        y_offset = (TARGET_H - IR_H) // 2
        x_offset = (TARGET_W - IR_W) // 2
        ir_padded[y_offset:y_offset + IR_H, x_offset:x_offset + IR_W] = ir_disp

        # --- RGB 패널 (감지 결과, 없으면 원본 프레임 그대로 표시됨) ---
        rgb_src = rgb_det_item[0] if rgb_det_item else None
        rgb_disp = _fit_to_target(rgb_src, TARGET_W, TARGET_H, cv2.INTER_AREA)

        # --- 캔버스 구성 (좌: LWIR(패딩), 우: RGB) ---
        combined = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

        x_lwir = 0
        x_rgb  = TARGET_W + GAP

        combined[0:TARGET_H, x_lwir:x_lwir + TARGET_W] = ir_padded
        combined[0:TARGET_H, x_rgb:x_rgb + TARGET_W]   = rgb_disp

        # --- 라벨링 ---
        put_label(combined, "LWIR",          (x_lwir + 10, 25))
        put_label(combined, "Visible", (x_rgb  + 10, 25))

        # --- 표시 ---
        cv2.imshow(win_name, combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = True
            print("Saving started...")
        elif key == ord('e'):
            saving = False
            print("Saving stopped.")

        # --- 저장: 원본 RGB + 16bit IR (saving=True일 때만 버퍼 읽기) ---
        if saving:
            rgb_item_raw = d_rgb.read()
            ir16_item_raw = d16_ir.read()
            
            if rgb_item_raw is not None and ir16_item_raw is not None:
                save_frames(rgb_item_raw, ir16_item_raw)
            else:
                if rgb_item_raw is None:
                    print("[DEBUG] Saving skipped: d_rgb is None")
                if ir16_item_raw is None:
                    print("[DEBUG] Saving skipped: d16_ir is None")

        time.sleep(0.01)


    cv2.destroyWindow(win_name)