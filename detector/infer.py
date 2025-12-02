#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 (TFLite) 배치 추론 - 폴더 이미지(visible) → 결과 저장(save)
- CPU 추론(기본)
- tflite_runtime 우선, 없으면 tf.lite.Interpreter 사용
- YOLOv8 TFLite 출력 (B,N,C)/(B,C,N) 자동 감지
- INT8/UINT8/FP 입력/출력 자동 양자화 처리
- NMS: NumPy 구현
"""

import os, time, glob
import cv2
import numpy as np
import yaml

# ========= 상단 설정 =========
# (1) 모델/라벨
MODEL_PATH    = "best_int8.tflite"          # 실험할 TFLite 모델 경로
METADATA_YAML = "./datasets/custom.yaml"    # {'names': [...]}

# (2) 가속기/스레드
USE_NPU       = False                       # 여기서는 CPU만 사용
DELEGATE_LIB  = "/usr/lib/libvx_delegate.so"
CPU_THREADS   = max(1, os.cpu_count() or 4)

# (3) 입력/출력 폴더
VISIBLE_DIR   = "visible"                   # 입력 이미지 폴더
SAVE_DIR      = "save"                      # 결과 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

# (4) 후처리
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45
MAX_DETS      = 300

# ========= TFLite 백엔드 선택 =========
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except Exception:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_BACKEND = "tensorflow"

# ========= 유틸 =========
def load_names(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            obj = yaml.safe_load(f)
        names = obj.get("names", None)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        if not isinstance(names, (list, tuple)):
            raise ValueError
        return names
    except Exception:
        return [f"id{i}" for i in range(1000)]

def make_interpreter(model_path, use_npu, delegate_lib, cpu_threads):
    delegates = None
    # 여기서는 USE_NPU=False 이므로 delegate 사용 안 함
    if use_npu and delegate_lib and os.path.exists(delegate_lib) and TFLITE_BACKEND == "tflite_runtime":
        try:
            delegates = [tflite.load_delegate(delegate_lib)]
            print(f"[INFO] VX delegate enabled: {delegate_lib}")
        except Exception as e:
            print(f"[WARN] Delegate load failed → CPU fallback: {e}")
            delegates = None
    itp = tflite.Interpreter(model_path=model_path,
                             experimental_delegates=delegates,
                             num_threads=cpu_threads)
    itp.allocate_tensors()
    print(f"[TFLite] backend={TFLITE_BACKEND}, threads={cpu_threads}, NPU={'ON' if delegates else 'OFF'}")
    return itp, bool(delegates)

def letterbox(img, new_shape, color=(114,114,114)):
    """비율 유지 리사이즈 + 패딩. 반환: (resized, (gain_w, gain_h), (pad_w, pad_h))"""
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nh, nw = new_shape  # (height, width)

    r = min(nh / h0, nw / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = nw - new_unpad[0], nh - new_unpad[1]
    dw /= 2; dh /= 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    gain_w, gain_h = r, r
    pad_w,  pad_h  = left, top
    return img, (gain_w, gain_h), (pad_w, pad_h)

def nms_numpy(boxes_xyxy, scores, iou_thr=0.45, top_k=300):
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1 = boxes_xyxy[:,0]; y1 = boxes_xyxy[:,1]
    x2 = boxes_xyxy[:,2]; y2 = boxes_xyxy[:,3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < top_k:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def preprocess_from_lb(lb_img, inp_dtype, inp_q):
    """letterbox된 BGR -> RGB/정규화 -> (1,H,W,C) -> dtype/양자화"""
    rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = rgb[None, ...]
    scale, zp = inp_q
    if inp_dtype == np.uint8:
        x = (x * 255.0 + 0.5).astype(np.uint8)
    elif inp_dtype == np.int8:
        if scale and scale > 0:
            x = np.clip(x / scale + zp, -128, 127).astype(np.int8)
        else:
            x = (x * 255.0 - 128).astype(np.int8)
    else:
        x = x.astype(inp_dtype)
    return x

def dequant(arr, q):
    if not np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32)
    scale, zp = q
    if scale and scale > 0:
        return (arr.astype(np.float32) - zp) * scale
    return arr.astype(np.float32)

def decode_yolov8_output(y, in_w, in_h, conf_thr, num_classes):
    """
    YOLOv8 TFLite 출력 디코드 (커스텀 클래스 수 지원)
      - 입력: y (TFLite output, shape (1,N,C) 또는 (1,C,N))
      - num_classes: len(names)
      - 채널:
        C == 4+nc  : [x,y,w,h, cls...]
        C == 5+nc  : [x,y,w,h,obj, cls...]

    반환: boxes(xyxy, 입력스케일 기준), scores, cls_ids
    """
    out = y
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected output shape: {out.shape}")

    # 배치 제거: (1, A, B) -> (A,B)
    if out.shape[0] == 1:
        out = out[0]  # now (A,B)

    a, b = out.shape
    nc = num_classes
    c1 = 4 + nc
    c2 = 5 + nc

    # 어느 축이 채널 축인지 자동 판별
    if a in (c1, c2) and b not in (c1, c2):
        # (C, N) 형태 → (N, C)로 transpose
        out = out.transpose(1, 0)
    elif b in (c1, c2) and a not in (c1, c2):
        # 이미 (N, C) 형태
        pass
    else:
        raise RuntimeError(
            f"Cannot determine channel dim. shape={out.shape}, "
            f"expected one dim in {{4+nc={c1}, 5+nc={c2}}} with nc={nc}"
        )

    N, C = out.shape
    if C not in (c1, c2):
        raise RuntimeError(f"Unsupported channel size: C={C}, expected {c1} or {c2}")

    xywh = out[:, :4].copy()
    # 정규화된 좌표일 경우 스케일 복원
    if xywh[:, 2:4].max() <= 2.0:
        xywh[:, 0] *= float(in_w)
        xywh[:, 1] *= float(in_h)
        xywh[:, 2] *= float(in_w)
        xywh[:, 3] *= float(in_h)

    if C == c1:
        # [x,y,w,h, cls...]
        cls = out[:, 4:]
        cls_score = cls.max(axis=1)
        conf = cls_score
        cls_id = cls.argmax(axis=1)
    else:
        # [x,y,w,h,obj, cls...]
        obj = out[:, 4]
        cls = out[:, 5:]
        cls_score = cls.max(axis=1)
        conf = obj * cls_score
        cls_id = cls.argmax(axis=1)

    m = conf >= conf_thr
    xywh, conf, cls_id = xywh[m], conf[m], cls_id[m]

    x, y, w, h = xywh.T
    x0 = x - w / 2.0
    y0 = y - h / 2.0
    x1 = x + w / 2.0
    y1 = y + h / 2.0
    boxes_xyxy = np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)
    return boxes_xyxy, conf.astype(np.float32), cls_id.astype(np.int32)


def unletterbox_xyxy(boxes_xyxy, gain, pad):
    gw, gh = gain
    pw, ph = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pw) / gw
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - ph) / gh
    return boxes

def draw_dets(img, boxes_xyxy, scores, cls_ids, names):
    H, W = img.shape[:2]
    for i in range(len(scores)):
        x0, y0, x1, y1 = boxes_xyxy[i]
        x0 = int(max(0, min(W - 1, x0))); y0 = int(max(0, min(H - 1, y0)))
        x1 = int(max(0, min(W - 1, x1))); y1 = int(max(0, min(H - 1, y1)))
        c  = int(cls_ids[i])
        name = names[c] if 0 <= c < len(names) else f"id{c}"
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        tag = f"{name} {scores[i]:.2f}"
        cv2.putText(img, tag, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, tag, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img

# ========= 단일 이미지 추론 함수 =========
def infer_one_image(itp, in_details, out_details, names, bgr_img):
    inp = in_details[0]
    out = out_details[0]
    in_idx  = inp["index"]
    out_idx = out["index"]
    in_h, in_w = int(inp["shape"][1]), int(inp["shape"][2])
    inp_dtype = inp["dtype"]
    inp_q     = inp.get("quantization", (0.0, 0))
    out_q     = out.get("quantization", (0.0, 0))

    # letterbox
    lb_img, (gw, gh), (pw, ph) = letterbox(bgr_img, (in_h, in_w))
    x = preprocess_from_lb(lb_img, inp_dtype, inp_q)

    # 추론
    t0 = time.time()
    itp.set_tensor(in_idx, x)
    itp.invoke()
    y = itp.get_tensor(out_idx)
    infer_ms = (time.time() - t0) * 1000.0

    # 후처리
    y = dequant(y, out_q)
    num_classes = len(names)
    boxes_in, scores, cls_ids = decode_yolov8_output(
        y, in_w, in_h, CONF_THRESH, num_classes
    )
    keep = nms_numpy(boxes_in, scores, IOU_THRESH, MAX_DETS)
    boxes_in = boxes_in[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]
    boxes_xyxy = unletterbox_xyxy(boxes_in, (gw, gh), (pw, ph))

    vis = draw_dets(bgr_img.copy(), boxes_xyxy, scores, cls_ids, names)
    return vis, infer_ms, len(scores)

# ========= 메인 =========
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")
    if not os.path.exists(METADATA_YAML):
        raise FileNotFoundError(f"라벨/메타 없음: {METADATA_YAML}")
    if not os.path.isdir(VISIBLE_DIR):
        raise FileNotFoundError(f"입력 폴더 없음: {VISIBLE_DIR}")

    cv2.setNumThreads(1)

    names = load_names(METADATA_YAML)
    itp, npu_on = make_interpreter(MODEL_PATH, USE_NPU, DELEGATE_LIB, CPU_THREADS)
    in_details  = itp.get_input_details()
    out_details = itp.get_output_details()

    # 입력 이미지 목록 (정렬된 상태로 고정)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(VISIBLE_DIR, e)))
    paths = sorted(paths)

    if not paths:
        raise RuntimeError(f"입력 이미지가 없습니다: {VISIBLE_DIR}")

    print(f"[INFO] 총 {len(paths)}개 이미지에 대해 추론을 수행합니다.")
    total_time = 0.0

    for i, p in enumerate(paths, start=1):
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[WARN] 이미지 로드 실패: {p}")
            continue

        vis, infer_ms, n_det = infer_one_image(itp, in_details, out_details, names, bgr)
        total_time += infer_ms

        # 저장 파일명
        base = os.path.basename(p)
        save_path = os.path.join(SAVE_DIR, base)
        cv2.imwrite(save_path, vis)

        print(f"[{i}/{len(paths)}] {base}  dets={n_det}  infer={infer_ms:.1f} ms  → {save_path}")

    print(f"[DONE] 평균 추론 시간: {total_time/len(paths):.2f} ms/image")

if __name__ == "__main__":
    main()
