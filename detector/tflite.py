# detector/yolov4tiny_tflite_infer.py
# --> YOLOv8 TFLite 전용 TFLiteWorker로 변경된 버전

import os
import cv2
import time
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
import logging

# ===== 로그 유틸 =====
LOG_EVERY_SEC = float(os.getenv("DET_LOG_EVERY", "2.0"))  # 0이면 하트비트 비활성
logger = logging.getLogger(__name__)

def _p(name, msg, level=logging.INFO):
    logger.log(level, "[%s] %s", name, msg)

# ===== YOLOv8 기본 파라미터 =====
SCORE_THRESH   = 0.15   # CONF_THRESH
NMS_IOU_THRESH = 0.45
MAX_DETS       = 300


# ===== 공통 유틸 (YOLOv8 배치 스크립트에서 가져온 로직) =====
def letterbox(img, new_shape, color=(114, 114, 114), cached_params=None):
    """
    비율 유지 리사이즈 + 패딩. 반환: (resized, (gain_w, gain_h), (pad_w, pad_h))
    
    cached_params: (r, new_unpad, top, bottom, left, right) - 캐시된 파라미터
    """
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nh, nw = new_shape  # (height, width)

    # 캐시된 파라미터 사용 or 새로 계산
    if cached_params is not None:
        r, new_unpad, top, bottom, left, right = cached_params
    else:
        r = min(nh / h0, nw / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = nw - new_unpad[0], nh - new_unpad[1]
        dw /= 2
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    gain_w, gain_h = r, r
    pad_w, pad_h = left, top
    
    # 캐시용 파라미터도 반환
    cache_params = (r, new_unpad, top, bottom, left, right)
    return img, (gain_w, gain_h), (pad_w, pad_h), cache_params


def nms_numpy(boxes_xyxy, scores, iou_thr=0.45, top_k=300):
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
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
    """
    letterbox된 BGR 이미지를 TFLite 입력 텐서로 변환.
    - int8, scale≈1/255, zp=-128 인 경우: 빠른 정수 경로 사용 (img - 128)
    - 그 외: 기존 float 기반 일반 경로 사용
    """
    scale, zp = inp_q

    # ===== [빠른 경로] 현재 네 모델 케이스: int8, scale=1/255, zp=-128 =====
    if inp_dtype == np.int8 and abs(scale - (1.0 / 255.0)) < 1e-6 and zp == -128:
        # BGR -> RGB 변환
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.int16)
        x -= 128                      # in-place 연산
        x = x.astype(np.int8)        # 여기서 이미 [-128,127] 범위 보장
        return x[None, ...]

    # ===== [일반 경로] 다른 모델에도 재사용 가능하도록 남겨둠 =====
    img = lb_img.astype(np.float32) / 255.0   # 0~1 정규화
    x = img[None, ...]
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

def preprocess_from_lb_inplace(lb_img, inp_dtype, inp_q, out_arr):
    """
    letterbox된 BGR 이미지를 TFLite 입력 텐서(out_arr)에 in-place로 변환.
    - out_arr: (1, H, W, C), dtype == inp_dtype
    - int8, scale≈1/255, zp=-128 인 경우: 빠른 정수 경로 사용 (img - 128)
    - 그 외: float 기반 일반 경로 사용 (필요시 확장 가능)
    """
    scale, zp = inp_q

    # ===== [빠른 경로] full-int8, scale=1/255, zp=-128 =====
    if inp_dtype == np.int8 and abs(scale - (1.0 / 255.0)) < 1e-6 and zp == -128:
        # BGR -> RGB 변환
        rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        # lb_img: uint8(0~255) 가정
        # q = img - 128  → real = (q + 128)/255 = img/255
        dst = out_arr[0]                  # (H, W, C)
        tmp = rgb.astype(np.int16)        # 오버플로우 방지
        tmp -= 128                        # in-place 연산
        dst[...] = tmp.astype(np.int8)    # [-128,127] 범위에 정확히 매핑
        return out_arr

    # ===== [일반 경로] (다른 모델에도 재사용 가능) =====
    img = lb_img.astype(np.float32) / 255.0
    x = img.astype(inp_dtype)
    out_arr[0, ...] = x
    return out_arr

def dequant(arr, q):
    if not np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32)
    scale, zp = q
    if scale and scale > 0:
        return (arr.astype(np.float32) - zp) * scale
    return arr.astype(np.float32)


def decode_yolov8_output(y, in_w, in_h, conf_thr, num_classes):
    """
    YOLOv8 TFLite 출력 디코드
      - 입력: y (TFLite output, shape (1,N,C) 또는 (1,C,N))
      - num_classes: len(labels)
      - 채널:
        C == 4+nc  : [x,y,w,h, cls...]
        C == 5+nc  : [x,y,w,h,obj, cls...]
    """
    out = y
    if out.ndim != 3:
        raise RuntimeError(f"Unexpected output shape: {out.shape}")

    # 배치 제거: (1, A, B) -> (A,B)
    if out.shape[0] == 1:
        out = out[0]

    a, b = out.shape
    nc = num_classes
    c1 = 4 + nc
    c2 = 5 + nc

    # 어느 축이 채널 축인지 자동 판별
    if a in (c1, c2) and b not in (c1, c2):
        # (C, N) -> (N, C)
        out = out.transpose(1, 0)
    elif b in (c1, c2) and a not in (c1, c2):
        # 이미 (N, C)
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

    if xywh.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

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


def _draw_boxes(frame_bgr, boxes_xyxy, classes, scores, labels,
                thr=SCORE_THRESH):
    """원본 프레임 좌표 기준 boxes_xyxy를 그대로 그림"""
    H, W = frame_bgr.shape[:2]
    base = min(H, W)
    thickness = max(1, int(round(base / 240)))   # 240 기준
    font_scale = max(0.4, base / 640.0)          # 640 기준

    for i in range(len(scores)):
        if scores[i] < thr:
            continue
        x0, y0, x1, y1 = boxes_xyxy[i]
        x0 = int(max(0, min(W - 1, x0)))
        y0 = int(max(0, min(H - 1, y0)))
        x1 = int(max(0, min(W - 1, x1)))
        y1 = int(max(0, min(H - 1, y1)))

        cls_id = int(classes[i])
        name = labels[cls_id] if 0 <= cls_id < len(labels) else f"id:{cls_id}"

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 255), thickness)
        tag = f"{name} {scores[i]:.2f}"

        # 가독성 있는 두겹 텍스트
        cv2.putText(frame_bgr, tag, (x0, max(0, y0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, tag, (x0, max(0, y0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0),
                    1, cv2.LINE_AA)
    return frame_bgr


class TFLiteWorker(threading.Thread):
    """
    YOLOv8 TFLite 전용 추론 스레드
    - input_buf: DoubleBuffer (read() -> (frame_bgr, ts))
    - output_buf: DoubleBuffer (write((vis_frame_bgr, ts)))
    - 표시 해상도(TARGET_W,H)에 맞춰 리사이즈 후, 그 위에 박스 오버레이해서 출력
      (※ 여기서는 원본 크기에 박스를 그린 뒤, 마지막에 리사이즈)
    """
    def __init__(self,
                 model_path: str,
                 labels_path: str,
                 input_buf,
                 output_buf,
                 # allowed_class_ids: list or tuple of ints to restrict detection to those classes
                 # e.g. allowed_class_ids=[1] will keep only class_id == 1
                 allowed_class_ids: list = None,
                 use_npu: bool = True,
                 delegate_lib: str = "/usr/lib/libvx_delegate.so",
                 cpu_threads: int = 1,                  # 기본 1로 완화
                 target_fps: float = 0,
                 target_res: tuple = (960, 540),
                 name: str = "DetWorker"):
        super().__init__(daemon=True, name=name)
        self.model_path = model_path
        self.labels = self._load_labels(labels_path)
        self.input_buf  = input_buf
        self.output_buf = output_buf
        self.use_npu = use_npu
        self.delegate_lib = delegate_lib
        self.cpu_threads = cpu_threads
        self.stop_evt = threading.Event()
        self._last_beat = 0.0
        self.target_period = 1.0/target_fps if target_fps and target_fps > 0 else 0.0
        self._last_tick = time.perf_counter()
        self.target_res = target_res
        # 리스트/튜플 -> numpy array for fast isin checks (dtype int32)
        self.allowed_class_ids = None if allowed_class_ids is None else np.asarray(allowed_class_ids, dtype=np.int32)
        
        cv2.setNumThreads(4)
        
        # === 통계 지표 ===
        self._ema_alpha = 0.3
        self._ema_total_ms = None
        self._ema_invoke_ms = None
        self._win_start_ts = time.time()
        self._win_frames = 0

        # === Letterbox 캐싱 (카메라 해상도 고정 시) ===
        self._lb_params_cache = None  # (r, new_unpad, top, bottom, left, right, expected_shape)
        self._lb_gain_pad_cache = None  # (gw, gh, pw, ph)

        self.itp, self.inp, self.outs, self.accel = self._make_interpreter()
        _p(self.name, f"init accel={self.accel}, threads={self.cpu_threads}, target_fps={(1.0/self.target_period) if self.target_period>0 else 0}")

        # ===== [변경] 입력 버퍼를 한 번만 만들어 재사용 =====
        in_shape = self.inp["shape"]                  # (1, H, W, C)
        in_dtype = self.inp["dtype"]                  # 보통 np.int8
        self._input_buf = np.empty(in_shape, dtype=in_dtype)

    def _load_labels(self, path):
        # 기존처럼 한 줄당 한 클래스 이름이 있는 txt 파일을 사용
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def _make_interpreter(self):
        delegates = None
        accel = "CPU"
        if self.use_npu and self.delegate_lib and os.path.exists(self.delegate_lib):
            try:
                delegates = [tflite.load_delegate(self.delegate_lib)]
                accel = "NPU"
                _p(self.name, f"VX delegate 로드: {self.delegate_lib}")
            except Exception as e:
                _p(self.name, f"delegate 로드 실패 → CPU: {e}")

        itp = tflite.Interpreter(model_path=self.model_path,
                                 experimental_delegates=delegates,
                                 num_threads=self.cpu_threads)
        itp.allocate_tensors()
        inp = itp.get_input_details()[0]
        outs = itp.get_output_details()
        _p(self.name, f"TFLite accel={accel}, threads={self.cpu_threads}")
        return itp, inp, outs, accel

    def _get_outputs_float(self):
        outs = []
        for od in self.outs:
            arr = self.itp.get_tensor(od["index"])
            if np.issubdtype(arr.dtype, np.integer):
                scale, zp = od["quantization"]
                arr = (arr.astype(np.float32) - zp) * (scale if scale != 0 else 1.0)
            else:
                arr = arr.astype(np.float32)
            outs.append(arr)
        return outs

    def _infer_once(self, frame_bgr):
        """
        한 프레임에 대해:
        - letterbox 전처리
        - TFLite 추론
        - YOLOv8 디코드 + NMS + unletterbox
        """
        t0 = time.perf_counter()

        # --- 입력 shape / quant 정보 ---
        in_shape = self.inp["shape"]  # (1,H,W,C)
        in_h, in_w = int(in_shape[1]), int(in_shape[2])
        inp_dtype = self.inp["dtype"]
        inp_q     = self.inp.get("quantization", (0.0, 0))

        # --- letterbox + 양자화 전처리 (캐싱 적용) ---
        h0, w0 = frame_bgr.shape[:2]
        
        # 캐시 확인: 프레임 크기가 같으면 letterbox 파라미터 재사용
        if self._lb_params_cache and self._lb_params_cache[6] == (h0, w0):
            # 캐시 히트: 계산 스킵
            cached_params = self._lb_params_cache[:6]
            lb_img, (gw, gh), (pw, ph), _ = letterbox(frame_bgr, (in_h, in_w), cached_params=cached_params)
        else:
            # 캐시 미스: 새로 계산하고 저장
            lb_img, (gw, gh), (pw, ph), cache_params = letterbox(frame_bgr, (in_h, in_w))
            self._lb_params_cache = cache_params + ((h0, w0),)
        
        # x = preprocess_from_lb(lb_img, inp_dtype, inp_q)
        x = preprocess_from_lb_inplace(lb_img, inp_dtype, inp_q, self._input_buf)
        t_pre = time.perf_counter()

        # ---- 핵심 추론 ----
        t_inv0 = time.perf_counter()
        self.itp.set_tensor(self.inp["index"], x)
        self.itp.invoke()
        outs = self._get_outputs_float()
        t_inv1 = time.perf_counter()

        # ---- 후처리(락 밖) ----
        # YOLOv8은 보통 출력 하나만 사용 (det)
        y = outs[0]
        # (B,N,C)/(B,C,N) → 박스/점수/클래스
        boxes_in, scores, classes = decode_yolov8_output(
            y, in_w, in_h, SCORE_THRESH, num_classes=len(self.labels)
        )
        # Optional: restrict to allowed classes before NMS to avoid cross-class suppression
        if self.allowed_class_ids is not None and classes.size > 0:
            mask = np.isin(classes, self.allowed_class_ids)
            if not mask.any():
                # nothing left
                return np.zeros((0,), np.float32), np.zeros((0,4), np.float32), np.zeros((0,), np.int32)
            boxes_in = boxes_in[mask]
            scores = scores[mask]
            classes = classes[mask]

        # NMS
        keep = nms_numpy(boxes_in, scores, NMS_IOU_THRESH, MAX_DETS)
        boxes_in = boxes_in[keep]
        scores   = scores[keep]
        classes  = classes[keep]
        # letterbox 역변환 → 원본 프레임 좌표
        boxes_xyxy = unletterbox_xyxy(boxes_in, (gw, gh), (pw, ph))

        t_post = time.perf_counter()

        # 상세 타이밍(ms)
        invoke_ms = (t_inv1 - t_inv0) * 1000.0
        pre_ms  = (t_pre - t0) * 1000.0
        post_ms = (t_post - t_inv1) * 1000.0
        total_ms= (t_post - t0) * 1000.0

        # 통계 업데이트 (탐지 건수 포함)
        self._update_stats(invoke_ms, total_ms, det_count=len(boxes_xyxy), raw_count=len(scores))

        # 디버깅이 필요하면 아래 주석 해제해서 세부 타이밍 로그 가능
        # _p(self.name, f"pre={pre_ms:5.1f} | invoke={invoke_ms:6.1f} | "
        #               f"post={post_ms:5.1f} | total={total_ms:6.1f} ms")

        return scores, boxes_xyxy, classes

    def run(self):
        while not self.stop_evt.is_set():
            item = self.input_buf.read()

            # === 프레임 없음 → 스트림별 백오프(time.sleep) 적용 ===
            if not item:
                if self.target_period > 0:
                    idle = max(0.001, min(0.25 * self.target_period, 0.010))
                    time.sleep(idle)
                else:
                    time.sleep(0.005)
                self._heartbeat()
                continue

            frame, ts = item
            scores, boxes_xyxy, classes = self._infer_once(frame)

            # 1) 원본 프레임에 박스 오버레이
            vis = frame.copy()
            vis = _draw_boxes(vis, boxes_xyxy, classes, scores, self.labels,
                              thr=SCORE_THRESH)

            # 2) 표시용 해상도(TARGET_W x TARGET_H)로 리사이즈
            # vis = cv2.resize(vis, self.target_res, interpolation=cv2.INTER_AREA)

            # 3) 검출 결과를 bbox 리스트로 변환 (x, y, w, h, confidence)
            detections = []
            if len(boxes_xyxy) > 0:
                for i, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    conf = scores[i] if i < len(scores) else 0.0
                    cls = classes[i] if i < len(classes) else 0
                    detections.append((float(x1), float(y1), float(w), float(h), float(conf), int(cls)))

            # 출력 버퍼로 전송 (vis, ts, detections)
            self.output_buf.write((vis, ts, detections))
            self._heartbeat()

            # === 타깃 FPS 페이싱: 루프 주기가 target_period보다 빠르면 남은 시간만큼 쉼 ===
            if self.target_period > 0:
                now = time.perf_counter()
                elapsed = now - self._last_tick
                if elapsed < self.target_period:
                    time.sleep(self.target_period - elapsed)
                self._last_tick = time.perf_counter()

    def _heartbeat(self):
        if LOG_EVERY_SEC <= 0:
            return
        now = time.time()
        if now - self._last_beat >= LOG_EVERY_SEC:
            # FPS 계산(윈도우)
            win_elapsed = max(1e-6, now - self._win_start_ts)
            fps = self._win_frames / win_elapsed
            self._win_start_ts = now
            self._win_frames = 0

            # EMA 값들
            et = self._ema_total_ms if self._ema_total_ms is not None else 0.0
            ei = self._ema_invoke_ms if self._ema_invoke_ms is not None else 0.0

            tgt = (1.0/self.target_period) if self.target_period>0 else 0
            det = getattr(self, "_win_det", 0)
            det_raw = getattr(self, "_win_det_raw", 0)
            _p(self.name, f"{self.accel} | FPS={fps:5.2f} (target={tgt}) | "
                          f"total={et:6.1f} ms | invoke={ei:6.1f} ms | det={det} raw={det_raw}")
            self._last_beat = now
            self._win_det = 0
            self._win_det_raw = 0

    def _update_stats(self, invoke_ms, total_ms, det_count=0, raw_count=0):
        # 윈도우 프레임 카운트
        self._win_frames += 1
        # EMA 업데이트
        a = self._ema_alpha
        def ema(prev, x):
            return x if prev is None else (a * x + (1.0 - a) * prev)
        self._ema_total_ms = ema(self._ema_total_ms, total_ms)
        self._ema_invoke_ms = ema(self._ema_invoke_ms, invoke_ms)
        self._win_det = getattr(self, "_win_det", 0) + det_count
        self._win_det_raw = getattr(self, "_win_det_raw", 0) + raw_count

    def stop(self):
        self.stop_evt.set()
