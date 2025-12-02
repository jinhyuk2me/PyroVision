import os
import csv
import time
import logging
import argparse
from collections import deque
from datetime import datetime
import json

import cv2
import numpy as np

from configs.get_cfg import get_cfg
from core.buffer import DoubleBuffer
from camera.source_factory import create_rgb_source, create_ir_source
from detector.tflite import TFLiteWorker


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ts_to_epoch_ms(ts):
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%y%m%d%H%M%S%f").timestamp() * 1000.0
    except Exception:
        return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def create_writer(path, frame_shape, codec, fps):
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {path}")
    return writer


def parse_args():
    parser = argparse.ArgumentParser(description="Capture synchronized RGB/IR streams")
    parser.add_argument("--output", help="Output directory override")
    parser.add_argument("--duration", type=float, help="Duration in seconds")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frame pairs")
    parser.add_argument("--save-det", action="store_true", help="Save detector results to JSONL")
    parser.add_argument("--det-json", help="Detector output JSONL path (default: output_dir/det.jsonl)")
    return parser.parse_args()


def main():
    setup_logging()
    logger = logging.getLogger("capture")

    args = parse_args()
    cfg = get_cfg()
    capture_cfg = cfg.get("CAPTURE", {})
    if not capture_cfg:
        raise RuntimeError("CAPTURE configuration not found in config.yaml")

    if args.output:
        capture_cfg['OUTPUT_DIR'] = args.output
    if args.duration is not None:
        capture_cfg['DURATION_SEC'] = max(0.0, args.duration)
    if args.max_frames is not None:
        capture_cfg['MAX_FRAMES'] = max(0, args.max_frames)

    output_dir = capture_cfg.get("OUTPUT_DIR", "./capture_session")
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "ir16"))

    duration_sec = capture_cfg.get("DURATION_SEC")
    max_frames = capture_cfg.get("MAX_FRAMES")
    max_diff_ms = capture_cfg.get("MAX_DIFF_MS", 80)

    save_rgb = capture_cfg.get("SAVE_RGB_VIDEO", True)
    save_ir_vis = capture_cfg.get("SAVE_IR_VIDEO", True)
    save_ir_raw = capture_cfg.get("SAVE_IR_RAW16", True)

    rgb_codec = capture_cfg.get("RGB_CODEC", "mp4v")
    ir_codec = capture_cfg.get("IR_CODEC", "mp4v")

    rgb_cfg = cfg['CAMERA']['RGB_FRONT']
    ir_cfg = cfg['CAMERA']['IR']
    input_cfg = cfg.get('INPUT', {})
    rgb_input_cfg = dict(input_cfg.get('RGB', {}))
    ir_input_cfg = dict(input_cfg.get('IR', {}))
    det_cfg = {
        'MODEL': cfg.get('MODEL'),
        'LABEL': cfg.get('LABEL'),
        'DELEGATE': cfg.get('DELEGATE'),
        'USE_NPU': True,
        'CPU_THREADS': 1,
        'NAME': "DetCapture",
        'ALLOWED_CLASSES': None,
    }

    d_rgb = DoubleBuffer()
    d_ir = DoubleBuffer()
    d16_ir = DoubleBuffer()
    det_in_buf = DoubleBuffer() if args.save_det else None
    d_rgb_det = DoubleBuffer() if args.save_det else None

    rgb_source = create_rgb_source(rgb_cfg, rgb_input_cfg, d_rgb)
    ir_source = create_ir_source(ir_cfg, ir_input_cfg, d_ir, d16_ir)

    logger.info("Starting RGB source: %s", getattr(rgb_source, 'name', 'RGB'))
    rgb_source.start()
    logger.info("Starting IR source: %s", getattr(ir_source, 'name', 'IR'))
    ir_source.start()

    rgb_queue = deque()
    ir_queue = deque()
    raw_map = {}
    meta_rows = []
    det_rows = []
    det_worker = None

    rgb_writer = None
    ir_writer = None
    det_json_path = ""

    start_time = time.time()
    last_rgb_ts = None
    last_ir_ts = None
    last_raw_ts = None
    saved = 0

    metadata_path = os.path.join(output_dir, "metadata.csv")
    logger.info("Recording... press Ctrl+C to stop")

    try:
        while True:
            now = time.time()
            if duration_sec and (now - start_time) >= duration_sec:
                logger.info("Duration reached")
                break
            if max_frames and saved >= max_frames:
                logger.info("Frame limit reached")
                break

            rgb_item = d_rgb.read()
            if rgb_item and rgb_item[0] is not None:
                ts = rgb_item[1]
                if ts != last_rgb_ts:
                    last_rgb_ts = ts
                    rgb_queue.append(rgb_item)

            ir_item = d_ir.read()
            if ir_item and ir_item[0] is not None:
                ts = ir_item[1]
                if ts != last_ir_ts:
                    last_ir_ts = ts
                    ir_queue.append(ir_item)

            raw_item = d16_ir.read()
            if raw_item and raw_item[0] is not None:
                ts = raw_item[1]
                if ts != last_raw_ts:
                    last_raw_ts = ts
                    raw_map[ts] = raw_item
                    if len(raw_map) > 200:
                        first_key = next(iter(raw_map))
                        raw_map.pop(first_key, None)

            if not rgb_queue or not ir_queue:
                time.sleep(0.005)
                continue

            rgb_ts = rgb_queue[0][1]
            ir_ts = ir_queue[0][1]
            t_rgb = ts_to_epoch_ms(rgb_ts)
            t_ir = ts_to_epoch_ms(ir_ts)
            if t_rgb is None or t_ir is None:
                if t_rgb is None:
                    rgb_queue.popleft()
                if t_ir is None:
                    ir_queue.popleft()
                continue

            diff = t_rgb - t_ir
            if abs(diff) > max_diff_ms:
                if t_rgb > t_ir:
                    ir_queue.popleft()
                else:
                    rgb_queue.popleft()
                continue

            rgb_frame, _ = rgb_queue.popleft()
            ir_frame_entry = ir_queue.popleft()
            ir_frame = ir_frame_entry[0]

            raw_entry = raw_map.pop(ir_ts, None)
            raw16 = raw_entry[0] if raw_entry else None

            if args.save_det:
                if det_worker is None:
                    det_json_path = args.det_json or os.path.join(output_dir, "det.jsonl")
                    det_worker = TFLiteWorker(
                        model_path=det_cfg['MODEL'],
                        labels_path=det_cfg['LABEL'],
                        input_buf=det_in_buf,
                        output_buf=d_rgb_det,
                        allowed_class_ids=det_cfg['ALLOWED_CLASSES'],
                        use_npu=det_cfg['USE_NPU'],
                        delegate_lib=det_cfg['DELEGATE'],
                        cpu_threads=det_cfg['CPU_THREADS'],
                        target_fps=rgb_cfg.get('FPS', 0),
                        target_res=tuple(cfg.get('TARGET_RES', (rgb_frame.shape[1], rgb_frame.shape[0]))),
                        name=det_cfg['NAME'],
                    )
                    det_worker.start()
                # push frame for detection
                det_in_buf.write((rgb_frame, rgb_ts))
                det_item = None
                # wait briefly for matching result
                for _ in range(10):
                    det_item = d_rgb_det.read()
                    if det_item and len(det_item) > 1 and det_item[1] == rgb_ts:
                        break
                    time.sleep(0.01)
                dets = det_item[2] if det_item and len(det_item) > 2 else []
            else:
                dets = []

            if save_rgb:
                if rgb_writer is None:
                    fps = capture_cfg.get("RGB_FPS", rgb_cfg['FPS'])
                    path = os.path.join(output_dir, "rgb.mp4")
                    rgb_writer = create_writer(path, rgb_frame.shape, rgb_codec, fps)
                rgb_writer.write(rgb_frame)

            if save_ir_vis:
                if ir_writer is None:
                    fps = capture_cfg.get("IR_FPS", ir_cfg['FPS'])
                    path = os.path.join(output_dir, "ir_vis.mp4")
                    ir_writer = create_writer(path, ir_frame.shape, ir_codec, fps)
                ir_writer.write(ir_frame)

            raw_path = ""
            if save_ir_raw and raw16 is not None:
                raw_path = os.path.join("ir16", f"{ir_ts}.npy")
                np.save(os.path.join(output_dir, raw_path), raw16)

            meta_rows.append([saved, rgb_ts, ir_ts, diff, raw_path])
            if args.save_det:
                det_rows.append({
                    "index": saved,
                    "rgb_ts": rgb_ts,
                    "ir_ts": ir_ts,
                    "diff_ms": diff,
                    "detections": [
                        {
                            "x": det[0],
                            "y": det[1],
                            "w": det[2],
                            "h": det[3],
                            "conf": det[4],
                            "cls": det[5],
                        } for det in dets
                    ],
                })
            saved += 1

    except KeyboardInterrupt:
        logger.info("Capture interrupted by user")
    finally:
        if rgb_writer:
            rgb_writer.release()
        if ir_writer:
            ir_writer.release()
        if det_worker:
            det_worker.stop()
            det_worker.join(timeout=2.0)

        try:
            rgb_source.stop()
        except Exception:
            pass
        try:
            ir_source.stop()
        except Exception:
            pass

        with open(metadata_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "rgb_ts", "ir_ts", "diff_ms", "ir_raw"])
            writer.writerows(meta_rows)
        if args.save_det and det_rows:
            with open(det_json_path, "w", encoding="utf-8") as f:
                for row in det_rows:
                    f.write(json.dumps(row, ensure_ascii=True))
                    f.write("\n")

        logger.info("Saved %d synchronized frames to %s", saved, output_dir)


if __name__ == "__main__":
    main()
