import os
import csv
import cv2
import numpy as np


class CaptureLoader:
    """
    캡처 세션 재사용 유틸리티.
    - metadata.csv를 읽어 RGB/IR 비디오와 RAW16 npy를 순서대로 반환
    - yield: dict(index, rgb_ts, ir_ts, diff_ms, rgb_frame, ir_frame, ir_raw)
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.meta_path = os.path.join(root_dir, "metadata.csv")
        self.rgb_path = os.path.join(root_dir, "rgb.mp4")
        self.ir_path = os.path.join(root_dir, "ir_vis.mp4")
        self.ir16_dir = os.path.join(root_dir, "ir16")
        self.meta_rows = self._load_meta(self.meta_path)
        self.rgb_cap = cv2.VideoCapture(self.rgb_path)
        self.ir_cap = cv2.VideoCapture(self.ir_path)

    def _load_meta(self, path):
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def __iter__(self):
        for row in self.meta_rows:
            idx = int(row["index"])
            rgb_ts = row["rgb_ts"]
            ir_ts = row["ir_ts"]
            diff_ms = float(row["diff_ms"])
            ok_rgb, rgb_frame = self.rgb_cap.read()
            ok_ir, ir_frame = self.ir_cap.read()
            if not ok_rgb or not ok_ir:
                break
            raw_path = row.get("ir_raw", "")
            ir_raw = None
            if raw_path:
                np_path = os.path.join(self.root_dir, raw_path)
                if os.path.exists(np_path):
                    ir_raw = np.load(np_path)
            yield {
                "index": idx,
                "rgb_ts": rgb_ts,
                "ir_ts": ir_ts,
                "diff_ms": diff_ms,
                "rgb": rgb_frame,
                "ir": ir_frame,
                "ir_raw": ir_raw,
            }

    def release(self):
        if self.rgb_cap:
            self.rgb_cap.release()
        if self.ir_cap:
            self.ir_cap.release()

