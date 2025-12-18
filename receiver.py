import base64
import json
import os
import socket
import struct
import time
import zlib

import cv2
import numpy as np

REQUIRED_IMAGES = {
    "rgb_det": ("data_b64", "shape", "dtype"),
    "ir": ("data_b64", "shape", "dtype"),
}

# ===== 저장 경로 설정 =====
SAVE_DIR_RGB = "save/visible"
SAVE_DIR_IR = "save/lwir"
os.makedirs(SAVE_DIR_RGB, exist_ok=True)
os.makedirs(SAVE_DIR_IR, exist_ok=True)


def _decode_image(entry):
    """JSON 패킷의 이미지 엔트리를 numpy 배열로 복원"""
    if not entry:
        return None
    # 필수 키 존재 여부 확인
    if not all(k in entry for k in ("data_b64", "shape", "dtype")):
        print("[Receiver] Invalid image entry schema")
        return None
    data_b64 = entry.get("data_b64")
    if not data_b64:
        return None
    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        print("[Receiver] base64 decode failed")
        return None

    if entry.get("compressed"):
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    dtype = np.dtype(entry.get("dtype"))
    shape = tuple(entry.get("shape", ()))
    # frombuffer로 만든 배열은 읽기 전용일 수 있어 OpenCV 그리기 시 에러가 날 수 있음
    arr = np.frombuffer(raw, dtype=dtype).copy()
    try:
        return arr.reshape(shape)
    except Exception:
        return None


def _maybe_decode_annots(entry):
    """fusion 결과의 eo_annotations를 파이썬 객체로 복원"""
    if not entry:
        return []
    if isinstance(entry, str):
        try:
            return json.loads(entry)
        except Exception:
            return []
    if isinstance(entry, dict):
        return entry.get("eo_annotations", []) or []
    if isinstance(entry, list):
        return entry
    return []


def _draw_max_temp_text(frame, ir_entry):
    """IR 프레임에 최고/최저 온도 정보를 오버레이"""
    if frame is None or not isinstance(ir_entry, dict):
        return frame

    # 일부 입력 배열이 read-only일 수 있어 쓰기 가능 사본으로 변환
    if not frame.flags.writeable:
        frame = frame.copy()

    meta = ir_entry.get("max_temp") or {}
    if "tau" not in meta and ir_entry.get("tau") is not None:
        meta = dict(meta)
        meta["tau"] = ir_entry.get("tau")
    temp = meta.get("temp_corrected", meta.get("temp_raw"))
    if temp is None:
        return frame

    tau = meta.get("tau")
    text = f"Max {temp:.1f}C"
    if meta.get("min_temp") is not None:
        text += f" / Min {meta['min_temp']:.1f}C"
    if tau is not None:
        text += f" | tau {float(tau):.3f}"

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(2, w - tw - 4)
    y = max(th + 2, h - 4)

    # 배경 박스로 가독성 확보
    cv2.rectangle(
        frame,
        (x - 2, y - th - baseline - 2),
        (x + tw + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _rotate_frame(frame, angle):
    """90도 단위 회전"""
    if frame is None or angle % 360 == 0:
        return frame
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class ImageReceiver:
    def __init__(self, host="0.0.0.0", port=9999, max_packet_mb=4.0):
        self.host = host
        self.port = port
        self.server_sock = None
        self.client_sock = None
        self.max_packet_bytes = int(max_packet_mb * 1024 * 1024)

    def start_server(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024 * 10)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(1)
            print(f"[Receiver] Server listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[Receiver] Server start failed: {e}")
            return False

    def wait_for_client(self):
        try:
            print("[Receiver] Waiting for client connection...")
            self.client_sock, addr = self.server_sock.accept()
            print(f"[Receiver] Client connected from {addr}")
            return True
        except Exception as e:
            print(f"[Receiver] Accept failed: {e}")
            return False

    def send_control_command(self, command):
        """Sender에게 JSON 제어 명령 전송"""
        try:
            if not self.client_sock:
                return False
            payload = json.dumps({"command": command}).encode("utf-8")
            size_header = struct.pack(">L", len(payload))
            self.client_sock.sendall(size_header)
            self.client_sock.sendall(payload)
            return True
        except Exception as e:
            print(f"[Receiver] Failed to send control command: {e}")
            return False

    def receive_frame_data(self):
        try:
            size_header = self._recv_exact(4)
            if not size_header:
                return None
            payload_size = struct.unpack(">L", size_header)[0]
            if payload_size <= 0 or payload_size > self.max_packet_bytes:
                print(f"[Receiver] Invalid payload size: {payload_size}")
                return None

            payload = self._recv_exact(payload_size)
            if not payload:
                return None

            # 압축 해제 시도
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                pass

            data_dict = json.loads(payload.decode("utf-8"))
            return data_dict
        except Exception as e:
            print(f"[Receiver] Receive failed: {e}")
            return None

    def _recv_exact(self, size):
        data = b""
        while len(data) < size:
            packet = self.client_sock.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data

    def close(self):
        if self.client_sock:
            try:
                self.client_sock.close()
            except Exception:
                pass
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass
        print("[Receiver] Server closed")


def save_frames(rgb_frame, ir_frame, timestamp):
    ts_str = f"{timestamp:.6f}".replace(".", "_")
    if rgb_frame is not None:
        rgb_file = os.path.join(SAVE_DIR_RGB, f"rgb_frame_{ts_str}.jpg")
        if cv2.imwrite(rgb_file, rgb_frame):
            print(f"[SAVED] RGB → {rgb_file}")
    if ir_frame is not None:
        ir_file = os.path.join(SAVE_DIR_IR, f"ir16_frame_{ts_str}.tiff")
        if cv2.imwrite(ir_file, ir_frame):
            print(f"[SAVED] IR → {ir_file}")


def receive_and_display(host="0.0.0.0", port=9999):
    receiver = ImageReceiver(host, port)

    if not receiver.start_server():
        return
    if not receiver.wait_for_client():
        receiver.close()
        return

    cv2.namedWindow("PyroVision", cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    saving = False

    decode_times = []
    display_times = []
    recv_times = []
    loop_times = []
    ir_scale = 1.0
    rgb_scale = 1.0
    ir_rot = 0
    rgb_rot = 0

    try:
        while True:
            t_loop_start = time.perf_counter()

            t_recv_start = time.perf_counter()
            packet = receiver.receive_frame_data()
            t_recv_end = time.perf_counter()
            recv_times.append((t_recv_end - t_recv_start) * 1000)

            if packet is None:
                print("[Receiver] Connection lost. Waiting for reconnection...")
                if receiver.wait_for_client():
                    print("[Receiver] Client reconnected")
                    continue
                else:
                    break

            timestamp = packet.get("timestamp", 0)
            images = packet.get("images", {})
            if not isinstance(images, dict):
                print("[Receiver] Invalid packet: images missing")
                continue
            # 필수 이미지 스키마 체크
            if any(
                name in images and not all(k in images[name] for k in REQUIRED_IMAGES[name])
                for name in REQUIRED_IMAGES
            ):
                print("[Receiver] Invalid image schema, skipping packet")
                continue

            ir_display = None
            rgb_det_display = None
            ir_entry = images.get("ir") if isinstance(images.get("ir"), dict) else None

            t_decode_start = time.perf_counter()
            if "ir" in images:
                ir_display = _decode_image(images.get("ir"))
            if "rgb_det" in images:
                rgb_det_info = images.get("rgb_det")
                rgb_det_display = _decode_image(rgb_det_info)
            t_decode_end = time.perf_counter()
            decode_times.append((t_decode_end - t_decode_start) * 1000)

            t_display_start = time.perf_counter()
            if rgb_det_display is not None:
                rgb_det_display = _rotate_frame(rgb_det_display, rgb_rot)
                if abs(rgb_scale - 1.0) > 1e-3:
                    new_w = max(1, int(rgb_det_display.shape[1] * rgb_scale))
                    new_h = max(1, int(rgb_det_display.shape[0] * rgb_scale))
                    rgb_det_display = cv2.resize(rgb_det_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                target_h, target_w = rgb_det_display.shape[:2]
                if ir_display is not None:
                    ir_display = _rotate_frame(ir_display, ir_rot)
                    if abs(ir_scale - 1.0) > 1e-3:
                        new_w = max(1, int(ir_display.shape[1] * ir_scale))
                        new_h = max(1, int(ir_display.shape[0] * ir_scale))
                        ir_display = cv2.resize(ir_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    ir_display = _draw_max_temp_text(ir_display, ir_entry)
                    ir_h, ir_w = ir_display.shape[:2]
                    pad_top = max(0, (target_h - ir_h) // 2)
                    pad_bottom = max(0, target_h - ir_h - pad_top)
                    pad_left = max(0, (target_w - ir_w) // 2)
                    pad_right = max(0, target_w - ir_w - pad_left)
                    ir_padded = cv2.copyMakeBorder(
                        ir_display,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    try:
                        combined = np.vstack([rgb_det_display, ir_padded])
                    except Exception:
                        # 최종 폭이 다르면 패딩을 다시 맞춰서 안전하게 이어붙이기
                        min_w = min(rgb_det_display.shape[1], ir_padded.shape[1])
                        rgb_crop = rgb_det_display[:, :min_w]
                        ir_crop = ir_padded[:, :min_w]
                        combined = np.vstack([rgb_crop, ir_crop])
                else:
                    combined = rgb_det_display
                cv2.imshow("PyroVision", combined)
            elif ir_display is not None:
                ir_display = _rotate_frame(ir_display, ir_rot)
                if abs(ir_scale - 1.0) > 1e-3:
                    new_w = max(1, int(ir_display.shape[1] * ir_scale))
                    new_h = max(1, int(ir_display.shape[0] * ir_scale))
                    ir_display = cv2.resize(ir_display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                ir_display = _draw_max_temp_text(ir_display, ir_entry)
                cv2.imshow("PyroVision", ir_display)
            t_display_end = time.perf_counter()
            display_times.append((t_display_end - t_display_start) * 1000)

            frame_count += 1
            t_loop_end = time.perf_counter()
            loop_times.append((t_loop_end - t_loop_start) * 1000)

            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                latency = (current_time - timestamp) * 1000 if timestamp else 0

                avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
                avg_display = sum(display_times) / len(display_times) if display_times else 0
                avg_recv = sum(recv_times) / len(recv_times) if recv_times else 0
                avg_loop = sum(loop_times) / len(loop_times) if loop_times else 0

                print(f"[Receiver] Frame: {frame_count}, FPS: {fps:.2f}, Latency: {latency:.1f}ms")
                print(
                    f"  → Recv: {avg_recv:.2f}ms, Decode: {avg_decode:.2f}ms, Display: {avg_display:.2f}ms, Total Loop: {avg_loop:.2f}ms"
                )

                last_print_time = current_time
                decode_times.clear()
                display_times.clear()
                recv_times.clear()
                loop_times.clear()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if not saving:
                    saving = True
                    print("[Receiver] Saving started...")
                    receiver.send_control_command("start_saving")
            elif key == ord("e"):
                if saving:
                    saving = False
                    print("[Receiver] Saving stopped.")
                    receiver.send_control_command("stop_saving")
            elif key == ord("["):  # IR 축소
                ir_scale = max(0.1, ir_scale - 0.1)
                print(f"[Receiver] IR scale: {ir_scale:.2f}")
            elif key == ord("]"):  # IR 확대
                ir_scale = min(4.0, ir_scale + 0.1)
                print(f"[Receiver] IR scale: {ir_scale:.2f}")
            elif key == ord("{"):  # RGB_det 축소
                rgb_scale = max(0.1, rgb_scale - 0.1)
                print(f"[Receiver] RGB_det scale: {rgb_scale:.2f}")
            elif key == ord("}"):  # RGB_det 확대
                rgb_scale = min(4.0, rgb_scale + 0.1)
                print(f"[Receiver] RGB_det scale: {rgb_scale:.2f}")
            elif key == ord("1"):  # IR 90도 회전
                ir_rot = (ir_rot + 90) % 360
                print(f"[Receiver] IR rotate: {ir_rot} deg")
            elif key == ord("4"):  # RGB_det 90도 회전
                rgb_rot = (rgb_rot + 90) % 360
                print(f"[Receiver] RGB_det rotate: {rgb_rot} deg")
            elif key == ord("0"):  # 회전 초기화
                ir_rot = 0
                rgb_rot = 0
                print("[Receiver] Rotation reset (IR/RGB_det 0 deg)")

            if saving:
                save_rgb = _decode_image(images.get("rgb"))
                save_ir16 = _decode_image(images.get("ir16"))
                save_frames(save_rgb, save_ir16, timestamp)

    except KeyboardInterrupt:
        print("\n[Receiver] Stopped by user")
    except Exception as e:
        print(f"[Receiver] Error: {e}")
    finally:
        cv2.destroyAllWindows()
        receiver.close()
        elapsed = time.time() - start_time
        if elapsed > 0:
            print(f"[Receiver] Total frames: {frame_count}, Average FPS: {frame_count / elapsed:.2f}")


if __name__ == "__main__":
    receive_and_display(host="0.0.0.0", port=9999)
