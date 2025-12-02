# test_video_inference.py
import cv2
import time
import numpy as np
from detector.tflite import TFLiteWorker
from configs.get_cfg import get_cfg

def test_video_inference(video_path, model_path, labels_path, 
                         use_npu=True, save_output=True):
    """
    비디오 파일 로드 → 프레임별 추론 → 시각화 → 저장
    
    Args:
        video_path: 입력 비디오 경로 (예: 'output.mp4')
        model_path: TFLite 모델 경로
        labels_path: 라벨 파일 경로
        use_npu: NPU 사용 여부
        save_output: 결과 비디오 저장 여부
    """
    
    # ===== 1. 비디오 로드 =====
    pipeline = (
        "filesrc location=output.mp4 ! "
        "qtdemux name=demux "
        "demux.video_0 ! queue ! h264parse ! avdec_h264 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # cap = cv2.VideoCapture('output.mp4', cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # ===== 2. 모델 로드 =====
    print(f"\n🤖 Loading model...")
    
    # TFLiteWorker 초기화 (버퍼 없이 직접 사용)
    cfg = get_cfg()
    worker = TFLiteWorker(
        model_path=model_path,
        labels_path=labels_path,
        input_buf=None,  # 비디오 파일 사용 시 불필요
        output_buf=None,
        use_npu=use_npu,
        delegate_lib=cfg.get('DELEGATE', '/usr/lib/libvx_delegate.so'),
        cpu_threads=2,
        target_fps=0,  # 제한 없음
        target_res=(1280, 720),  # 출력 해상도
        name="VideoTest"
    )
    
    print(f"✅ Model loaded (accel={worker.accel})")
    
    # ===== 3. 출력 비디오 설정 =====
    output_path = None
    video_writer = None
    
    if save_output:
        output_path = video_path.replace('.mp4', '_detected.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, 
            (worker.target_res[0] if worker.target_res else width,
             worker.target_res[1] if worker.target_res else height)
        )
        print(f"💾 Output will be saved to: {output_path}")
    
    # ===== 4. 프레임별 추론 =====
    print(f"\n🔥 Processing frames...\n")
    
    frame_count = 0
    total_time = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 추론 실행
        start_time = time.time()
        scores, boxes_xyxy, classes = worker._infer_once(frame)
        inference_time = (time.time() - start_time) * 1000  # ms
        total_time += inference_time
        
        # 화점 탐지 여부 (class 0 or 1)
        has_fire = any((cls in [0, 1] and score > 0.25) 
                       for cls, score in zip(classes, scores))
        
        if has_fire:
            detection_count += 1
        
        # 바운딩 박스 그리기
        vis_frame = frame.copy()
        from detector.tflite import _draw_boxes
        vis_frame = _draw_boxes(
            vis_frame, boxes_xyxy, classes, scores, 
            worker.labels, thr=0.25
        )
        
        # target_res로 리사이즈
        if worker.target_res:
            vis_frame = cv2.resize(
                vis_frame, 
                (worker.target_res[0], worker.target_res[1]),
                interpolation=cv2.INTER_AREA
            )
        
        # 화면 표시
        display_frame = vis_frame.copy()
        
        # 상태 정보 오버레이
        status_text = f"Frame: {frame_count}/{total_frames}"
        fire_text = "FIRE DETECTED!" if has_fire else "No Fire"
        fps_text = f"FPS: {1000/inference_time:.1f} ({inference_time:.1f}ms)"
        
        cv2.putText(display_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, fire_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 0, 255) if has_fire else (0, 255, 0), 2)
        cv2.putText(display_frame, fps_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Fire Detection', display_frame)
        
        # 비디오 저장
        if video_writer:
            video_writer.write(vis_frame)
        
        # 진행률 출력
        if frame_count % 30 == 0:
            progress = frame_count / total_frames * 100
            avg_fps = 1000 / (total_time / frame_count)
            print(f"Progress: {progress:.1f}% | "
                  f"Avg FPS: {avg_fps:.1f} | "
                  f"Detections: {detection_count}")
        
        # 'q' 키로 중단
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Stopped by user")
            break
    
    # ===== 5. 정리 =====
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # ===== 6. 결과 출력 =====
    print(f"\n📊 Results:")
    print(f"   Processed frames: {frame_count}/{total_frames}")
    print(f"   Total detection count: {detection_count}")
    print(f"   Detection rate: {detection_count/frame_count*100:.1f}%")
    print(f"   Average inference time: {total_time/frame_count:.1f} ms")
    print(f"   Average FPS: {1000/(total_time/frame_count):.1f}")
    
    if save_output and output_path:
        print(f"\n✅ Output saved: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Fire Detection Test')
    parser.add_argument('video', type=str, help='Input video path (e.g., output.mp4)')
    parser.add_argument('--model', type=str, 
                        default='/root/vision-ai-nxp/model/best_full_integer_quant.tflite',
                        help='TFLite model path')
    parser.add_argument('--labels', type=str,
                        default='/root/vision-ai-nxp/model/labels.txt',
                        help='Labels file path')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of NPU')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output video')
    
    args = parser.parse_args()
    
    test_video_inference(
        video_path=args.video,
        model_path=args.model,
        labels_path=args.labels,
        use_npu=not args.cpu,
        save_output=not args.no_save
    )