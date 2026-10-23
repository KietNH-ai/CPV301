"""
fh4_viewonly.py — Test pipeline THỜI GIAN THỰC trên Forza Horizon 4 (VIEW-ONLY).

Chỉ ĐỌC màn hình (mss) -> detect làn -> vẽ kết quả lên cửa sổ overlay.
KHÔNG gửi bất kỳ lệnh điều khiển nào vào game -> an toàn, không đụng anti-cheat.

Yêu cầu:
  - Windows, FH4 đang chạy ở chế độ WINDOWED (hoặc borderless), góc nhìn bumper cam.
  - Chạy HOÀN TOÀN OFFLINE (không kết nối online).
  - pip install mss opencv-python (đã có trong requirements.txt).
  - Nếu dùng --detector yolo/proposed: cần ultralytics + weights yolo11n-seg.pt.

Usage:
    # Vùng game (left, top, width, height). Hough chạy ~60-90 FPS, không cần GPU.
    python scripts/fh4_viewonly.py --detector hough --region 0 0 1280 720

    # YOLO đã fine-tune trên TuSimple
    python scripts/fh4_viewonly.py --detector yolo --weights runs/segment/train/weights/best.pt

    # Ghi lại video kết quả
    python scripts/fh4_viewonly.py --detector hough --save data/output/fh4_viewonly.mp4

Phím: q / ESC để thoát.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.model_config import build_pipeline, GENERAL  # noqa: E402
from src.capture import ScreenCapture  # noqa: E402


def draw_hud(frame, detector_name: str, fps: float, n_lanes: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    text = f"{detector_name.upper()}  |  FPS {fps:5.1f}  |  lanes {n_lanes}  |  VIEW-ONLY (q/ESC thoat)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["hough", "lanenet", "yolo", "proposed"], default="hough")
    ap.add_argument("--weights", default=None,
                    help="Đường dẫn weights YOLO (chỉ dùng với --detector yolo/proposed)")
    ap.add_argument("--conf", type=float, default=0.5, help="Ngưỡng confidence YOLO")
    ap.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "W", "H"), default=None,
                    help="Vùng game: left top width height. Mặc định = toàn màn hình.")
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--save", default=None, help="Ghi video kết quả ra file .mp4")
    ap.add_argument("--no-display", action="store_true", help="Không hiện cửa sổ (chỉ chạy + ghi)")
    args = ap.parse_args()

    print("=" * 64)
    print(" FH4 VIEW-ONLY TEST — chỉ đọc màn hình, KHÔNG điều khiển xe")
    print(" ⚠️  Chạy HOÀN TOÀN OFFLINE. Dùng bumper cam. q/ESC để thoát.")
    print("=" * 64)

    detector_cfg = {}
    if args.detector in ("yolo", "proposed") and args.weights:
        key = "model_path" if args.detector == "yolo" else "yolo_weights"
        detector_cfg[key] = args.weights
        detector_cfg["confidence_threshold"] = args.conf

    print(f"Đang dựng pipeline ({args.detector})... thiết bị = {GENERAL.default_device}")
    pipeline = build_pipeline(args.detector, detector_cfg=detector_cfg)

    writer = None
    with ScreenCapture(monitor=args.monitor, region=tuple(args.region) if args.region else None) as cap:
        geom = cap.capture_geometry
        print(f"Vùng chụp: {geom}")
        try:
            while True:
                t0 = time.monotonic()
                frame = cap.grab()
                annotated = pipeline.process_frame(frame)
                fps = 1.0 / max(time.monotonic() - t0, 1e-6)

                # Đếm số làn (nếu detector trả masks) — chỉ cho HUD
                n_lanes = 0
                try:
                    _, masks = pipeline.detector.detect(frame) if pipeline.detector else (None, [])
                    n_lanes = len(masks or [])
                except Exception:
                    pass

                annotated = draw_hud(annotated, args.detector, fps, n_lanes)

                if args.save:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
                        )
                    writer.write(annotated)

                if not args.no_display:
                    cv2.imshow("FH4 View-Only Lane Detection (q/ESC thoat)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
        except KeyboardInterrupt:
            print("\nDừng bởi người dùng.")
        finally:
            if writer is not None:
                writer.release()
            if not args.no_display:
                cv2.destroyAllWindows()
            print(f"FPS trung bình (capture): {cap.fps:.1f}")
            if args.save:
                print(f"Đã ghi video: {args.save}")


if __name__ == "__main__":
    main()
