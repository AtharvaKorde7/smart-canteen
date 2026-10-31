"""
Smart Canteen Billing Assistant: Food Image Recognition
Real-Time OpenCV Live Camera Demo (app/live_demo.py)

Key Features:
1. Low-latency camera pipeline (640x480 resolution, buffer size = 1 to eliminate frame lag).
2. Non-blocking asynchronous inference worker thread (keeps UI rendering at 30-60 FPS).
3. Multi-Item Detection Pipeline: region proposals + crop classification with MobileNetV3 + NMS.
4. Live per-class counts, bounding boxes with badges, and subtotal calculation.
5. Quantity-based billing: [SPACE] adds ALL currently detected food items to running bill.
6. Bill merging: increments quantities for existing dishes and computes total price & calories.
7. Hotkeys: [SPACE] add items, [C] clear bill, [S] save snapshot & receipt, [D] debug, [G] Grad-CAM, [M] mirror, [Q] quit.

Authors: Capstone Project Team (Ramdeobaba University)
"""

import sys
import os
import time
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# Add project root and app directory to sys.path so modules import cleanly
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from app.predictor import FoodPredictor, load_class_names, load_menu, compute_gradcam
    from app.detector import detect_food_items, draw_detections
    from app.billing import aggregate_detections, merge_into_bill, generate_receipt_text
except ImportError:
    from predictor import FoodPredictor, load_class_names, load_menu, compute_gradcam
    from detector import detect_food_items, draw_detections
    from billing import aggregate_detections, merge_into_bill, generate_receipt_text


def parse_args():
    """Parses command-line arguments for live webcam demo."""
    parser = argparse.ArgumentParser(
        description="Smart Canteen Real-Time Food Recognition Live Demo (Multi-Item Detection)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "onnx"],
        help="Inference engine: 'torch' (PyTorch) or 'onnx' (ONNX Runtime). Default: torch",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenetv3",
        choices=["mobilenetv3", "efficientnet_b0"],
        help="Model architecture for PyTorch backend. Default: mobilenetv3",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional custom path to .pth or .onnx model file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence rejection threshold (0.0 to 1.0). Default: 0.60",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam camera index. Default: 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cuda' or 'cpu'. Default: auto-detects GPU",
    )
    parser.add_argument(
        "--predict-interval",
        type=int,
        default=4,
        help="Submit frames to inference worker every N frames. Default: 4",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        default=True,
        help="Mirror video horizontally for natural interaction. Default: True",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode at startup: logs proposals and candidates.",
    )
    return parser.parse_args()


# ==============================================================================
# ASYNC MULTI-ITEM INFERENCE WORKER THREAD (NON-BLOCKING)
# ==============================================================================
class AsyncMultiItemPredictor:
    """
    Decouples multi-item food detection and classification from OpenCV's video rendering loop.
    - Captures and displays video at the camera's native frame rate (30-60 FPS).
    - Runs multi-item region proposals and crop classification in the background.
    - Maintains thread-safe access to the latest detection and billing aggregation results.
    """

    def __init__(self, predictor: FoodPredictor, threshold: float = 0.60):
        self.predictor = predictor
        self.threshold = threshold
        self.lock = threading.Lock()
        self.pending_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[Dict[str, Any]] = None
        self.running = True
        self.debug_flag = False

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def submit_frame(self, frame_rgb: np.ndarray, debug: bool = False):
        """Submits a fresh RGB frame to the worker thread without blocking."""
        with self.lock:
            self.pending_frame = frame_rgb.copy()
            if debug:
                self.debug_flag = True

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Safely retrieves the most recent multi-item detection result."""
        with self.lock:
            return self.latest_result

    def _worker_loop(self):
        """Worker loop executing multi-item detection asynchronously."""
        while self.running:
            frame_to_process = None
            is_debug = False
            with self.lock:
                if self.pending_frame is not None:
                    frame_to_process = self.pending_frame
                    self.pending_frame = None
                    is_debug = self.debug_flag
                    self.debug_flag = False

            if frame_to_process is not None:
                try:
                    # Run region-proposal + crop classification pipeline
                    det_result = detect_food_items(
                        frame_to_process,
                        self.predictor,
                        threshold=self.threshold,
                        nms_iou_threshold=0.35,
                        max_proposals=20,
                    )
                    # Aggregate detected items by class for quantity-based billing
                    agg_result = aggregate_detections(
                        det_result["detections"],
                        self.predictor.menu_df
                    )
                    combined = {
                        "detections": det_result["detections"],
                        "total_detected": det_result["total_detected"],
                        "ignored_count": det_result["ignored_count"],
                        "aggregated": agg_result["items"],
                        "total_qty": agg_result["total_qty"],
                        "total_price": agg_result["total_price"],
                        "total_kcal": agg_result["total_kcal"],
                    }
                    with self.lock:
                        self.latest_result = combined

                    if is_debug:
                        print(f"\n[DEBUG] Detected {combined['total_detected']} items "
                              f"(Ignored: {combined['ignored_count']}):")
                        for d in combined["detections"]:
                            print(f"  • {d['class_name']} ({d['confidence']*100:.1f}%) "
                                  f"at box {d['box']} - Rs. {d['price']}")
                except Exception as err:
                    print(f"[AsyncPredictor Error] {err}")
            else:
                time.sleep(0.005)

    def stop(self):
        """Stops the worker thread cleanly."""
        self.running = False
        self.worker.join(timeout=1.0)


# ==============================================================================
# HUD DRAWING HELPERS
# ==============================================================================
def draw_hud_banner(frame, x, y, w, h, bg_color=(20, 20, 20), alpha=0.75):
    """Draws a semi-transparent dark banner behind text for high readability."""
    import cv2
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub_img = frame[y1:y2, x1:x2]
    rect = np.full_like(sub_img, bg_color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub_img, 1.0 - alpha, 0, frame[y1:y2, x1:x2])


def draw_corner_brackets(frame, x1, y1, x2, y2, color, length=18, thickness=2):
    """Draws targeting brackets on the corners of a bounding box."""
    import cv2
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


# Palette for multiple bounding boxes (BGR format for OpenCV)
BOX_PALETTE_BGR = [
    (235, 99, 37),    # Blue
    (129, 185, 16),   # Emerald Green
    (11, 158, 245),   # Amber
    (68, 68, 239),    # Crimson Red
    (246, 92, 139),   # Purple
    (153, 72, 236),   # Magenta Pink
]


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================
def main():
    args = parse_args()

    try:
        import cv2
    except ImportError:
        print("\n[ERROR] OpenCV is not installed. Please run: pip install opencv-python\n")
        sys.exit(1)

    print("=" * 70)
    print("🍽️  SMART CANTEEN LIVE CAMERA (MULTI-ITEM DETECTION)")
    print("=" * 70)
    print(f"Backend Engine   : {args.backend.upper()}")
    print(f"Architecture     : {args.arch}")
    print(f"Safety Threshold : {args.threshold * 100:.0f}%")
    print(f"Camera Index     : {args.camera}")
    print(f"Resolution       : 640x480 (Low Latency)")
    print(f"Buffer Size      : 1 (Zero Stale Frames)")
    print(f"Multi-Item Mode  : Enabled (Region Proposals + Crop Classification + NMS)")

    # Initialize predictor
    try:
        predictor = FoodPredictor(
            backend=args.backend,
            arch_name=args.arch,
            model_path=args.model_path,
            threshold=args.threshold,
            window_size=10,
            device_str=args.device,
        )
        print(f"Model Loaded     : {predictor.model_path.name}")
        print(f"Classes Supported: 14 Classes (13 Dishes + 'not_food')")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize predictor: {e}")
        sys.exit(1)

    # Launch non-blocking background predictor
    async_predictor = AsyncMultiItemPredictor(predictor, threshold=args.threshold)

    # Open webcam
    print(f"\nConnecting to webcam index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not open webcam at index {args.camera}!")
        print("Please check that camera privacy permissions are enabled and no other app is using it.\n")
        async_predictor.stop()
        sys.exit(1)

    print("[PASS] Webcam opened successfully!")
    print("\nInteractive Hotkeys:")
    print("  [SPACE] Add ALL currently detected food items to running bill")
    print("  [C]     Clear running bill to Rs. 0")
    print("  [D]     Debug Mode: print candidate proposals & detection details")
    print("  [G]     Grad-CAM: compute explainability heatmap for detected item")
    print("  [M]     Toggle mirror view")
    print("  [S]     Save camera snapshot & generate bill receipt file")
    print("  [Q]     Quit live demo")
    print("-" * 70)

    # State variables
    bill_items: List[Dict[str, Any]] = []
    last_action_msg = ""
    last_action_time = 0.0
    mirror_mode = args.mirror
    debug_mode = args.debug

    # FPS counter variables
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    # Screenshot & output directory
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps > 0 else (1.0 / elapsed)
            prev_time = curr_time

            # Optional horizontal mirror flip for natural webcam interaction
            if mirror_mode:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            # Convert BGR frame to RGB for PyTorch model inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Submit frame to background multi-item worker every Nth frame (non-blocking)
            if frame_count % args.predict_interval == 0:
                async_predictor.submit_frame(frame_rgb, debug=debug_mode)

            # Retrieve latest multi-item detection result
            latest_res = async_predictor.get_latest_result()

            # Draw all detected bounding boxes
            if latest_res is not None and latest_res.get("detections"):
                detections = latest_res["detections"]
                for i, det in enumerate(detections):
                    bx, by, bw, bh = det["box"]
                    color = BOX_PALETTE_BGR[i % len(BOX_PALETTE_BGR)]

                    # Draw bounding box rectangle and corner accents
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
                    draw_corner_brackets(frame, bx, by, bx + bw, by + bh, color, length=16, thickness=2)

                    # Badge above box
                    badge_w = max(150, bw)
                    badge_h = 36
                    badge_y = max(42, by - badge_h - 4)
                    draw_hud_banner(frame, bx, badge_y, badge_w, badge_h, bg_color=(15, 15, 15), alpha=0.80)

                    conf_pct = det["confidence"] * 100
                    line1 = f"#{i+1} {det['class_name']} ({conf_pct:.0f}%)"
                    line2 = f"Rs. {det['price']} | {det['kcal']} kcal"

                    cv2.putText(
                        frame, line1, (bx + 6, badge_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, line2, (bx + 6, badge_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 120), 1, cv2.LINE_AA
                    )

            # Draw Left Live Detection Summary Panel
            panel_x, panel_y = 12, 44
            panel_w, panel_h = 210, 115
            draw_hud_banner(frame, panel_x, panel_y, panel_w, panel_h, bg_color=(15, 15, 15), alpha=0.82)

            if latest_res is not None and latest_res.get("aggregated"):
                agg_items = latest_res["aggregated"]
                tot_qty = latest_res["total_qty"]
                tot_p = latest_res["total_price"]
                tot_k = latest_res["total_kcal"]

                cv2.putText(
                    frame, f"LIVE DETECTED ({tot_qty} items):", (panel_x + 8, panel_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 120), 1, cv2.LINE_AA
                )

                line_y = panel_y + 34
                for idx, line in enumerate(agg_items[:3]):
                    txt = f"{line['name']}: {line['qty']}x  (Rs.{line['total_price']})"
                    cv2.putText(
                        frame, txt, (panel_x + 8, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA
                    )
                    line_y += 18

                if len(agg_items) > 3:
                    cv2.putText(
                        frame, f"+ {len(agg_items) - 3} more dish(es)...", (panel_x + 8, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA
                    )
                    line_y += 16

                cv2.putText(
                    frame, f"Total: Rs. {tot_p} | {tot_k} kcal", (panel_x + 8, panel_y + panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 215, 0), 1, cv2.LINE_AA
                )
            else:
                cv2.putText(
                    frame, "LIVE DETECTIONS:", (panel_x + 8, panel_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame, "No food detected", (panel_x + 8, panel_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 240), 1, cv2.LINE_AA
                )
                cv2.putText(
                    frame, "Hold dishes in camera view", (panel_x + 8, panel_y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA
                )

            # Top HUD Bar: Title, Engine, Classes & Live FPS
            top_bar_h = 36
            draw_hud_banner(frame, 0, 0, w, top_bar_h, bg_color=(25, 25, 25), alpha=0.88)
            hud_title = f"Smart Canteen Assistant | 14 Classes (13 Dishes) | FPS: {fps:.1f}"
            cv2.putText(
                frame, hud_title, (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Running Bill Banner in top-right corner
            bill_total_price = sum(item["total_price"] for item in bill_items)
            bill_total_qty = sum(item["qty"] for item in bill_items)
            bill_text = f"Bill: Rs. {bill_total_price} ({bill_total_qty} items)"
            (b_w, _), _ = cv2.getTextSize(bill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.putText(
                frame, bill_text, (w - b_w - 14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 120), 2, cv2.LINE_AA
            )

            # Bottom HUD Bar: Hotkeys
            bottom_bar_h = 30
            draw_hud_banner(frame, 0, h - bottom_bar_h, w, bottom_bar_h, bg_color=(25, 25, 25), alpha=0.88)
            key_guide = "[SPACE] Add All  [C] Clear  [S] Receipt  [D] Debug  [G] Grad-CAM  [M] Mirror  [Q] Quit"
            cv2.putText(
                frame, key_guide, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA
            )

            # Temporary notification toast
            if last_action_msg and (curr_time - last_action_time < 2.5):
                toast_color = (0, 220, 100) if "Added" in last_action_msg else (0, 160, 255)
                draw_hud_banner(frame, w // 2 - 210, h - bottom_bar_h - 40, 420, 32, bg_color=(10, 10, 10), alpha=0.92)
                cv2.putText(
                    frame, last_action_msg, (w // 2 - 195, h - bottom_bar_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, toast_color, 2, cv2.LINE_AA
                )

            # Display composite live frame
            cv2.imshow("Smart Canteen - Real-Time Multi-Item Food Recognition", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            # [Q] or [ESC] -> Quit
            if key in [ord("q"), ord("Q"), 27]:
                print("\n[INFO] Quitting live demo...")
                break

            # [SPACE] -> Add all currently detected food items to running bill
            elif key == 32:
                if latest_res is not None and latest_res.get("aggregated"):
                    new_lines = latest_res["aggregated"]
                    bill_items = merge_into_bill(bill_items, new_lines)
                    tot_added_qty = latest_res["total_qty"]
                    tot_added_price = latest_res["total_price"]
                    last_action_msg = f"Added {tot_added_qty} items (+Rs. {tot_added_price})"
                    last_action_time = curr_time
                    current_grand_total = sum(i["total_price"] for i in bill_items)
                    current_grand_qty = sum(i["qty"] for i in bill_items)
                    print(f"[BILL +] Added {tot_added_qty} item(s) to bill. Grand Total: Rs. {current_grand_total} ({current_grand_qty} items)")
                else:
                    last_action_msg = "Blocked: No food items currently detected!"
                    last_action_time = curr_time
                    print(f"[REJECTED] {last_action_msg}")

            # [C] -> Clear running bill
            elif key in [ord("c"), ord("C")]:
                bill_items.clear()
                last_action_msg = "Bill cleared to Rs. 0"
                last_action_time = curr_time
                print("[BILL RESET] Running bill cleared to Rs. 0.")

            # [D] -> Debug Mode (logs details about proposals and candidate boxes)
            elif key in [ord("d"), ord("D")]:
                if latest_res is not None:
                    print(f"\n[DEBUG] Total Detections: {latest_res['total_detected']} | Ignored: {latest_res['ignored_count']}")
                    for i, d in enumerate(latest_res["detections"]):
                        print(f"  #{i+1} {d['class_name']} ({d['confidence']*100:.1f}%) - Box: {d['box']} - Rs. {d['price']}")
                last_action_msg = "Logged debug details to console."
                last_action_time = curr_time

            # [G] -> On-Demand Grad-CAM for the highest confidence detected food item
            elif key in [ord("g"), ord("G")]:
                print("\n[GRAD-CAM] Computing explainability heatmap for highest confidence item...")
                from PIL import Image
                crop_to_cam = None

                if latest_res is not None and latest_res.get("detections"):
                    # Use crop of highest confidence item
                    top_det = latest_res["detections"][0]
                    bx, by, bw, bh = top_det["box"]
                    crop_rgb = frame_rgb[by:by + bh, bx:bx + bw]
                    if crop_rgb.size > 0:
                        crop_to_cam = Image.fromarray(crop_rgb)
                else:
                    # Fallback to center region
                    c_sz = int(min(h, w) * 0.65)
                    cx1 = (w - c_sz) // 2
                    cy1 = (h - c_sz) // 2
                    crop_to_cam = Image.fromarray(frame_rgb[cy1:cy1 + c_sz, cx1:cx1 + c_sz])

                if crop_to_cam is not None:
                    cam_img, cam_err = compute_gradcam(predictor.model, predictor.target_layer, crop_to_cam)
                    if cam_img is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cam_path = output_dir / f"gradcam_{timestamp}.jpg"
                        cv2.imwrite(str(cam_path), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
                        last_action_msg = f"Saved Grad-CAM: {cam_path.name}"
                        last_action_time = curr_time
                        print(f"[GRAD-CAM] Saved explainability heatmap to: {cam_path}")
                    else:
                        last_action_msg = f"Grad-CAM Error: {cam_err}"
                        last_action_time = curr_time
                        print(f"[GRAD-CAM ERROR] {cam_err}")

            # [M] -> Toggle Mirror Mode
            elif key in [ord("m"), ord("M")]:
                mirror_mode = not mirror_mode
                last_action_msg = f"Mirror Mode: {'ON' if mirror_mode else 'OFF'}"
                last_action_time = curr_time

            # [S] -> Save screenshot and export bill receipt
            elif key in [ord("s"), ord("S")]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = output_dir / f"canteen_snap_{timestamp}.jpg"
                cv2.imwrite(str(snap_path), frame)
                last_action_msg = f"Saved: {snap_path.name}"
                last_action_time = curr_time
                print(f"[SCREENSHOT] Saved screenshot to {snap_path}")

                if bill_items:
                    receipt_text = generate_receipt_text(bill_items, bill_no=f"LIVE-{timestamp}")
                    receipt_path = output_dir / f"canteen_receipt_{timestamp}.txt"
                    with open(receipt_path, "w", encoding="utf-8") as rf:
                        rf.write(receipt_text)
                    print(f"[RECEIPT] Generated printable bill receipt: {receipt_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user keyboard interrupt.")
    finally:
        async_predictor.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[CLEANUP] Camera released and OpenCV windows closed successfully.\n")


if __name__ == "__main__":
    main()
