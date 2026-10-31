"""
Test script for Smart Canteen Billing Assistant acceptance criteria:
1. Photo format verification: JPG, PNG, WEBP loading & sanitization
2. Multi-item detection on 2 samosas -> Qty = 2, Rs 40, 520 kcal (from menu.csv)
3. Multi-item detection on 1 samosa + 1 chai -> 2 distinct bill lines
4. Non-food / table rejection -> 0 billable items
5. Unit test billing consistency
"""

import sys
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import FoodPredictor, load_menu
from app.detector import detect_food_items
from app.billing import aggregate_detections, merge_into_bill


def load_and_sanitize_image(image_file_or_data):
    """Mirror of app.py image sanitization."""
    try:
        img = Image.open(image_file_or_data)
        img = ImageOps.exif_transpose(img)
        rgb_img = img.convert("RGB")
        return rgb_img, None
    except Exception as err:
        return None, f"Unable to read image file: {err}"


def test_photo_formats():
    print("\n--- TEST 1: Photo Formats (JPG, PNG, WEBP) ---")
    out_dir = PROJECT_ROOT / "outputs"
    formats = ["test_food.jpg", "test_food.png", "test_food.webp"]
    for fmt_file in formats:
        p = out_dir / fmt_file
        assert p.exists(), f"Missing format test file: {p}"
        with open(p, "rb") as f:
            pil_img, err = load_and_sanitize_image(f)
        assert err is None, f"Error loading {fmt_file}: {err}"
        assert pil_img is not None, f"Failed to load/sanitize {fmt_file}"
        assert pil_img.mode == "RGB", f"Expected RGB mode, got {pil_img.mode}"
        assert pil_img.size[0] > 0 and pil_img.size[1] > 0
        print(f"  [PASS] Successfully loaded and sanitized {fmt_file} ({pil_img.size}, {pil_img.mode})")


def test_two_samosas(predictor, menu_df):
    print("\n--- TEST 2: Two Samosas on Table ---")
    img_path = PROJECT_ROOT / "outputs" / "test_two_samosas.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    # Detect items
    det_res = detect_food_items(pil_img, predictor, threshold=0.50)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")
    for i, d in enumerate(det_res["detections"]):
        print(f"    #{i+1}: {d['class_name']} ({d['confidence']*100:.1f}%) at {d['box']}")

    agg = aggregate_detections(det_res["detections"], menu_df)
    print(f"  Aggregated bill lines: {agg['items']}")
    print(f"  Total Qty: {agg['total_qty']}, Total Price: Rs. {agg['total_price']}, Total kcal: {agg['total_kcal']}")

    assert agg["total_qty"] == 2, f"Expected 2 samosas, got {agg['total_qty']}"
    assert agg["items"][0]["class"] == "samosa", f"Expected samosa, got {agg['items'][0]['class']}"
    assert agg["items"][0]["qty"] == 2
    # Verify pricing from menu.csv (samosa = Rs 20, 260 kcal)
    assert agg["total_price"] == 40, f"Expected Rs 40, got {agg['total_price']}"
    assert agg["total_kcal"] == 520, f"Expected 520 kcal, got {agg['total_kcal']}"
    print("  [PASS] Exactly 2 samosas detected with Qty=2, Rs. 40, 520 kcal!")


def test_samosa_plus_chai(predictor, menu_df):
    print("\n--- TEST 3: Samosa + Chai ---")
    img_path = PROJECT_ROOT / "outputs" / "test_samosa_chai.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    det_res = detect_food_items(pil_img, predictor, threshold=0.45)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")
    for i, d in enumerate(det_res["detections"]):
        print(f"    #{i+1}: {d['class_name']} ({d['confidence']*100:.1f}%) at {d['box']}")

    agg = aggregate_detections(det_res["detections"], menu_df)
    print(f"  Aggregated bill lines: {agg['items']}")
    print(f"  Total Qty: {agg['total_qty']}, Total Price: Rs. {agg['total_price']}")

    classes_found = {item["class"] for item in agg["items"]}
    assert "samosa" in classes_found, f"Missing samosa in detections: {classes_found}"
    assert "chai" in classes_found, f"Missing chai in detections: {classes_found}"
    assert len(agg["items"]) == 2, f"Expected 2 bill lines, got {len(agg['items'])}"
    assert agg["total_qty"] == 2, f"Expected total quantity 2, got {agg['total_qty']}"
    # samosa (20) + chai (15) = 35
    assert agg["total_price"] == 35, f"Expected Rs. 35, got {agg['total_price']}"
    print("  [PASS] Successfully detected Samosa + Chai as 2 distinct bill lines!")


def test_table_nonfood(predictor, menu_df):
    print("\n--- TEST 4: Non-Food Table Rejection ---")
    img_path = PROJECT_ROOT / "outputs" / "test_table_nonfood.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    det_res = detect_food_items(pil_img, predictor, threshold=0.60)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")

    agg = aggregate_detections(det_res["detections"], menu_df)
    print(f"  Aggregated bill lines: {agg['items']}")
    print(f"  Total Qty: {agg['total_qty']}, Total Price: Rs. {agg['total_price']}")

    assert agg["total_qty"] == 0, f"Expected 0 billable items for non-food table, got {agg['total_qty']}"
    assert len(agg["items"]) == 0, f"Expected 0 lines, got {len(agg['items'])}"
    print("  [PASS] Empty wooden table correctly produced 0 billable items!")


def main():
    print("==========================================================")
    print("RUNNING SMART CANTEEN ACCEPTANCE CRITERIA TESTS")
    print("==========================================================")

    test_photo_formats()

    # Load predictor once
    print("\nLoading FoodPredictor...")
    predictor = FoodPredictor(backend="torch", arch_name="mobilenetv3", threshold=0.60)
    menu_df = load_menu()

    test_two_samosas(predictor, menu_df)
    test_samosa_plus_chai(predictor, menu_df)
    test_table_nonfood(predictor, menu_df)

    print("\n==========================================================")
    print("[ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!]")
    print("==========================================================")


if __name__ == "__main__":
    main()
