"""
Smart Canteen Billing Assistant: End-to-End Acceptance Criteria Test Suite (tests/test_acceptance_criteria.py)
==============================================================================================================

Purpose:
--------
Verification test suite validating critical system acceptance requirements:
1. Input Format Robustness: Verifies error-free loading and EXIF sanitization for JPG, PNG, and WEBP formats.
2. Multi-Item Samosa Detection: Confirms detection of two distinct samosas with Qty=2, Rs 40, 520 kcal.
3. Multi-Item Samosa + Chai Detection: Confirms separate bounding boxes and billing lines for mixed food items.
4. Non-Food False Positive Suppression: Confirms empty wooden canteen tables produce 0 billable food items.

Usage:
------
    python tests/test_acceptance_criteria.py
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Any
from PIL import Image, ImageOps
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import FoodPredictor, load_menu
from app.detector import detect_food_items
from app.billing import aggregate_detections, merge_into_bill


def load_and_sanitize_image(image_file_or_data: Any) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Sanitizes raw uploaded or loaded image data by applying EXIF transposition and converting to RGB.

    Args:
        image_file_or_data: File-like object, byte buffer, or path to image file.

    Returns:
        Tuple[Optional[Image.Image], Optional[str]]: (Sanitized RGB PIL Image, error_message if failed).
    """
    try:
        img = Image.open(image_file_or_data)
        img = ImageOps.exif_transpose(img)
        rgb_img = img.convert("RGB")
        return rgb_img, None
    except Exception as err:
        return None, f"Unable to read image file: {err}"


def test_photo_formats():
    """
    Test 1: Validates that the application gracefully ingests and standardizes
    JPG, PNG, and WEBP food photos.
    """
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


def test_two_samosas(predictor: FoodPredictor, menu_df):
    """
    Test 2: Verifies multi-item detection and quantity aggregation on an image
    containing exactly two samosas.
    """
    print("\n--- TEST 2: Two Samosas on Table ---")
    img_path = PROJECT_ROOT / "outputs" / "test_two_samosas.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    # Detect items
    det_res = detect_food_items(pil_img, predictor, threshold=0.50)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")
    for d in det_res["detections"]:
        print(f"    #{d['item_number']}: {d['class_name']} ({d['confidence']*100:.1f}%) at {d['box']}")

    # Aggregate bill
    bill_summary = aggregate_detections(det_res["detections"], menu_df=menu_df)
    print(f"  Aggregated bill lines: {bill_summary['items']}")
    print(f"  Total Qty: {bill_summary['total_qty']}, Total Price: Rs. {bill_summary['total_price']}, Total kcal: {bill_summary['total_kcal']}")

    assert bill_summary["total_qty"] == 2, f"Expected total Qty 2, got {bill_summary['total_qty']}"
    assert len(bill_summary["items"]) == 1, f"Expected 1 unique class (samosa), got {len(bill_summary['items'])}"
    assert bill_summary["items"][0]["class"] == "samosa", f"Expected samosa, got {bill_summary['items'][0]['class']}"
    assert bill_summary["items"][0]["qty"] == 2, f"Expected samosa Qty 2, got {bill_summary['items'][0]['qty']}"
    assert bill_summary["total_price"] == 40, f"Expected Rs. 40, got {bill_summary['total_price']}"
    assert bill_summary["total_kcal"] == 520, f"Expected 520 kcal (from menu.csv), got {bill_summary['total_kcal']}"
    print("  [PASS] Exactly 2 samosas detected with Qty=2, Rs. 40, 520 kcal!")


def test_samosa_plus_chai(predictor: FoodPredictor, menu_df):
    """
    Test 3: Verifies multi-item detection on a mixed tray containing 1 samosa and 1 chai cup.
    """
    print("\n--- TEST 3: Samosa + Chai ---")
    img_path = PROJECT_ROOT / "outputs" / "test_samosa_chai.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    det_res = detect_food_items(pil_img, predictor, threshold=0.50)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")
    for d in det_res["detections"]:
        print(f"    #{d['item_number']}: {d['class_name']} ({d['confidence']*100:.1f}%) at {d['box']}")

    bill_summary = aggregate_detections(det_res["detections"], menu_df=menu_df)
    print(f"  Aggregated bill lines: {bill_summary['items']}")
    print(f"  Total Qty: {bill_summary['total_qty']}, Total Price: Rs. {bill_summary['total_price']}")

    assert bill_summary["total_qty"] == 2, f"Expected total Qty 2, got {bill_summary['total_qty']}"
    classes_detected = {item["class"] for item in bill_summary["items"]}
    assert "samosa" in classes_detected, f"Expected samosa in bill, got {classes_detected}"
    assert "chai" in classes_detected, f"Expected chai in bill, got {classes_detected}"
    print("  [PASS] Successfully detected Samosa + Chai as 2 distinct bill lines!")


def test_nonfood_table(predictor: FoodPredictor, menu_df):
    """
    Test 4: Verifies financial safety rejection on non-food background surfaces (empty tables, trays).
    """
    print("\n--- TEST 4: Non-Food Table Rejection ---")
    img_path = PROJECT_ROOT / "outputs" / "test_table_nonfood.jpg"
    assert img_path.exists(), f"Missing test image: {img_path}"
    with open(img_path, "rb") as f:
        pil_img, _ = load_and_sanitize_image(f)

    det_res = detect_food_items(pil_img, predictor, threshold=0.60)
    print(f"  Raw detections count: {det_res['total_detected']}, Ignored candidates: {det_res['ignored_count']}")

    bill_summary = aggregate_detections(det_res["detections"], menu_df=menu_df)
    print(f"  Aggregated bill lines: {bill_summary['items']}")
    print(f"  Total Qty: {bill_summary['total_qty']}, Total Price: Rs. {bill_summary['total_price']}")

    assert det_res["total_detected"] == 0, f"Expected 0 detections on empty table, got {det_res['total_detected']}"
    assert bill_summary["total_qty"] == 0, f"Expected 0 billable items, got {bill_summary['total_qty']}"
    print("  [PASS] Empty wooden table correctly produced 0 billable items!")


def main():
    """Runs the complete acceptance criteria test suite."""
    print("=" * 58)
    print("RUNNING SMART CANTEEN ACCEPTANCE CRITERIA TESTS")
    print("=" * 58)

    test_photo_formats()

    print("\nLoading FoodPredictor...")
    predictor = FoodPredictor(backend="torch", arch_name="mobilenetv3", threshold=0.60)
    menu_df = load_menu()

    test_two_samosas(predictor, menu_df)
    test_samosa_plus_chai(predictor, menu_df)
    test_nonfood_table(predictor, menu_df)

    print("\n" + "=" * 58)
    print("[ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY!]")
    print("=" * 58)


if __name__ == "__main__":
    main()
