#!/usr/bin/env python3
"""
Real Photos Test & Evaluation Suite (tests/evaluate_real_photos.py)

Runs all test images in tests/real_photos/ against tests/real_photos/expected.csv:
- Compares expected class counts with detected class counts.
- Prints: filename | expected | detected | PASS/FAIL
- Computes overall pass rate across real-world photos.
- Outputs detailed debug proposals and classification breakdowns for failed images.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
from PIL import Image, ImageOps

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predictor import FoodPredictor, load_menu
from app.detector import detect_food_items
from app.billing import aggregate_detections


def load_expected_csv(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Parses expected.csv with columns: filename, class, count.
    Returns: {filename: {class_raw: count}}
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected file not found: {csv_path}")

    expected = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = [h.strip().lower() for h in lines[0].split(",")]
    fn_idx = header.index("filename")
    cls_idx = header.index("class")
    cnt_idx = header.index("count")

    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        fname = parts[fn_idx]
        cname = parts[cls_idx].lower().replace(" ", "_")
        count = int(parts[cnt_idx])

        if fname not in expected:
            expected[fname] = {}
        expected[fname][cname] = count

    return expected


def format_counts(counts_dict: Dict[str, int]) -> str:
    """Formats {samosa: 1, chai: 1} as 'chai: 1, samosa: 1' or 'none'."""
    if not counts_dict:
        return "none"
    items = sorted(counts_dict.items())
    return ", ".join(f"{k}: {v}" for k, v in items)


def evaluate_real_photos(
    real_photos_dir: Path,
    expected_csv_path: Path,
    threshold: float = 0.50,
    save_debug_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs evaluation on all images in real_photos_dir against expected_csv_path."""
    print("=" * 80)
    print("SMART CANTEEN — REAL PHOTOS TEST SUITE")
    print(f"Directory: {real_photos_dir}")
    print(f"Ground Truth: {expected_csv_path}")
    print(f"Safety Threshold: {threshold:.2f}")
    print("=" * 80)

    expected_data = load_expected_csv(expected_csv_path)

    # Initialize predictor
    print("\nLoading FoodPredictor (MobileNetV3)...")
    predictor = FoodPredictor(backend="torch", arch_name="mobilenetv3", threshold=threshold)
    menu_df = load_menu()

    # Find image files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([
        f for f in real_photos_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ])

    if not image_files:
        print(f"No test images found in {real_photos_dir}!")
        return {"total": 0, "passed": 0, "pass_rate": 0.0}

    results = []
    failed_details = []

    print("\nEvaluating images...\n")
    header_line = f"{'Filename':<24} | {'Expected':<22} | {'Detected':<22} | {'Result':<10}"
    sep_line = "-" * len(header_line)
    print(header_line)
    print(sep_line)

    for img_path in image_files:
        fname = img_path.name
        img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")

        # Run multi-item detector
        det_res = detect_food_items(img_pil, predictor, threshold=threshold)
        detections = det_res["detections"]
        proposal_method = det_res["proposal_method"]

        # Aggregate detected item counts
        detected_counts = Counter()
        for d in detections:
            detected_counts[d["class_raw"]] += 1
        detected_dict = dict(detected_counts)

        # Expected counts
        exp_dict = expected_data.get(fname, {})

        # Compare
        is_pass = (detected_dict == exp_dict)
        status_str = "PASS" if is_pass else "FAIL"

        exp_str = format_counts(exp_dict)
        det_str = format_counts(detected_dict)

        print(f"{fname:<24} | {exp_str:<22} | {det_str:<22} | {status_str:<10}")

        results.append({
            "filename": fname,
            "expected": exp_dict,
            "detected": detected_dict,
            "pass": is_pass,
            "proposal_method": proposal_method,
            "detections": detections,
            "all_candidates": det_res["all_candidates"]
        })

        if not is_pass:
            failed_details.append({
                "filename": fname,
                "expected": exp_dict,
                "detected": detected_dict,
                "proposal_method": proposal_method,
                "candidates": det_res["all_candidates"]
            })

        # Save debug view image if directory requested
        if save_debug_dir:
            save_debug_dir.mkdir(parents=True, exist_ok=True)
            dbg_path = save_debug_dir / f"debug_{img_path.stem}.jpg"
            Image.fromarray(det_res["image_debug"]).save(dbg_path)

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    pass_rate = (passed / total) * 100.0 if total > 0 else 0.0

    print(sep_line)
    print(f"\nOverall Result: {passed} / {total} Passed ({pass_rate:.1f}% pass rate)")

    if failed_details:
        print("\n" + "=" * 80)
        print("FAILED IMAGES — DETAILED DEBUG VIEW BREAKDOWN")
        print("=" * 80)
        for fail in failed_details:
            print(f"\n[FAIL] Image: {fail['filename']}")
            print(f"  Proposal Engine Used: {fail['proposal_method']}")
            print(f"  Expected: {format_counts(fail['expected'])}")
            print(f"  Detected: {format_counts(fail['detected'])}")
            print("  Candidate Proposals Evaluated:")
            for c in fail["candidates"]:
                print(
                    f"    - Box #{c['index']} {c['box']}: "
                    f"Predicted '{c['class_name']}' ({c['confidence']*100:.1f}%) | "
                    f"Status: {c['status']} | "
                    f"Reason: {c['reject_reason']}"
                )
                if c["top_predictions"]:
                    top_strs = [f"{tp['class_raw']} ({tp['confidence']*100:.1f}%)" for tp in c["top_predictions"][:3]]
                    print(f"      Top predictions: {', '.join(top_strs)}")

    print("=" * 80)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "results": results,
        "failed_details": failed_details
    }


def main():
    photos_dir = PROJECT_ROOT / "tests" / "real_photos"
    csv_path = photos_dir / "expected.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "tests" / "expected.csv"

    out_dir = PROJECT_ROOT / "outputs"
    res = evaluate_real_photos(photos_dir, csv_path, threshold=0.50, save_debug_dir=out_dir)

    # Return exit code: 0 if all passed, 1 if any failed
    sys.exit(0 if res["passed"] == res["total"] and res["total"] > 0 else 1)


if __name__ == "__main__":
    main()
