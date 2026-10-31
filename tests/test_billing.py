"""
Unit tests for app/billing.py
Smart Canteen Billing Assistant

Test Acceptance Criteria:
1. 2x samosa (20 Rs, 250 kcal) gives Qty 2, Rs 40, 500 kcal.
2. samosa + chai (10 Rs, 20 kcal) gives Qty 2, Rs 30, 270 kcal.
3. not_food is properly excluded from the bill.
4. Receipts and CSV exports format correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add app and root to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

try:
    from billing import (
        aggregate_detections,
        merge_into_bill,
        generate_receipt_text,
        generate_receipt_csv
    )
except ImportError:
    from app.billing import (
        aggregate_detections,
        merge_into_bill,
        generate_receipt_text,
        generate_receipt_csv
    )


def test_two_samosas():
    """Test 1: 2x samosa (20 Rs, 250 kcal) gives Qty 2, Rs 40, 500 kcal."""
    fake_menu = pd.DataFrame([
        {"class": "samosa", "price": 20, "kcal": 250},
        {"class": "chai", "price": 10, "kcal": 20},
    ])

    # Two detections of samosa
    detections = [
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.88},
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.91},
    ]

    result = aggregate_detections(detections, menu_df=fake_menu)

    assert result["total_qty"] == 2, f"Expected Qty 2, got {result['total_qty']}"
    assert result["total_price"] == 40, f"Expected Rs 40, got {result['total_price']}"
    assert result["total_kcal"] == 500, f"Expected 500 kcal, got {result['total_kcal']}"

    items = result["items"]
    assert len(items) == 1
    assert items[0]["class"] == "samosa"
    assert items[0]["qty"] == 2
    assert items[0]["price"] == 20
    assert items[0]["total_price"] == 40
    assert items[0]["kcal"] == 250
    assert items[0]["total_kcal"] == 500
    print("[PASS] test_two_samosas: 2x samosa gives Qty 2, Rs 40, 500 kcal.")


def test_samosa_plus_chai():
    """Test 2: samosa + chai (samosa: 20 Rs, 250 kcal; chai: 10 Rs, 20 kcal) gives Qty 2, Rs 30, 270 kcal."""
    fake_menu = pd.DataFrame([
        {"class": "samosa", "price": 20, "kcal": 250},
        {"class": "chai", "price": 10, "kcal": 20},
    ])

    detections = [
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.92},
        {"class_raw": "chai", "class_name": "Chai", "confidence": 0.85},
    ]

    result = aggregate_detections(detections, menu_df=fake_menu)

    assert result["total_qty"] == 2, f"Expected total Qty 2, got {result['total_qty']}"
    assert result["total_price"] == 30, f"Expected total Rs 30, got {result['total_price']}"
    assert result["total_kcal"] == 270, f"Expected total 270 kcal, got {result['total_kcal']}"
    assert len(result["items"]) == 2, "Expected 2 separate bill lines"

    # Verify line items
    item_dict = {item["class"]: item for item in result["items"]}
    assert item_dict["samosa"]["qty"] == 1
    assert item_dict["samosa"]["total_price"] == 20
    assert item_dict["samosa"]["total_kcal"] == 250

    assert item_dict["chai"]["qty"] == 1
    assert item_dict["chai"]["total_price"] == 10
    assert item_dict["chai"]["total_kcal"] == 20
    print("[PASS] test_samosa_plus_chai: samosa + chai gives Qty 2, Rs 30, 270 kcal.")


def test_not_food_excluded():
    """Test 3: 'not_food' is properly ignored and not billed."""
    fake_menu = pd.DataFrame([
        {"class": "samosa", "price": 20, "kcal": 250},
        {"class": "not_food", "price": 0, "kcal": 0},
    ])

    detections = [
        {"class_raw": "not_food", "class_name": "Not Food", "confidence": 0.95},
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.80},
    ]

    result = aggregate_detections(detections, menu_df=fake_menu)
    assert result["total_qty"] == 1
    assert result["total_price"] == 20
    assert len(result["items"]) == 1
    assert result["items"][0]["class"] == "samosa"
    print("[PASS] test_not_food_excluded: 'not_food' properly excluded.")


def test_merge_into_bill():
    """Test 4: Merging items updates quantities and recalculates totals correctly."""
    bill = [
        {"class": "samosa", "name": "Samosa", "qty": 1, "price": 20, "kcal": 250, "total_price": 20, "total_kcal": 250}
    ]
    new_items = [
        {"class": "samosa", "name": "Samosa", "qty": 2, "price": 20, "kcal": 250},
        {"class": "chai", "name": "Chai", "qty": 1, "price": 10, "kcal": 20}
    ]
    merged = merge_into_bill(bill, new_items)

    assert len(merged) == 2
    samosa_line = next(i for i in merged if i["class"] == "samosa")
    chai_line = next(i for i in merged if i["class"] == "chai")

    assert samosa_line["qty"] == 3
    assert samosa_line["total_price"] == 60
    assert samosa_line["total_kcal"] == 750

    assert chai_line["qty"] == 1
    assert chai_line["total_price"] == 10
    assert chai_line["total_kcal"] == 20
    print("[PASS] test_merge_into_bill: Successfully merged lines and updated quantities.")


def test_receipt_generation():
    """Test 5: Receipt text and CSV export generation."""
    bill = [
        {"class": "samosa", "name": "Samosa", "qty": 2, "price": 20, "kcal": 250, "total_price": 40, "total_kcal": 500},
        {"class": "chai", "name": "Chai", "qty": 1, "price": 15, "kcal": 80, "total_price": 15, "total_kcal": 80}
    ]
    text = generate_receipt_text(bill, "BILL-TEST-001")
    assert "Samosa" in text
    assert "Chai" in text
    assert "Rs. 55" in text

    csv_data = generate_receipt_csv(bill)
    assert "Samosa,samosa,2,20,40,250,500" in csv_data
    print("[PASS] test_receipt_generation: Text and CSV receipts generated correctly.")


if __name__ == "__main__":
    test_two_samosas()
    test_samosa_plus_chai()
    test_not_food_excluded()
    test_merge_into_bill()
    test_receipt_generation()
    print("\n[ALL TESTS PASSED] All billing unit tests completed successfully!")
