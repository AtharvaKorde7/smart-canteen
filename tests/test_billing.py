"""
Smart Canteen Billing Assistant: Billing Unit Tests (tests/test_billing.py)
===========================================================================

Purpose:
--------
Automated test suite verifying the arithmetic correctness and business logic
of the canteen billing engine (`app/billing.py`).

Verification Scope:
-------------------
1. Quantity Multipliers: 2x samosa (20 Rs, 250 kcal) -> Qty 2, Rs 40, 500 kcal.
2. Multi-Item Aggregation: samosa + chai -> Qty 2, Rs 30, 270 kcal across 2 distinct lines.
3. Financial Protection: 'not_food' background detections are strictly ignored and not billed.
4. Incremental Bill Merging: Correctly merges new detections into existing running tabs.
5. Export Formatting: Validates ASCII thermal print layout and RFC 4180 CSV generation.

Usage:
------
    python tests/test_billing.py
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
    from app.billing import (
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
    """
    Test 1: Verifies quantity aggregation for multiple instances of the same dish.
    Asserts: 2x samosa (Rs 20, 250 kcal each) yields Qty 2, Rs 40 total, 500 kcal total.
    """
    fake_menu = pd.DataFrame([
        {"class": "samosa", "price": 20, "kcal": 250},
        {"class": "chai", "price": 10, "kcal": 20},
    ])

    # Two distinct detections of samosa
    detections = [
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.88},
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.92},
    ]

    result = aggregate_detections(detections, menu_df=fake_menu)

    assert result["total_qty"] == 2, f"Expected total Qty 2, got {result['total_qty']}"
    assert result["total_price"] == 40, f"Expected total Rs 40, got {result['total_price']}"
    assert result["total_kcal"] == 500, f"Expected total 500 kcal, got {result['total_kcal']}"
    assert len(result["items"]) == 1, "Expected single consolidated line item for samosa"

    item = result["items"][0]
    assert item["class"] == "samosa"
    assert item["qty"] == 2
    assert item["price"] == 20
    assert item["kcal"] == 250
    assert item["total_price"] == 40
    assert item["total_kcal"] == 500
    print("[PASS] test_two_samosas: 2x samosa gives Qty 2, Rs 40, 500 kcal.")


def test_samosa_plus_chai():
    """
    Test 2: Verifies multi-item aggregation for different dishes on the same tray.
    Asserts: samosa (Rs 20, 250 kcal) + chai (Rs 10, 20 kcal) yields Qty 2, Rs 30 total, 270 kcal total.
    """
    fake_menu = pd.DataFrame([
        {"class": "samosa", "price": 20, "kcal": 250},
        {"class": "chai", "price": 10, "kcal": 20},
    ])

    detections = [
        {"class_raw": "samosa", "class_name": "Samosa", "confidence": 0.85},
        {"class_raw": "chai", "class_name": "Chai", "confidence": 0.94},
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
    """
    Test 3: Financial safety rejection test.
    Asserts: 'not_food' background detections (empty tables, trays, cutlery) are completely excluded from the bill.
    """
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
    """
    Test 4: Cashier workflow verification.
    Asserts: Merging newly scanned items increments quantities for existing dishes and creates new lines for new dishes.
    """
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
    """
    Test 5: Receipt export formatting test.
    Asserts: Text format creates correct ASCII thermal layout and CSV format generates standard tabular records.
    """
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


def main():
    """Runs all billing unit test cases."""
    print("=" * 60)
    print("RUNNING SMART CANTEEN BILLING UNIT TESTS")
    print("=" * 60)
    test_two_samosas()
    test_samosa_plus_chai()
    test_not_food_excluded()
    test_merge_into_bill()
    test_receipt_generation()
    print("\n[ALL TESTS PASSED] All billing unit tests completed successfully!\n")


if __name__ == "__main__":
    main()
