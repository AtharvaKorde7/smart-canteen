"""
Smart Canteen Billing Assistant: Billing & Aggregation Module (app/billing.py)

Pure Python functions for:
- Aggregating multi-item detections into quantity-based billing lines
- Calculating line item totals (price, calories) and grand totals
- Merging detections into running bills
- Generating printable receipts (ASCII text, HTML) and downloadable CSVs

Single source of truth: app/menu.csv (unit price, unit kcal).
"""

import csv
import io
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


def create_bill_line(
    class_raw: str,
    display_name: str,
    qty: int,
    unit_price: int,
    unit_kcal: int
) -> Dict[str, Any]:
    """Creates a validated single bill line dictionary."""
    q = max(0, int(qty))
    p = max(0, int(unit_price))
    k = max(0, int(unit_kcal))
    return {
        "class": class_raw,
        "name": display_name,
        "qty": q,
        "price": p,
        "kcal": k,
        "total_price": q * p,
        "total_kcal": q * k
    }


def aggregate_detections(
    detections: List[Dict[str, Any]],
    menu_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Aggregates raw detection boxes by food class.
    Counts multiple occurrences of the same food dish into a single line with Quantity.
    Filters out 'not_food' and items with confidence below safety threshold.

    Args:
        detections: List of detection dicts (each having 'class_raw', optional 'class_name', 'confidence', 'price', 'kcal')
        menu_df: Optional DataFrame loaded from app/menu.csv containing ['class', 'price', 'kcal']

    Returns:
        dict containing:
            'items': List of aggregated line items
            'total_qty': Total quantity of all food items
            'total_price': Grand total price (INR)
            'total_kcal': Grand total calories (kcal)
    """
    counts: Dict[str, int] = {}
    display_names: Dict[str, str] = {}
    prices: Dict[str, int] = {}
    kcals: Dict[str, int] = {}

    for det in detections:
        c_raw = det.get("class_raw", "").strip()
        if not c_raw or c_raw == "not_food":
            continue

        counts[c_raw] = counts.get(c_raw, 0) + 1

        # Resolve display name
        if c_raw not in display_names:
            name = det.get("class_name")
            if not name or name == "No food / Unknown food":
                name = c_raw.replace("_", " ").title()
            display_names[c_raw] = name

        # Resolve price and kcal from menu_df if provided, else from detection dict
        if c_raw not in prices:
            if menu_df is not None and not menu_df.empty:
                match = menu_df[menu_df["class"] == c_raw]
                if not match.empty:
                    prices[c_raw] = int(match.iloc[0]["price"])
                    kcals[c_raw] = int(match.iloc[0]["kcal"])
                else:
                    prices[c_raw] = int(det.get("price", 0))
                    kcals[c_raw] = int(det.get("kcal", 0))
            else:
                prices[c_raw] = int(det.get("price", 0))
                kcals[c_raw] = int(det.get("kcal", 0))

    aggregated_items: List[Dict[str, Any]] = []
    total_qty = 0
    total_price = 0
    total_kcal = 0

    for c_raw, qty in sorted(counts.items()):
        unit_p = prices.get(c_raw, 0)
        unit_k = kcals.get(c_raw, 0)
        line = create_bill_line(
            class_raw=c_raw,
            display_name=display_names[c_raw],
            qty=qty,
            unit_price=unit_p,
            unit_kcal=unit_k
        )
        aggregated_items.append(line)
        total_qty += line["qty"]
        total_price += line["total_price"]
        total_kcal += line["total_kcal"]

    return {
        "items": aggregated_items,
        "total_qty": total_qty,
        "total_price": total_price,
        "total_kcal": total_kcal
    }


def merge_into_bill(
    current_bill: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merges newly detected or added items into the current running bill.
    If an item already exists in the bill, increments its quantity and updates totals.
    """
    updated_bill = [dict(item) for item in current_bill]

    for new_item in new_items:
        c_raw = new_item["class"]
        qty_to_add = new_item.get("qty", 1)
        if qty_to_add <= 0 or c_raw == "not_food":
            continue

        found = False
        for item in updated_bill:
            if item["class"] == c_raw:
                item["qty"] += qty_to_add
                item["total_price"] = item["qty"] * item["price"]
                item["total_kcal"] = item["qty"] * item["kcal"]
                found = True
                break

        if not found:
            line = create_bill_line(
                class_raw=c_raw,
                display_name=new_item.get("name", c_raw.replace("_", " ").title()),
                qty=qty_to_add,
                unit_price=new_item.get("price", 0),
                unit_kcal=new_item.get("kcal", 0)
            )
            updated_bill.append(line)

    return updated_bill


def generate_receipt_text(
    bill_items: List[Dict[str, Any]],
    bill_number: str = "BILL-001",
    timestamp_str: Optional[str] = None
) -> str:
    """Generates a clean text receipt formatted for printing or viewing."""
    if not timestamp_str:
        timestamp_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")

    total_price = sum(item["total_price"] for item in bill_items)
    total_kcal = sum(item["total_kcal"] for item in bill_items)
    total_qty = sum(item["qty"] for item in bill_items)

    lines = [
        "========================================",
        "      SMART CANTEEN BILLING ASSISTANT   ",
        "         Ramdeobaba University          ",
        "========================================",
        f" Receipt No : {bill_number}",
        f" Date & Time: {timestamp_str}",
        "----------------------------------------",
        f" {'Item':<16} {'Qty':>3} {'Rate':>6} {'Amount':>8}",
        "----------------------------------------"
    ]

    for item in bill_items:
        name = item["name"][:16]
        lines.append(f" {name:<16} {item['qty']:>3} {item['price']:>6} {item['total_price']:>8}")

    lines.extend([
        "----------------------------------------",
        f" Total Items: {total_qty}",
        f" Total Calories: {total_kcal} kcal",
        f" GRAND TOTAL   : Rs. {total_price}",
        "========================================",
        "      Thank you for your visit!         ",
        "========================================"
    ])

    return "\n".join(lines)


def generate_receipt_html(
    bill_items: List[Dict[str, Any]],
    bill_number: str = "BILL-001",
    timestamp_str: Optional[str] = None
) -> str:
    """Generates an HTML receipt suitable for browser printing or rendering."""
    if not timestamp_str:
        timestamp_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")

    total_price = sum(item["total_price"] for item in bill_items)
    total_kcal = sum(item["total_kcal"] for item in bill_items)
    total_qty = sum(item["qty"] for item in bill_items)

    rows_html = ""
    for item in bill_items:
        rows_html += f"""
        <tr>
            <td style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">{item['name']}</td>
            <td style="padding: 6px 8px; text-align: center; border-bottom: 1px solid #e5e7eb;">{item['qty']}</td>
            <td style="padding: 6px 8px; text-align: right; border-bottom: 1px solid #e5e7eb;">₹{item['price']}</td>
            <td style="padding: 6px 8px; text-align: right; border-bottom: 1px solid #e5e7eb; font-weight: 600;">₹{item['total_price']}</td>
            <td style="padding: 6px 8px; text-align: right; border-bottom: 1px solid #e5e7eb; color: #6b7280;">{item['total_kcal']} kcal</td>
        </tr>
        """

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px; background: #ffffff;">
        <div style="text-align: center; border-bottom: 2px dashed #d1d5db; padding-bottom: 12px; margin-bottom: 16px;">
            <h2 style="margin: 0 0 4px 0; color: #1e3a8a;">🍽️ Smart Canteen</h2>
            <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">Automated AI Billing Receipt</p>
            <p style="margin: 4px 0 0 0; color: #374151; font-size: 0.85rem;"><strong>Receipt #:</strong> {bill_number} | <strong>Date:</strong> {timestamp_str}</p>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
            <thead>
                <tr style="background: #f3f4f6; color: #374151;">
                    <th style="padding: 8px; text-align: left;">Item</th>
                    <th style="padding: 8px; text-align: center;">Qty</th>
                    <th style="padding: 8px; text-align: right;">Rate</th>
                    <th style="padding: 8px; text-align: right;">Amount</th>
                    <th style="padding: 8px; text-align: right;">Energy</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <div style="margin-top: 16px; border-top: 2px dashed #d1d5db; padding-top: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 1.15rem; font-weight: 700; color: #111827; margin-bottom: 6px;">
                <span>Total Amount:</span>
                <span style="color: #059669;">₹ {total_price}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; color: #4b5563;">
                <span>Total Items: {total_qty}</span>
                <span style="color: #d97706; font-weight: 600;">Total Calories: {total_kcal} kcal</span>
            </div>
        </div>
        <div style="text-align: center; margin-top: 18px; font-size: 0.8rem; color: #9ca3af;">
            Thank you for dining with us!
        </div>
    </div>
    """
    return html


def generate_receipt_csv(bill_items: List[Dict[str, Any]]) -> str:
    """Generates a CSV string representation of the bill."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Item", "Class", "Quantity", "Unit Price (INR)", "Total Price (INR)", "Unit Calories (kcal)", "Total Calories (kcal)"])
    for item in bill_items:
        writer.writerow([
            item["name"],
            item["class"],
            item["qty"],
            item["price"],
            item["total_price"],
            item["kcal"],
            item["total_kcal"]
        ])
    return output.getvalue()
