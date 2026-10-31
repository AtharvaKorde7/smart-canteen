"""
Smart Canteen Billing Assistant: Multi-Item Food Detector (app/detector.py)

Pipeline (Detect-then-Classify):
1. Region Proposal Generation:
   - Visual saliency, adaptive thresholding, and morphological contour detection
   - Sub-splitting of multi-item composite contours (wide/tall containers)
   - Multi-scale spatial partitions (halves, quadrants, center focus, full frame)
2. Crop & Classify:
   - Extracts each candidate proposal crop
   - Converts to PIL Image to guarantee 100% identical preprocessing with MobileNetV3
   - Classifies each crop using the EXISTING MobileNetV3 FoodPredictor
   - Rejects crops classified as 'not_food' or below the safety confidence threshold
3. Disambiguation & Non-Maximum Suppression (NMS):
   - Suppresses composite parent boxes that encompass multiple disjoint child items (prevents 2 items on 1 plate from collapsing into 1)
   - Eliminates redundant overlapping boxes using IoU and containment thresholds
   - Backwards compatible: single dish images fall back smoothly to the full frame / center dish
4. Annotation & Overlay:
   - Renders styled bounding boxes, class labels, confidences, and item indices
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def calculate_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def calculate_containment(innerBox: Tuple[int, int, int, int], outerBox: Tuple[int, int, int, int]) -> float:
    """Calculates what fraction of innerBox is contained inside outerBox."""
    xA = max(innerBox[0], outerBox[0])
    yA = max(innerBox[1], outerBox[1])
    xB = min(innerBox[0] + innerBox[2], outerBox[0] + outerBox[2])
    yB = min(innerBox[1] + innerBox[3], outerBox[1] + outerBox[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    innerArea = innerBox[2] * innerBox[3]
    if innerArea <= 0:
        return 0.0
    return interArea / float(innerArea)


def generate_region_proposals(
    image_rgb: np.ndarray,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.85
) -> List[Tuple[int, int, int, int]]:
    """
    Generates candidate bounding boxes for food items in the image using:
    1. Multi-channel Otsu, adaptive thresholding, and saturation mask contours.
    2. Sub-splitting of wide or tall multi-item contours into candidate sub-boxes.
    3. Geometric multi-scale spatial partitions (halves, quadrants, center focus).
    4. The full image bounding box.
    """
    h, w = image_rgb.shape[:2]
    total_area = float(h * w)
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    proposals: List[Tuple[int, int, int, int]] = []

    # 1. Image preprocessing for contour detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Convert to HSV to detect food colors vs plate/table
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # Combine Otsu grayscale + saturation thresholding
    _, thresh_gray = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_sat = cv2.threshold(saturation, 35, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(thresh_gray, thresh_sat)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.25 <= aspect <= 4.0:
                proposals.append((bx, by, bw, bh))

                # If wide container (e.g. plate with 2 side-by-side items), propose sub-halves
                if bw >= bh * 1.15:
                    half_sub_w = int(bw * 0.55)
                    proposals.append((bx, by, half_sub_w, bh))
                    proposals.append((bx + bw - half_sub_w, by, half_sub_w, bh))
                # If tall container (e.g. 2 items stacked vertically), propose sub-halves
                elif bh >= bw * 1.15:
                    half_sub_h = int(bh * 0.55)
                    proposals.append((bx, by, bw, half_sub_h))
                    proposals.append((bx, by + bh - half_sub_h, bw, half_sub_h))

    # 2. Geometric spatial grid partitions
    half_w = w // 2
    half_h = h // 2

    # Left & Right halves
    proposals.append((0, 0, half_w, h))
    proposals.append((half_w, 0, w - half_w, h))

    # Top & Bottom halves
    proposals.append((0, 0, w, half_h))
    proposals.append((0, half_h, w, h - half_h))

    # 4 Quadrants
    proposals.append((0, 0, half_w, half_h))
    proposals.append((half_w, 0, w - half_w, half_h))
    proposals.append((0, half_h, half_w, h - half_h))
    proposals.append((half_w, half_h, w - half_w, h - half_h))

    # Center focus crop (standard single-dish focus)
    c_sz = int(min(h, w) * 0.70)
    cx = (w - c_sz) // 2
    cy = (h - c_sz) // 2
    proposals.append((cx, cy, c_sz, c_sz))

    # Full frame
    proposals.append((0, 0, w, h))

    # Deduplicate proposals having high IoU
    unique_proposals: List[Tuple[int, int, int, int]] = []
    for p in proposals:
        is_dup = False
        for u in unique_proposals:
            if calculate_iou(p, u) > 0.85:
                is_dup = True
                break
        if not is_dup:
            unique_proposals.append(p)

    return unique_proposals


def detect_food_items(
    image_input: Any,
    predictor,
    threshold: Optional[float] = None,
    nms_iou_threshold: float = 0.28,
    max_proposals: int = 25
) -> Dict[str, Any]:
    """
    Runs multi-item food detection and classification on an input image.

    Args:
        image_input: PIL Image or numpy array (RGB)
        predictor: FoodPredictor instance
        threshold: Confidence threshold (defaults to predictor.threshold)
        nms_iou_threshold: IoU threshold for non-maximum suppression
        max_proposals: Maximum number of proposal regions to evaluate

    Returns:
        Dict containing:
            - 'detections': List of detected items (box, class_raw, class_name, confidence, price, kcal)
            - 'total_detected': Number of valid food items detected
            - 'ignored_count': Number of candidate regions rejected (not_food or low confidence)
            - 'image_annotated': RGB numpy array with bounding boxes and labels drawn
    """
    conf_thresh = float(threshold) if threshold is not None else float(predictor.threshold)

    # 1. Convert input to RGB numpy array and PIL Image
    if isinstance(image_input, Image.Image):
        pil_source = image_input.convert("RGB")
        img_rgb = np.array(pil_source)
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input.copy()
        pil_source = Image.fromarray(img_rgb)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    img_h, img_w = img_rgb.shape[:2]

    # 2. Generate candidate region proposals
    proposals = generate_region_proposals(img_rgb)
    if len(proposals) > max_proposals:
        proposals = proposals[:max_proposals]

    candidates: List[Dict[str, Any]] = []
    ignored_count = 0

    # 3. Classify each candidate proposal
    for (bx, by, bw, bh) in proposals:
        # Add 5% context margin
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(img_w, bx + bw + pad_x)
        y2 = min(img_h, by + bh + pad_y)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        crop_pil = pil_source.crop((x1, y1, x2, y2))
        res = predictor.predict(crop_pil, is_bgr=False, smooth=False)
        top_conf = res["confidence"]
        top_class = res["class_raw"]

        # Rejection: must not be 'not_food' and must meet threshold
        if top_class == "not_food" or top_conf < conf_thresh:
            ignored_count += 1
            continue

        candidates.append({
            "box": (bx, by, bw, bh),
            "class_raw": top_class,
            "class_name": res["class_name"],
            "confidence": top_conf,
            "price": res["price"],
            "kcal": res["kcal"],
            "area": bw * bh,
            "is_full_frame": (bw >= img_w * 0.90 and bh >= img_h * 0.90)
        })

    # 4. Multi-Item Disambiguation: Detect and suppress composite parent boxes
    # A parent box (like a whole plate or full frame) that encompasses 2 or more mutually disjoint
    # child food items should be suppressed so that individual items are counted accurately.
    composite_indices = set()
    for i, c_parent in enumerate(candidates):
        children = []
        for j, c_child in enumerate(candidates):
            if i == j:
                continue
            if c_child["area"] < c_parent["area"] * 0.85:
                containment = calculate_containment(c_child["box"], c_parent["box"])
                iou = calculate_iou(c_child["box"], c_parent["box"])
                if containment > 0.40 or iou > 0.25:
                    children.append(c_child)

        # Check if parent covers at least 2 distinct items that do NOT overlap with each other
        has_multiple_disjoint = False
        for c1_idx in range(len(children)):
            for c2_idx in range(c1_idx + 1, len(children)):
                if calculate_iou(children[c1_idx]["box"], children[c2_idx]["box"]) < 0.25:
                    has_multiple_disjoint = True
                    break
            if has_multiple_disjoint:
                break

        if has_multiple_disjoint:
            composite_indices.add(i)

    filtered_candidates = [c for idx, c in enumerate(candidates) if idx not in composite_indices]

    # Sort remaining candidates by confidence descending
    filtered_candidates.sort(key=lambda c: c["confidence"], reverse=True)

    kept_candidates: List[Dict[str, Any]] = []
    for cand in filtered_candidates:
        boxA = cand["box"]
        suppress = False
        for kept in kept_candidates:
            boxB = kept["box"]
            iou = calculate_iou(boxA, boxB)
            containmentA = calculate_containment(boxA, boxB)
            containmentB = calculate_containment(boxB, boxA)

            # Suppress if high IoU or significant containment
            if iou > nms_iou_threshold or containmentA > 0.45 or containmentB > 0.45:
                suppress = True
                break
        if not suppress:
            kept_candidates.append(cand)

    # Single-dish fallback: if no candidate survived disambiguation, use top candidate
    if not kept_candidates and candidates:
        kept_candidates.append(max(candidates, key=lambda c: c["confidence"]))

    # Sort remaining detections from left to right (natural reading order)
    kept_candidates.sort(key=lambda c: c["box"][0])

    # 5. Draw bounding boxes on image
    annotated = img_rgb.copy()
    draw_detections(annotated, kept_candidates)

    return {
        "detections": kept_candidates,
        "total_detected": len(kept_candidates),
        "ignored_count": ignored_count,
        "image_annotated": annotated
    }


def draw_detections(
    image_rgb: np.ndarray,
    detections: List[Dict[str, Any]]
) -> np.ndarray:
    """Draws styled bounding boxes, labels, and prices onto an RGB image."""
    COLORS = [
        (37, 99, 235),   # Blue
        (16, 185, 129),  # Green
        (245, 158, 11),  # Amber
        (239, 68, 68),   # Red
        (139, 92, 246),  # Purple
        (236, 72, 153),  # Pink
    ]

    h, w = image_rgb.shape[:2]

    for i, det in enumerate(detections):
        bx, by, bw, bh = det["box"]
        color = COLORS[i % len(COLORS)]

        # Draw main bounding box
        cv2.rectangle(image_rgb, (bx, by), (bx + bw, by + bh), color, 2)

        # Draw corner accents
        c_len = min(20, bw // 4, bh // 4)
        cv2.line(image_rgb, (bx, by), (bx + c_len, by), color, 4)
        cv2.line(image_rgb, (bx, by), (bx, by + c_len), color, 4)
        cv2.line(image_rgb, (bx + bw, by), (bx + bw - c_len, by), color, 4)
        cv2.line(image_rgb, (bx + bw, by), (bx + bw, by + c_len), color, 4)
        cv2.line(image_rgb, (bx, by + bh), (bx + c_len, by + bh), color, 4)
        cv2.line(image_rgb, (bx, by + bh), (bx, by + bh - c_len), color, 4)
        cv2.line(image_rgb, (bx + bw, by + bh), (bx + bw - c_len, by + bh), color, 4)
        cv2.line(image_rgb, (bx + bw, by + bh), (bx + bw, by + bh - c_len), color, 4)

        # Label text: "#1 Samosa (88%) - Rs 20"
        label = f"#{i+1} {det['class_name']} ({det['confidence']*100:.0f}%)"
        sub_label = f"Rs. {det['price']} | {det['kcal']} kcal"

        # Badge dimensions
        badge_w = max(160, bw)
        badge_h = 38
        badge_y = max(0, by - badge_h - 4)

        # Draw dark translucent badge background
        sub_img = image_rgb[badge_y:badge_y + badge_h, bx:min(w, bx + badge_w)]
        if sub_img.size > 0:
            rect = np.full_like(sub_img, 20)
            alpha = 0.80
            cv2.addWeighted(rect, alpha, sub_img, 1 - alpha, 0, sub_img)

        # Draw label text
        cv2.putText(
            image_rgb, label, (bx + 6, badge_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            image_rgb, sub_label, (bx + 6, badge_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 120), 1, cv2.LINE_AA
        )

    return image_rgb
