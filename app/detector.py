"""
Smart Canteen Billing Assistant: Multi-Item Food Detector (app/detector.py)

Architecture (Detect-then-Classify):
1. Region Proposal Generation:
   (a) Ultralytics YOLO-World (open-vocabulary) with generic prompts:
       ["food", "samosa", "cup of tea", "bowl", "bun", "plate of food", "cup", "plate"]
   (b) FastSAM (segmentation-based proposal method) as secondary fallback.
   (c) Heuristic proposals (Otsu, contours, multi-scale spatial partitions) as last-resort fallback.
2. Tight Crop & Context Masking:
   - Each tight candidate box is padded by ~5%.
   - Overlapping regions with other peer dishes are masked out (filled with neutral gray 128)
     so a crop never contains features of multiple dishes. Container/plate boxes are excluded
     from masking dishes since food sits on top of containers.
   - Crops are classified using the existing MobileNetV3 FoodPredictor (unchanged).
3. Disambiguation & Non-Maximum Suppression:
   - One physical item = one box.
   - Suppresses boxes mostly inside a larger plate/container only if the plate box is a valid food class.
     If the container is not food, or contains multiple disjoint items, the container is suppressed.
   - Class-agnostic NMS eliminates redundant candidate boxes for the same physical item.
4. Visualization & Debug View:
   - Clean production overlay for verified food items.
   - Comprehensive Debug View visualizing ALL candidate proposals with class, confidence,
     and reject reasons (not_food / low confidence / NMS / container suppression).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Project roots and weights
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT

# Cached model instances
_YOLO_WORLD_MODEL = None
_FASTSAM_MODEL = None

DEFAULT_PROMPTS = [
    "food",
    "samosa",
    "cup of tea",
    "bowl",
    "bun",
    "plate of food"
]

CONTAINER_LABELS = {"plate", "plate of food", "tray", "table", "bowl"}


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


# ==============================================================================
# 1. REGION PROPOSAL GENERATORS
# ==============================================================================
def get_yolo_world_model():
    """Initializes and caches the YOLO-World open-vocabulary detector."""
    global _YOLO_WORLD_MODEL
    if _YOLO_WORLD_MODEL is not None:
        return _YOLO_WORLD_MODEL

    try:
        from ultralytics import YOLOWorld
        import torch

        weights_candidates = [
            WEIGHTS_DIR / "yolov8s-worldv2.pt",
            WEIGHTS_DIR / "yolov8m-worldv2.pt",
            Path("yolov8s-worldv2.pt"),
            Path("yolov8m-worldv2.pt"),
        ]
        chosen_weight = None
        for cand in weights_candidates:
            if cand.exists():
                chosen_weight = cand
                break

        if chosen_weight is None:
            chosen_weight = "yolov8s-worldv2.pt"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLOWorld(str(chosen_weight))
        model.to(device)
        model.set_classes(DEFAULT_PROMPTS)
        _YOLO_WORLD_MODEL = model
        return _YOLO_WORLD_MODEL
    except Exception as e:
        print(f"[Detector] Failed to load YOLO-World: {e}")
        return None


def get_fastsam_model():
    """Initializes and caches the FastSAM segmentation proposal model."""
    global _FASTSAM_MODEL
    if _FASTSAM_MODEL is not None:
        return _FASTSAM_MODEL

    try:
        from ultralytics import FastSAM
        import torch

        weights_candidates = [
            WEIGHTS_DIR / "FastSAM-s.pt",
            Path("FastSAM-s.pt"),
        ]
        chosen_weight = None
        for cand in weights_candidates:
            if cand.exists():
                chosen_weight = cand
                break

        if chosen_weight is None:
            chosen_weight = "FastSAM-s.pt"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FastSAM(str(chosen_weight))
        model.to(device)
        _FASTSAM_MODEL = model
        return _FASTSAM_MODEL
    except Exception as e:
        print(f"[Detector] Failed to load FastSAM: {e}")
        return None


def generate_yolo_world_proposals(
    pil_img: Image.Image,
    conf_threshold: float = 0.08
) -> Optional[List[Dict[str, Any]]]:
    """Generates candidate proposals using YOLO-World open-vocabulary detector."""
    model = get_yolo_world_model()
    if model is None:
        return None

    try:
        results = model.predict(pil_img, conf=conf_threshold, verbose=False)
        if not results or len(results) == 0:
            return []

        res = results[0]
        raw_boxes = []
        img_w, img_h = pil_img.size

        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            cls_name = res.names.get(cls_id, "object") if hasattr(res, "names") else "object"
            conf = float(b.conf[0])

            bx = max(0, int(round(x1)))
            by = max(0, int(round(y1)))
            bw = min(img_w - bx, int(round(x2 - x1)))
            bh = min(img_h - by, int(round(y2 - y1)))

            if bw < 20 or bh < 20:
                continue

            raw_boxes.append({
                "box": (bx, by, bw, bh),
                "source": "yolo_world",
                "label": cls_name,
                "score": conf,
                "is_container": cls_name.lower() in CONTAINER_LABELS
            })

        # Pre-filter duplicate detector proposals of the exact same class with high IoU
        raw_boxes.sort(key=lambda x: x["score"], reverse=True)
        proposals: List[Dict[str, Any]] = []
        for cand in raw_boxes:
            dup = False
            for p in proposals:
                if cand["label"] == p["label"] and calculate_iou(cand["box"], p["box"]) > 0.65:
                    dup = True
                    break
            if not dup:
                proposals.append(cand)

        return proposals
    except Exception as e:
        print(f"[Detector] YOLO-World prediction failed: {e}")
        return None


def generate_fastsam_proposals(
    pil_img: Image.Image,
    conf_threshold: float = 0.25
) -> Optional[List[Dict[str, Any]]]:
    """Generates candidate proposals using FastSAM instance segmentation."""
    model = get_fastsam_model()
    if model is None:
        return None

    try:
        results = model(pil_img, conf=conf_threshold, iou=0.7, verbose=False)
        if not results or len(results) == 0:
            return []

        res = results[0]
        proposals = []
        img_w, img_h = pil_img.size
        total_area = float(img_w * img_h)

        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])

            bx = max(0, int(round(x1)))
            by = max(0, int(round(y1)))
            bw = min(img_w - bx, int(round(x2 - x1)))
            bh = min(img_h - by, int(round(y2 - y1)))

            area = bw * bh
            if bw < 25 or bh < 25 or area > total_area * 0.95 or area < total_area * 0.015:
                continue

            proposals.append({
                "box": (bx, by, bw, bh),
                "source": "fastsam",
                "label": "segment",
                "score": conf,
                "is_container": False
            })

        return proposals
    except Exception as e:
        print(f"[Detector] FastSAM prediction failed: {e}")
        return None


def generate_heuristic_proposals(
    image_rgb: np.ndarray,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.85
) -> List[Dict[str, Any]]:
    """Last-resort fallback: morphological contours and spatial partitions."""
    h, w = image_rgb.shape[:2]
    total_area = float(h * w)
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio

    raw_boxes: List[Tuple[int, int, int, int]] = []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, thresh_gray = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh_sat = cv2.threshold(saturation, 35, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(thresh_gray, thresh_sat)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh) if bh > 0 else 0
            if 0.25 <= aspect <= 4.0:
                raw_boxes.append((bx, by, bw, bh))

    half_w = w // 2
    half_h = h // 2
    raw_boxes.append((0, 0, half_w, h))
    raw_boxes.append((half_w, 0, w - half_w, h))
    raw_boxes.append((0, 0, half_w, half_h))
    raw_boxes.append((half_w, 0, w - half_w, half_h))
    raw_boxes.append((0, half_h, half_w, h - half_h))
    raw_boxes.append((half_w, half_h, w - half_w, h - half_h))

    c_sz = int(min(h, w) * 0.70)
    cx = (w - c_sz) // 2
    cy = (h - c_sz) // 2
    raw_boxes.append((cx, cy, c_sz, c_sz))
    raw_boxes.append((0, 0, w, h))

    unique: List[Tuple[int, int, int, int]] = []
    for p in raw_boxes:
        if not any(calculate_iou(p, u) > 0.85 for u in unique):
            unique.append(p)

    return [{
        "box": b,
        "source": "heuristic_fallback",
        "label": "heuristic",
        "score": 0.50,
        "is_container": False
    } for b in unique]


def get_candidate_proposals(
    pil_img: Image.Image,
    preferred_method: str = "auto"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Tiered proposal selection:
    1. YOLO-World (Open-Vocabulary generic detection)
    2. FastSAM (Segment Anything fallback)
    3. Heuristics (Otsu & spatial partition fallback)
    """
    img_rgb = np.array(pil_img)

    if preferred_method in ["auto", "yolo_world"]:
        props = generate_yolo_world_proposals(pil_img)
        if props is not None and len(props) > 0:
            return props, "yolo_world"

    if preferred_method in ["auto", "fastsam"]:
        props = generate_fastsam_proposals(pil_img)
        if props is not None and len(props) > 0:
            return props, "fastsam"

    props = generate_heuristic_proposals(img_rgb)
    return props, "heuristic_fallback"


# ==============================================================================
# 2. DETECT-THEN-CLASSIFY PIPELINE
# ==============================================================================
def detect_food_items(
    image_input: Any,
    predictor,
    threshold: Optional[float] = None,
    nms_iou_threshold: float = 0.35,
    proposal_method: str = "auto",
    max_proposals: int = 30
) -> Dict[str, Any]:
    """
    Detects individual food items using tight open-vocabulary proposals, 5% padding,
    overlap masking, and MobileNetV3 classification.
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
    proposals, active_method = get_candidate_proposals(pil_source, preferred_method=proposal_method)
    if len(proposals) > max_proposals:
        proposals.sort(key=lambda p: p["score"], reverse=True)
        proposals = proposals[:max_proposals]

    # Separate container/tableware proposals from discrete item proposals
    # Food dishes sit ON containers, so containers should NOT mask dishes
    candidate_items = [p for p in proposals if not p.get("is_container", False)]
    candidate_containers = [p for p in proposals if p.get("is_container", False)]

    # We evaluate all proposals (items + containers) through the classifier
    eval_list = candidate_items + candidate_containers
    all_evaluated_candidates: List[Dict[str, Any]] = []

    # 3. Crop, pad (~5%), mask overlapping peer dishes, and classify
    for idx, prop in enumerate(eval_list):
        bx, by, bw, bh = prop["box"]
        is_prop_container = prop.get("is_container", False)

        # 5% context padding
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        cx1 = max(0, bx - pad_x)
        cy1 = max(0, by - pad_y)
        cx2 = min(img_w, bx + bw + pad_x)
        cy2 = min(img_h, by + bh + pad_y)

        if (cx2 - cx1) < 20 or (cy2 - cy1) < 20:
            continue

        crop = pil_source.crop((cx1, cy1, cx2, cy2)).copy()

        # Mask overlapping peer items with gray (128, 128, 128)
        # Prevents one crop from containing two distinct dishes.
        # Only other discrete ITEMS mask this crop (containers/plates are never used to mask dishes).
        draw = ImageDraw.Draw(crop)
        masked_peer = False

        if not is_prop_container:
            for other_item in candidate_items:
                obx, oby, obw, obh = other_item["box"]
                if (obx, oby, obw, obh) == (bx, by, bw, bh):
                    continue

                # Only mask genuine peer items touching at border (not duplicates, containers, or enclosed items)
                iou = calculate_iou((bx, by, bw, bh), (obx, oby, obw, obh))
                contA = calculate_containment((bx, by, bw, bh), (obx, oby, obw, obh))
                contB = calculate_containment((obx, oby, obw, obh), (bx, by, bw, bh))

                if iou >= 0.35 or contA >= 0.35 or contB >= 0.35:
                    continue

                # Intersection between this crop and other peer item
                ox1 = max(cx1, obx)
                oy1 = max(cy1, oby)
                ox2 = min(cx2, obx + obw)
                oy2 = min(cy2, oby + obh)

                if ox2 > ox1 and oy2 > oy1:
                    lx1 = ox1 - cx1
                    ly1 = oy1 - cy1
                    lx2 = ox2 - cx1
                    ly2 = oy2 - cy1
                    draw.rectangle([lx1, ly1, lx2, ly2], fill=(128, 128, 128))
                    masked_peer = True

        # Classify with existing MobileNetV3 FoodPredictor (unchanged)
        pred = predictor.predict(crop, is_bgr=False, smooth=False)
        top_conf = float(pred["confidence"])
        top_class = pred["class_raw"]

        # Human-readable display name (independent of whether it passes threshold)
        match_row = predictor.menu_df[predictor.menu_df["class"] == top_class]
        if not match_row.empty:
            display_name = str(match_row.iloc[0]["display_name"])
            price = int(match_row.iloc[0]["price"])
            kcal = int(match_row.iloc[0]["kcal"])
        else:
            display_name = top_class.replace("_", " ").title()
            price = 40
            kcal = 250

        is_food = (top_class != "not_food")

        candidate_record = {
            "index": idx + 1,
            "box": (bx, by, bw, bh),
            "padded_box": (cx1, cy1, cx2 - cx1, cy2 - cy1),
            "area": bw * bh,
            "class_raw": top_class,
            "class_name": display_name,
            "confidence": top_conf,
            "price": price,
            "kcal": kcal,
            "is_food": is_food,
            "masked_peer": masked_peer,
            "yolo_label": prop["label"],
            "yolo_conf": prop["score"],
            "is_container_prop": is_prop_container,
            "status": "pending",
            "reject_reason": None,
            "top_predictions": pred.get("top_predictions", [])
        }

        # Filter: not_food or low confidence
        # Tableware/container proposals (plate, tray, bowl) must be confident food dishes to be accepted
        min_required_conf = max(0.60, conf_thresh) if is_prop_container else conf_thresh
        if not is_food:
            candidate_record["status"] = "rejected_not_food"
            candidate_record["reject_reason"] = "Classified as not_food"
        elif top_conf < min_required_conf:
            candidate_record["status"] = "rejected_low_confidence"
            candidate_record["reject_reason"] = (
                f"Container proposal below safety threshold ({top_conf*100:.1f}% < {min_required_conf*100:.0f}%)"
                if is_prop_container else f"Low confidence ({top_conf*100:.1f}% < {conf_thresh*100:.0f}%)"
            )
        else:
            candidate_record["status"] = "candidate"

        all_evaluated_candidates.append(candidate_record)

    # 4. Multi-item & Container Disambiguation (Requirement 3)
    # "suppress a box that is mostly inside a larger plate/container box only if the plate box itself is not a food class."
    for i, c_parent in enumerate(all_evaluated_candidates):
        # Find all child items mostly inside this parent box
        children = []
        for j, c_child in enumerate(all_evaluated_candidates):
            if i == j:
                continue
            if c_child["area"] < c_parent["area"] * 0.70:
                containment = calculate_containment(c_child["box"], c_parent["box"])
                if containment > 0.55:
                    children.append(c_child)

        if not children:
            continue

        # Check if parent encloses 2 or more distinct disjoint food items
        has_multiple_disjoint = False
        for c1_idx in range(len(children)):
            for c2_idx in range(c1_idx + 1, len(children)):
                if calculate_iou(children[c1_idx]["box"], children[c2_idx]["box"]) < 0.25:
                    has_multiple_disjoint = True
                    break
            if has_multiple_disjoint:
                break

        if has_multiple_disjoint:
            # Parent is a composite platter holding multiple dishes -> suppress parent container
            c_parent["status"] = "suppressed_composite_parent"
            c_parent["reject_reason"] = "Composite container enclosing multiple disjoint items"
        elif len(children) >= 1:
            # Parent covers child item(s):
            # If parent itself is NOT a food class (or not_food / container), suppress parent, keep child
            # If parent itself IS a valid food class (e.g. Pizza or Pav Bhaji), suppress child, keep parent
            if not c_parent["is_food"] or c_parent["status"] != "candidate" or c_parent["is_container_prop"]:
                if c_parent["status"] == "candidate":
                    c_parent["status"] = "suppressed_container"
                    c_parent["reject_reason"] = "Container enclosing food dish"
            else:
                # Parent is a food dish itself; suppress child sub-boxes
                for ch in children:
                    if ch["status"] == "candidate":
                        ch["status"] = "suppressed_inside_food"
                        ch["reject_reason"] = f"Inside larger food item #{c_parent['index']} ({c_parent['class_name']})"

    # 5. Class-Agnostic Non-Maximum Suppression (NMS) (Requirement 3)
    surviving = [c for c in all_evaluated_candidates if c["status"] == "candidate"]
    surviving.sort(key=lambda c: c["confidence"], reverse=True)

    accepted_detections: List[Dict[str, Any]] = []
    for cand in surviving:
        boxA = cand["box"]
        suppress = False
        suppressed_by = None
        for kept in accepted_detections:
            boxB = kept["box"]
            iou = calculate_iou(boxA, boxB)
            contA = calculate_containment(boxA, boxB)
            contB = calculate_containment(boxB, boxA)

            if iou > nms_iou_threshold or contA > 0.65 or contB > 0.65:
                suppress = True
                suppressed_by = kept["index"]
                break

        if suppress:
            cand["status"] = "suppressed_nms"
            cand["reject_reason"] = f"NMS overlap with item #{suppressed_by}"
        else:
            cand["status"] = "accepted"
            accepted_detections.append(cand)

    # Sort accepted detections left-to-right (natural reading order)
    accepted_detections.sort(key=lambda c: c["box"][0])
    for i, det in enumerate(accepted_detections):
        det["item_number"] = i + 1

    # Render overlays
    annotated = img_rgb.copy()
    draw_detections(annotated, accepted_detections)

    debug_annotated = img_rgb.copy()
    draw_debug_candidates(debug_annotated, all_evaluated_candidates, active_method)

    ignored_count = len(all_evaluated_candidates) - len(accepted_detections)

    return {
        "detections": accepted_detections,
        "total_detected": len(accepted_detections),
        "ignored_count": ignored_count,
        "proposal_method": active_method,
        "all_candidates": all_evaluated_candidates,
        "image_annotated": annotated,
        "image_debug": debug_annotated
    }


# ==============================================================================
# 3. DRAWING & RENDERING (PRODUCTION & DEBUG VIEW)
# ==============================================================================
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

        cv2.rectangle(image_rgb, (bx, by), (bx + bw, by + bh), color, 2)

        c_len = min(20, max(6, bw // 4), max(6, bh // 4))
        cv2.line(image_rgb, (bx, by), (bx + c_len, by), color, 4)
        cv2.line(image_rgb, (bx, by), (bx, by + c_len), color, 4)
        cv2.line(image_rgb, (bx + bw, by), (bx + bw - c_len, by), color, 4)
        cv2.line(image_rgb, (bx + bw, by), (bx + bw, by + c_len), color, 4)
        cv2.line(image_rgb, (bx, by + bh), (bx + c_len, by + bh), color, 4)
        cv2.line(image_rgb, (bx, by + bh), (bx, by + bh - c_len), color, 4)
        cv2.line(image_rgb, (bx + bw, by + bh), (bx + bw - c_len, by + bh), color, 4)
        cv2.line(image_rgb, (bx + bw, by + bh), (bx + bw, by + bh - c_len), color, 4)

        item_idx = det.get("item_number", i + 1)
        label = f"#{item_idx} {det['class_name']} ({det['confidence']*100:.0f}%)"
        sub_label = f"Rs. {det['price']} | {det['kcal']} kcal"

        badge_w = max(170, bw)
        badge_h = 38
        badge_y = max(0, by - badge_h - 4)

        sub_img = image_rgb[badge_y:badge_y + badge_h, bx:min(w, bx + badge_w)]
        if sub_img.size > 0:
            rect = np.full_like(sub_img, 20)
            cv2.addWeighted(rect, 0.82, sub_img, 0.18, 0, sub_img)

        cv2.putText(
            image_rgb, label, (bx + 6, badge_y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            image_rgb, sub_label, (bx + 6, badge_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 120), 1, cv2.LINE_AA
        )

    return image_rgb


def draw_debug_candidates(
    image_rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    proposal_method: str
) -> np.ndarray:
    """
    Renders the comprehensive Debug View showing ALL candidate boxes with:
    - Candidate index
    - Classifier class & confidence
    - Status & reject reason (not_food / low confidence / NMS / container)
    - Active proposal method banner
    """
    h, w = image_rgb.shape[:2]

    STATUS_COLORS = {
        "accepted": (16, 185, 129),                # Vibrant Green
        "rejected_not_food": (239, 68, 68),        # Red
        "rejected_low_confidence": (245, 158, 11), # Amber
        "suppressed_nms": (168, 85, 247),          # Purple
        "suppressed_composite_parent": (99, 102, 241), # Indigo
        "suppressed_inside_food": (148, 163, 184), # Gray
        "suppressed_container": (100, 116, 139)    # Slate
    }

    # Top banner for Proposal Method
    is_narrow = (w < 620)
    banner_h = 44 if is_narrow else 32
    banner = image_rgb[0:banner_h, 0:w]
    if banner.size > 0:
        overlay = np.full_like(banner, 15)
        cv2.addWeighted(overlay, 0.85, banner, 0.15, 0, banner)

    method_labels = {
        "yolo_world": "YOLO-World (Open-Vocabulary)",
        "fastsam": "FastSAM (Segment Proposals)",
        "heuristic_fallback": "Heuristic Contours & Spatial Grids (Fallback)"
    }
    method_text = f"Proposal Engine: {method_labels.get(proposal_method, proposal_method)}"
    accepted_count = sum(1 for c in candidates if c["status"] == "accepted")
    counts_text = f"Total: {len(candidates)} | Accepted: {accepted_count} | Suppressed/Rejected: {len(candidates) - accepted_count}"

    if is_narrow:
        cv2.putText(image_rgb, method_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(image_rgb, counts_text, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(image_rgb, method_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(image_rgb, counts_text, (max(10, w - 380), 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    for cand in candidates:
        bx, by, bw, bh = cand["box"]
        status = cand["status"]
        color = STATUS_COLORS.get(status, (200, 200, 200))
        thickness = 2 if status == "accepted" else 1

        cv2.rectangle(image_rgb, (bx, by), (bx + bw, by + bh), color, thickness)

        if status == "accepted":
            tag = f"[ACCEPTED] #{cand['index']} {cand['class_name']} ({cand['confidence']*100:.0f}%)"
            subtag = f"Prop: {cand['yolo_label']}"
        else:
            reason = cand["reject_reason"] or status
            tag = f"#{cand['index']} {cand['class_name']} ({cand['confidence']*100:.0f}%)"
            subtag = f"REJECT: {reason}"

        tag_w = min(w - bx, max(150, bw))
        tag_h = 28
        tag_y = max(banner_h + 2, by - tag_h - 2)

        tag_bg = image_rgb[tag_y:tag_y + tag_h, bx:bx + tag_w]
        if tag_bg.size > 0:
            bg_rect = np.full_like(tag_bg, 25)
            cv2.addWeighted(bg_rect, 0.85, tag_bg, 0.15, 0, tag_bg)

        cv2.putText(
            image_rgb, tag, (bx + 4, tag_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            image_rgb, subtag, (bx + 4, tag_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA
        )

    return image_rgb
