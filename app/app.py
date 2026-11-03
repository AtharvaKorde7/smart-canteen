"""
Smart Canteen Billing Assistant: Food Image Recognition using Transfer Learning
Streamlit Demo Web Application for Laptop (app/app.py)

Key Features:
1. Mutual-exclusion input selector (Upload Photo, Webcam Snapshot, Live Stream)
   - Eliminates Windows camera device collisions (NotReadableError).
2. Robust Image Sanitization:
   - Handles JPG, PNG (RGBA), WEBP, CMYK, Palette, and EXIF orientation transposition.
3. Multi-Item Food Detection (Detect-then-Classify):
   - Proposes candidate food regions, classifies each with MobileNetV3, applies NMS.
   - Displays bounding boxes, labels, and confidence overlays on image.
4. Quantity-based Billing & Editable Cashier Review:
   - Aggregates multi-dish counts into bill lines (app/billing.py).
   - Interactive st.data_editor allows cashier to adjust quantities before adding to bill.
   - Running bill panel with receipt generation (CSV download, printable text/HTML).
5. Explainable AI:
   - On-demand Grad-CAM computation with guaranteed PyTorch gradient enabling.

Authors: Capstone Project Team (Ramdeobaba University)
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add app directory and project root to sys.path so app modules import cleanly
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

# Safe import of OpenCV to handle headless/cloud environments gracefully
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_IMPORT_ERROR = None
except Exception as _cv2_err:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_IMPORT_ERROR = str(_cv2_err)

# Import shared prediction, detection, and billing utility functions
try:
    from predictor import (
        FoodPredictor,
        load_menu,
        load_class_names,
        compute_gradcam,
        MODEL_DIR,
        MENU_FILE,
    )
    from detector import (
        detect_food_items,
        CV2_AVAILABLE as DET_CV2_AVAILABLE,
        CV2_IMPORT_ERROR as DET_CV2_ERROR,
    )
    from billing import (
        aggregate_detections,
        merge_into_bill,
        generate_receipt_text,
        generate_receipt_html,
        generate_receipt_csv,
    )
except ImportError:
    from app.predictor import (
        FoodPredictor,
        load_menu,
        load_class_names,
        compute_gradcam,
        MODEL_DIR,
        MENU_FILE,
    )
    from app.detector import (
        detect_food_items,
        CV2_AVAILABLE as DET_CV2_AVAILABLE,
        CV2_IMPORT_ERROR as DET_CV2_ERROR,
    )
    from app.billing import (
        aggregate_detections,
        merge_into_bill,
        generate_receipt_text,
        generate_receipt_html,
        generate_receipt_csv,
    )

# Optional WebRTC import for continuous live webcam streaming in browser
try:
    import av
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode,
    )
    WEBRTC_AVAILABLE = CV2_AVAILABLE
except ImportError:
    WEBRTC_AVAILABLE = False


# ==============================================================================
# CONFIGURATION & GLOBAL SETTINGS
# ==============================================================================
ACTIVE_MODEL_ARCH = "mobilenetv3"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"

st.set_page_config(
    page_title="Smart Canteen Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .receipt-box {
        background-color: #FAFAFA;
        border: 1px dashed #9CA3AF;
        padding: 1.2rem;
        border-radius: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data
def get_cached_menu():
    """Loads the 14-class canteen menu with prices (INR) and calories (kcal)."""
    return load_menu(MENU_FILE)


@st.cache_data
def get_cached_classes():
    """Loads the 14 class names."""
    return load_class_names()


@st.cache_resource
def get_predictor_instance(arch_name: str, threshold: float):
    """Initializes and caches the FoodPredictor model instance."""
    try:
        predictor = FoodPredictor(
            backend="torch",
            arch_name=arch_name,
            threshold=threshold,
            window_size=10
        )
        return predictor, None
    except Exception as e:
        return None, str(e)


def load_and_sanitize_image(image_file_or_data) -> tuple[Optional[Image.Image], Optional[str]]:
    """
    Safely opens uploaded or snapshot images:
    - Transposes EXIF orientation (fixes rotated mobile photos)
    - Converts RGBA, CMYK, Palette, Greyscale, and WebP cleanly to RGB
    - Gracefully catches corrupted files with user-friendly error messages
    """
    try:
        img = Image.open(image_file_or_data)
        img = ImageOps.exif_transpose(img)
        rgb_img = img.convert("RGB")
        return rgb_img, None
    except Exception as err:
        return None, f"Unable to read image file: {err}"


# ==============================================================================
# WEBRTC VIDEO PROCESSOR CLASS (LIVE STREAMING)
# ==============================================================================
if WEBRTC_AVAILABLE:
    class LiveFoodVideoProcessor(VideoProcessorBase):
        """Processes continuous live frames from browser webcam using shared predictor."""

        def __init__(self):
            self.predictor = None
            self.last_result = None
            self.frame_count = 0

        def set_predictor(self, predictor: FoodPredictor):
            self.predictor = predictor

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            h, w = img.shape[:2]

            # Center guide box coordinates (65% of screen height)
            box_size = int(min(h, w) * 0.65)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Run inference every 3rd frame to optimize latency
            if self.predictor is not None and (self.frame_count % 3 == 0 or self.last_result is None):
                crop = img[y1:y2, x1:x2]
                self.last_result = self.predictor.predict_cv2(crop, smooth=True)

            # Determine box styling
            if self.last_result is not None:
                res = self.last_result
                if res["is_food"]:
                    color = (0, 220, 0)  # Green
                    text = f"{res['class_name']} ({res['confidence']*100:.1f}%)"
                    sub_text = f"Rs. {res['price']}  |  {res['kcal']} kcal"
                else:
                    color = (0, 0, 230)  # Red
                    text = "Unknown food"
                    sub_text = "Place food dish inside guide box"
            else:
                color = (200, 200, 200)
                text = "Detecting food..."
                sub_text = "Place food dish inside guide box"

            # Draw targeting box with corner brackets
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            c_len = 22
            cv2.line(img, (x1, y1), (x1 + c_len, y1), color, 3)
            cv2.line(img, (x1, y1), (x1, y1 + c_len), color, 3)
            cv2.line(img, (x2, y1), (x2 - c_len, y1), color, 3)
            cv2.line(img, (x2, y1), (x2, y1 + c_len), color, 3)
            cv2.line(img, (x1, y2), (x1 + c_len, y2), color, 3)
            cv2.line(img, (x1, y2), (x1, y2 - c_len), color, 3)
            cv2.line(img, (x2, y2), (x2 - c_len, y2), color, 3)
            cv2.line(img, (x2, y2), (x2, y2 - c_len), color, 3)

            # Draw status banners
            cv2.putText(img, text, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            cv2.putText(img, sub_text, (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

            # Show Top-3 predictions
            if self.last_result is not None and "top_predictions" in self.last_result:
                for i, tp in enumerate(self.last_result["top_predictions"][:3]):
                    t_str = f"{i+1}. {tp['class_name']}: {tp['confidence']*100:.1f}%"
                    t_col = (0, 220, 0) if i == 0 and self.last_result["is_food"] else (220, 220, 220)
                    cv2.putText(img, t_str, (12, 24 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, t_col, 1, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# SESSION STATE INITIALIZATION (RUNNING BILL & DETECTIONS)
# ==============================================================================
if "bill_items" not in st.session_state:
    st.session_state.bill_items = []

if "generated_bill_data" not in st.session_state:
    st.session_state.generated_bill_data = None


def clear_bill():
    """Resets the running bill."""
    st.session_state.bill_items = []
    st.session_state.generated_bill_data = None


# ==============================================================================
# SIDEBAR CONTROLS & CANTEEN MENU
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Settings & Controls")

    selected_arch = st.selectbox(
        "Model Architecture",
        options=["mobilenetv3", "efficientnet_b0"],
        index=0 if ACTIVE_MODEL_ARCH == "mobilenetv3" else 1,
        help="Select model architecture matching your downloaded weights file."
    )

    conf_threshold = st.slider(
        "Confidence Safety Threshold",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help="Predictions below this confidence or classified as 'not_food' are rejected to prevent billing errors."
    )

    st.divider()
    st.subheader("🛠️ Developer Options")
    debug_mode = st.checkbox(
        "🐞 Debug Mode",
        value=False,
        help="Inspect the exact 224x224 tensor image that the neural network sees, plus proposal counts."
    )

    st.divider()
    st.subheader("📋 Canteen Menu & Rates (14 Classes / 13 Dishes)")
    menu_df = get_cached_menu()
    public_menu = menu_df[menu_df["class"] != "not_food"]
    st.dataframe(
        public_menu[["display_name", "price", "kcal"]].rename(
            columns={"display_name": "Dish", "price": "Price (₹)", "kcal": "Calories (kcal)"}
        ),
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.caption("Capstone Project — DL Lab 2026-27\nRamdeobaba University, Nagpur")


# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.markdown('<div class="main-title">🍽️ Smart Canteen Billing Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Food Image Recognition using Deep Learning (14 Classes: 13 Dishes + <code>not_food</code>) '
    'with automated multi-item detection, quantity counting, calorie estimation, and cashier billing.</div>',
    unsafe_allow_html=True
)

predictor, model_error = get_predictor_instance(selected_arch, conf_threshold)
class_names = get_cached_classes()

if not CV2_AVAILABLE or not DET_CV2_AVAILABLE:
    st.error(
        f"⚠️ **OpenCV (`cv2`) failed to import:** `{CV2_IMPORT_ERROR or DET_CV2_ERROR}`\n\n"
        "If running on Streamlit Community Cloud (Debian Linux), ensure `packages.txt` exists in the repository root containing:\n"
        "```\nlibgl1\nlibglib2.0-0\n```"
    )

if model_error:
    st.error(
        f"⚠️ **Model weights missing or failed to load:**\n\n`{model_error}`\n\n"
        f"Please ensure `mobilenetv3.pth` is placed inside `app/model/`."
    )

# ==============================================================================
# TWO COLUMN LAYOUT: DETECTION & BILLING
# ==============================================================================
col_left, col_right = st.columns([1.15, 0.85], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: FOOD CAPTURE & MULTI-ITEM DETECTION
# ------------------------------------------------------------------------------
with col_left:
    st.subheader("📸 Food Image Capture & Multi-Item Detection")

    # TASK A FIX: Use mutually exclusive radio selector instead of st.tabs
    # This prevents st.camera_input and streamlit-webrtc from competing for the webcam
    input_mode = st.radio(
        "Select Input Method:",
        options=["📁 Upload Photo", "📸 Webcam Snapshot", "🎥 Live Camera Stream"],
        horizontal=True,
        key="app_input_mode_selector"
    )

    image_to_process: Optional[Image.Image] = None

    # Option 1: Upload Photo (Zero camera access)
    if input_mode == "📁 Upload Photo":
        uploaded_file = st.file_uploader(
            "Select an image file (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            key="food_photo_uploader"
        )
        if uploaded_file is not None:
            clean_img, err = load_and_sanitize_image(uploaded_file)
            if err:
                st.error(f"❌ {err}")
            else:
                image_to_process = clean_img

    # Option 2: Webcam Snapshot (Exclusive st.camera_input access)
    elif input_mode == "📸 Webcam Snapshot":
        cam_photo = st.camera_input("Take a photo of the food plate:", key="food_camera_input")
        if cam_photo is not None:
            clean_img, err = load_and_sanitize_image(cam_photo)
            if err:
                st.error(f"❌ {err}")
            else:
                image_to_process = clean_img

    # Option 3: Continuous Live Camera Stream (Exclusive WebRTC access)
    elif input_mode == "🎥 Live Camera Stream":
        st.write("**Continuous Video Stream:** Show your food plate to the camera for live real-time classification.")
        if WEBRTC_AVAILABLE and predictor is not None:
            rtc_config = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            webrtc_ctx = webrtc_streamer(
                key="smart-canteen-live-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=LiveFoodVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.set_predictor(predictor)
                live_res = webrtc_ctx.video_processor.last_result
                if live_res is not None:
                    st.divider()
                    if live_res["is_food"]:
                        st.success(f"### Live Detected: **{live_res['class_name']}** ({live_res['confidence']*100:.1f}%)")
                        lc1, lc2 = st.columns(2)
                        with lc1:
                            st.metric("Unit Rate", f"₹ {live_res['price']}")
                        with lc2:
                            st.metric("Energy", f"{live_res['kcal']} kcal")

                        if st.button("➕ Add Live Item to Bill", type="primary", use_container_width=True, key="btn_add_live"):
                            st.session_state.bill_items = merge_into_bill(
                                st.session_state.bill_items,
                                [{
                                    "class": live_res["class_raw"],
                                    "name": live_res["class_name"],
                                    "qty": 1,
                                    "price": live_res["price"],
                                    "kcal": live_res["kcal"]
                                }]
                            )
                            st.toast(f"Added {live_res['class_name']} to bill! 🧾")
                            st.rerun()
                    else:
                        st.warning("⚠️ **No food / Unknown food** — Low confidence or non-food item detected.")
        elif not WEBRTC_AVAILABLE:
            st.warning("⚠️ `streamlit-webrtc` is not installed. Use **📸 Webcam Snapshot** or run `python app/live_demo.py`.")
        else:
            st.info("Please load model weights first to enable live camera recognition.")

    # --------------------------------------------------------------------------
    # PROCESS STATIC IMAGE (FROM UPLOAD OR SNAPSHOT)
    # --------------------------------------------------------------------------
    if image_to_process is not None and predictor is not None:
        st.divider()
        st.subheader("🔍 Multi-Item Detection Results")

        # Run Detect-then-Classify multi-item pipeline
        with st.spinner("Detecting and classifying food items..."):
            det_result = detect_food_items(
                image_to_process,
                predictor,
                threshold=conf_threshold
            )

        detections = det_result["detections"]
        annotated_img = det_result["image_annotated"]
        ignored_count = det_result["ignored_count"]

        # Notice when running in lightweight mode without YOLO
        if det_result.get("proposal_method") == "heuristic_fallback":
            st.info("ℹ️ Lightweight mode: YOLO unavailable")

        # Toggle between Clean Production View and Detailed Debug View
        col_img_hdr, col_dbg_chk = st.columns([0.60, 0.40])
        with col_img_hdr:
            st.caption(f"**Proposal Engine:** `{det_result['proposal_method']}` | **Found:** {len(detections)} item(s)")
        with col_dbg_chk:
            show_debug_view = st.checkbox("🔍 Debug View (All Proposals)", value=debug_mode, key="chk_debug_view")

        if show_debug_view:
            st.image(
                det_result["image_debug"],
                caption=f"Debug View: {det_result['proposal_method']} proposals with status & reject reasons",
                use_container_width=True
            )
            with st.expander("🛠️ Proposal Candidates Breakdown & Reject Reasons", expanded=True):
                dbg_rows = []
                for c in det_result.get("all_candidates", []):
                    dbg_rows.append({
                        "#": c["index"],
                        "Proposal": f"{c['yolo_label']} ({c['yolo_conf']:.2f})",
                        "Predicted Class": c["class_name"],
                        "Confidence": f"{c['confidence']*100:.1f}%",
                        "Status": c["status"].upper(),
                        "Reject Reason": c["reject_reason"] or "Accepted (Passed safety & NMS)",
                        "Box (x, y, w, h)": str(c["box"])
                    })
                st.dataframe(pd.DataFrame(dbg_rows), hide_index=True, use_container_width=True)
        else:
            st.image(
                annotated_img,
                caption=f"Detection Overlay: {len(detections)} item(s) found",
                use_container_width=True
            )

        if ignored_count > 0:
            st.caption(f"ℹ️ Filtered out {ignored_count} candidate region(s) classified as non-food or below {conf_threshold*100:.0f}% confidence.")

        # Aggregate detections into quantities and lines
        agg_result = aggregate_detections(detections, menu_df=menu_df)
        detected_lines = agg_result["items"]

        if not detected_lines:
            st.warning(
                "⚠️ **No food items detected** — All candidate regions were classified as `not_food` "
                f"or fell below the safety threshold of {conf_threshold*100:.0f}%."
            )
        else:
            st.success(f"🎉 **Found {agg_result['total_qty']} item(s) across {len(detected_lines)} dish category/categories!**")

            # Cashier review & quantity editing table
            st.write("#### ✍️ Cashier Verification & Quantity Adjustment")
            st.info("Review detected quantities before billing. You can edit quantities, add rows, or remove items below:")

            # Prepare data for st.data_editor
            editor_rows = []
            for item in detected_lines:
                editor_rows.append({
                    "Dish": item["name"],
                    "class": item["class"],
                    "Qty": item["qty"],
                    "Unit Price (₹)": item["price"],
                    "Total Price (₹)": item["total_price"],
                    "Unit Energy (kcal)": item["kcal"],
                    "Total Energy (kcal)": item["total_kcal"],
                })

            df_editor = pd.DataFrame(editor_rows)
            edited_df = st.data_editor(
                df_editor,
                column_config={
                    "Dish": st.column_config.TextColumn("Dish Name", disabled=True),
                    "class": None,  # Hide internal class identifier
                    "Qty": st.column_config.NumberColumn("Quantity", min_value=1, max_value=50, step=1, required=True),
                    "Unit Price (₹)": st.column_config.NumberColumn("Price (₹)", disabled=True),
                    "Total Price (₹)": st.column_config.NumberColumn("Total (₹)", disabled=True),
                    "Unit Energy (kcal)": st.column_config.NumberColumn("Energy (kcal)", disabled=True),
                    "Total Energy (kcal)": st.column_config.NumberColumn("Total Energy (kcal)", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
                key="cashier_review_editor"
            )

            # Re-compute totals dynamically based on cashier edits
            items_to_add: List[Dict[str, Any]] = []
            revised_total_price = 0
            revised_total_kcal = 0

            for _, row in edited_df.iterrows():
                q = int(row["Qty"])
                p = int(row["Unit Price (₹)"])
                k = int(row["Unit Energy (kcal)"])
                tot_p = q * p
                tot_k = q * k
                revised_total_price += tot_p
                revised_total_kcal += tot_k
                items_to_add.append({
                    "class": row["class"],
                    "name": row["Dish"],
                    "qty": q,
                    "price": p,
                    "kcal": k,
                    "total_price": tot_p,
                    "total_kcal": tot_k,
                })

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Total Items", sum(i["qty"] for i in items_to_add))
            with col_m2:
                st.metric("Bill Subtotal", f"₹ {revised_total_price}", f"{revised_total_kcal} kcal")

            if st.button("➕ Add Verified Items to Canteen Bill", type="primary", use_container_width=True, key="btn_add_verified"):
                st.session_state.bill_items = merge_into_bill(st.session_state.bill_items, items_to_add)
                st.toast(f"Added {len(items_to_add)} item line(s) to canteen bill! 🧾")
                st.rerun()

        # Debug Mode Inspector
        if debug_mode:
            with st.expander("🐞 Debug Inspector: Candidate Proposals & Classifications", expanded=True):
                st.write(f"**Total proposals generated:** {len(detections) + ignored_count}")
                st.write(f"**Valid food detections retained:** {len(detections)}")
                for i, d in enumerate(detections):
                    st.write(f"- **Box #{i+1}** `{d['class_raw']}` ({d['confidence']*100:.1f}%) at coords `{d['box']}`")

        # Explainable AI: Grad-CAM Expander
        with st.expander("🔬 Explainable AI: View Grad-CAM Heatmap"):
            st.info("Grad-CAM highlights the exact visual features that led the neural network to its decision.")
            if detections:
                target_dish_idx = st.selectbox(
                    "Select item for Grad-CAM inspection:",
                    options=range(len(detections)),
                    format_func=lambda i: f"#{i+1} {detections[i]['class_name']} ({detections[i]['confidence']*100:.0f}%)"
                )
                sel_box = detections[target_dish_idx]["box"]
                crop_pil = image_to_process.crop((sel_box[0], sel_box[1], sel_box[0] + sel_box[2], sel_box[1] + sel_box[3]))
            else:
                crop_pil = image_to_process

            if st.button("🔍 Generate Grad-CAM Heatmap", key="btn_gradcam_trigger"):
                with st.spinner("Computing activation heatmap on model..."):
                    gradcam_img, cam_err = compute_gradcam(predictor.model, predictor.target_layer, crop_pil)
                    if gradcam_img is not None:
                        st.image(gradcam_img, caption="Grad-CAM Activation Heatmap Overlay", use_container_width=True)
                    else:
                        st.warning(f"Grad-CAM visualization unavailable: {cam_err}")


# ------------------------------------------------------------------------------
# RIGHT COLUMN: RUNNING BILL, QUANTITIES & NUTRITIONAL RECEIPT
# ------------------------------------------------------------------------------
with col_right:
    st.subheader("🧾 Running Canteen Bill")

    # Manual Add Option
    with st.expander("➕ Manually Add Dish from Menu"):
        orderable_dishes = menu_df[menu_df["class"] != "not_food"]["class"].tolist()
        manual_dish = st.selectbox(
            "Select item from menu:",
            options=orderable_dishes,
            format_func=lambda x: x.replace("_", " ").title(),
            key="manual_add_dish"
        )
        manual_qty = st.number_input("Quantity:", min_value=1, max_value=20, value=1, step=1, key="manual_add_qty")

        if st.button("Add Manual Dish to Bill"):
            row = menu_df[menu_df["class"] == manual_dish].iloc[0]
            st.session_state.bill_items = merge_into_bill(
                st.session_state.bill_items,
                [{
                    "class": row["class"],
                    "name": row["display_name"],
                    "qty": manual_qty,
                    "price": int(row["price"]),
                    "kcal": int(row["kcal"])
                }]
            )
            st.rerun()

    if not st.session_state.bill_items:
        st.info("Your billing cart is currently empty. Upload a food photo, take a snapshot, or manually add items above.")
    else:
        # Display bill summary table
        bill_data = []
        grand_total_price = 0
        grand_total_kcal = 0
        grand_total_qty = 0

        for idx, item in enumerate(st.session_state.bill_items):
            grand_total_price += item["total_price"]
            grand_total_kcal += item["total_kcal"]
            grand_total_qty += item["qty"]
            bill_data.append({
                "Item": item["name"],
                "Qty": item["qty"],
                "Rate (₹)": item["price"],
                "Amount (₹)": item["total_price"],
                "Calories (kcal)": item["total_kcal"]
            })

        bill_table = pd.DataFrame(bill_data)
        st.dataframe(bill_table, hide_index=True, use_container_width=True)

        # Total Summary Box
        st.markdown(f"""
        <div class="receipt-box">
            <h4 style="margin: 0 0 10px 0; color: #1E3A8A;">Total Bill Summary</h4>
            <div style="font-size: 1.5rem; font-weight: bold; color: #111827;">
                Total Amount: <span style="color: #059669;">₹ {grand_total_price}</span>
            </div>
            <div style="font-size: 1.05rem; color: #4B5563; margin-top: 5px;">
                Total Items: <strong>{grand_total_qty}</strong> &nbsp;|&nbsp;
                Total Energy: <span style="color: #D97706; font-weight: 600;">{grand_total_kcal} kcal</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🗑️ Clear Bill", use_container_width=True):
                clear_bill()
                st.rerun()
        with b_col2:
            if st.button("🧾 Generate Bill", type="primary", use_container_width=True):
                bill_no = f"BILL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                st.session_state.generated_bill_data = {
                    "bill_no": bill_no,
                    "items": list(st.session_state.bill_items),
                    "timestamp": datetime.now().strftime("%d-%b-%Y %I:%M %p")
                }

        # Generated Receipt Modal / Download Section
        if st.session_state.generated_bill_data:
            gb = st.session_state.generated_bill_data
            st.divider()
            st.success(f"✅ Generated Bill #{gb['bill_no']}")

            receipt_text = generate_receipt_text(gb["items"], gb["bill_no"], gb["timestamp"])
            receipt_csv = generate_receipt_csv(gb["items"])
            receipt_html = generate_receipt_html(gb["items"], gb["bill_no"], gb["timestamp"])

            # Render printable receipt
            st.markdown(receipt_html, unsafe_allow_html=True)

            st.write("")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    "📥 Download CSV",
                    data=receipt_csv,
                    file_name=f"{gb['bill_no']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                st.download_button(
                    "📄 Download Text Receipt",
                    data=receipt_text,
                    file_name=f"{gb['bill_no']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown(
    "<center><small>Smart Canteen Billing Assistant • Deep Learning Lab Capstone • "
    "MobileNetV3 (14 Classes / 13 Dishes with Multi-Item Detection)</small></center>",
    unsafe_allow_html=True
)
