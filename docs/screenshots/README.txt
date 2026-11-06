================================================================================
SMART CANTEEN - SAMPLE OUTPUT SCREENSHOTS GUIDE
================================================================================

This folder stores sample UI screenshots of the running Smart Canteen Billing
Assistant application for capstone submission, documentation, and the README.

Recommended screenshots to capture and drop into this folder:

1. `01_single_dish_detection.png`
   - Description: Streamlit app showing a single dish (e.g. Masala Dosa, Burger, or Pizza)
     classified with high confidence (>90%), price, and nutrition badge.

2. `02_multi_item_canteen_bill.png`
   - Description: Streamlit app showing multi-item detection (e.g. Samosa + Chai or Two Samosas)
     with bounding boxes, cashier verification table, and the generated canteen bill (totals in INR and kcal).

3. `03_detector_debug_view.png`
   - Description: Streamlit app with "Debug View" checkbox enabled, displaying all candidate proposal
     boxes, confidence scores, proposal engine label (YOLO-World / Heuristics), and reject reasons.

4. `04_gradcam_explainability.png`
   - Description: Streamlit Grad-CAM expander showing the class activation heatmap overlay on a dish
     proving focus on salient food textures rather than plates or tables.

5. `05_live_camera_demo.png`
   - Description: Desktop OpenCV live camera HUD (app/live_demo.py) showing real-time detection,
     10-frame smoothed probability bar, bounding brackets, and live cashier keys (SPACE/C/S/G/M/Q).

Files referenced in README.md:
- docs/screenshots/01_single_dish_detection.png
- docs/screenshots/02_multi_item_canteen_bill.png
- docs/screenshots/03_detector_debug_view.png
- docs/screenshots/04_gradcam_explainability.png
- docs/screenshots/05_live_camera_demo.png
================================================================================

