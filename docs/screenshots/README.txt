================================================================================
SMART CANTEEN - USER INTERFACE SCREENSHOTS
================================================================================

This directory stores user interface screenshots of the Smart Canteen Billing
Assistant application for documentation and capstone presentation.

Existing Screenshots:
---------------------
1. `chai_samosa_success.png`
   - Multi-item detection result (Samosa x1 + Chai x1) with bounding boxes and
     cashier verification table.
2. `single_dish_detection.png`
   - Streamlit application interface demonstrating non-food rejection on an
     empty plate with cutlery, reporting 0 items detected and safety filter warning.
3. `single_dish_detection_heatmap.png`
   - Grad-CAM Explainable AI expander in Streamlit showing activation heatmap
     overlay concentrated on a fork for non-food rejection.
4. `README.txt`
   - This manifest document.

Missing Screenshots (OPTIONAL - to be captured from local runs if desired):
---------------------------------------------------------------------------
The following application interface screenshots are OPTIONAL and not currently
required for code execution:
- `multi_item_canteen_bill.png` : (OPTIONAL) Full canteen bill summary receipt view with price and calorie totals.
- `live_camera_demo.png` : (OPTIONAL) Desktop OpenCV live camera HUD window (app/live_demo.py).
- `detector_debug_view.png` : (OPTIONAL) Streamlit app with candidate proposal debug checkbox active showing all proposed bounding boxes.
================================================================================
