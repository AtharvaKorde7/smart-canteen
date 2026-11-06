===========================================================
Smart Canteen Billing Assistant - Model Weights Directory
===========================================================

This directory contains the production deep learning model artifacts used by
the Smart Canteen Billing Assistant web app and live desktop demo:

1. `mobilenetv3.pth` (~16.29 MB, 17,086,535 bytes)
   - PyTorch state_dict for MobileNetV3-Large fine-tuned on the 14 canteen classes.
   - Deployed production model achieving 93.71% Top-1 test accuracy and 20.65 ms CPU inference.

2. `classes.json` (185 bytes)
   - Ordered JSON list of the 14 target classes (13 food dishes + 1 not_food background class):
     ["burger", "chai", "chole_bhature", "fried_rice", "idli", "jalebi", "masala_dosa",
      "momos", "not_food", "pav_bhaji", "pizza", "poha", "samosa", "vada_pav"]

3. `food_model.onnx` (Optional, ~16 MB or ~4.23 MB INT8 quantized)
   - Exported ONNX model (opset 17) for high-speed edge POS inference via ONNX Runtime (4.40 ms CPU latency).

How to reproduce or retrain:
- Open `notebooks/capstone_training_kaggle.ipynb` on Kaggle with a free NVIDIA Tesla T4 GPU.
- After training completes, the notebook exports `mobilenetv3.pth` and `classes.json` directly into `/kaggle/working/`.
===========================================================
