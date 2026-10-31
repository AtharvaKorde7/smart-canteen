===========================================================
Smart Canteen Billing Assistant - Model Weights Directory
===========================================================

Please place your trained model artifacts in this directory after running the training notebook in Google Colab:

1. `mobilenetv3.pth` (or `efficientnet_b0.pth`)
   - The PyTorch state_dict saved at the end of training.
   
2. `classes.json`
   - The JSON list mapping class IDs to human-readable dish names.
   
3. `food_model.onnx` (Optional for high-speed ONNX runtime)
   - The exported ONNX model.

How to get these files:
1. Open and run `notebooks/capstone_training.ipynb` on Google Colab with a free T4 GPU.
2. The notebook saves all trained artifacts directly into your Google Drive at:
   `/content/drive/MyDrive/capstone/`
3. Download the files from Google Drive and copy them into this folder:
   `smart-canteen/app/model/`
