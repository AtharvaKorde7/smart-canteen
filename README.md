# 🍽️ Smart Canteen Billing Assistant: Food Image Recognition using Transfer Learning

**Department of Electronics Engineering, Ramdeobaba University (RBU), Nagpur**  
**Course:** Deep Learning Lab Capstone Project (20 Marks) | **Semester:** V  
**Team Members:**
1. **Atharva Korde (Roll No: 08)** — *Dataset collection, curation, merge pipeline & preprocessing*
2. **Dhruv Khedkar (Roll No: 13)** — *Model development, transfer learning fine-tuning & training*
3. **Divyam Sharma (Roll No: 14)** — *Evaluation benchmarks, confusion matrices & local dataset extension*
4. **Divyatwa Kiranpure (Roll No: 15)** — *Streamlit application, live camera OpenCV demo, explainability (Grad-CAM) & report/viva*

---

## 📌 Table of Contents
1. [Project Overview & Key Innovations](#-project-overview--key-innovations)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Section A: One-Time Setup on Laptop (RTX 5050 CUDA 12.8 & Python 3.12)](#-section-a-one-time-setup-on-laptop)
4. [Section B: Kaggle API Token Setup (Colab Secrets & Local)](#-section-b-kaggle-api-token-setup)
5. [Section C: Training on Google Colab (Free T4 GPU)](#-section-c-training-on-google-colab-free-t4-gpu)
6. [Section D: Moving Trained Weights & Running the Demos](#-section-d-moving-trained-weights--running-the-demos)
7. [Section E: How to Photograph Custom Canteen Data (vada_pav, poha & not_food)](#-section-e-how-to-photograph-custom-canteen-data)
8. [Section F: Viva Voce Preparation (10 Questions & Answers)](#-section-f-viva-voce-preparation-10-qas)
9. [Section G: Team Work Distribution](#-section-g-team-work-distribution)

---

## 🌟 Project Overview & Key Innovations

In busy college canteens and office cafeterias, manual billing causes long queues and human accounting mistakes. The **Smart Canteen Billing Assistant** solves this with an end-to-end computer vision billing system:

1. **Automated Vision Billing:** A customer or cashier shows a food plate to a camera or uploads an image.
2. **Lightweight Deep Neural Networks:** A fine-tuned **MobileNetV3-Large** or **EfficientNet-B0** classifies the image into one of **16 categories** (15 delicious canteen dishes + a dedicated `not_food` background rejection class).
3. **Automated POS Tally & Nutrition:** Looks up real-time canteen rates from `menu.csv` and automatically tallies an interactive **running bill** with monetary total (₹) and **nutritional energy estimate** (kcal).
4. **Live Real-Time Camera Detection:** An OpenCV live camera demo (`app/live_demo.py`) with 10-frame probability smoothing, center-cropping, and HUD overlays running directly on laptop GPU/CPU, plus a browser live camera tab via `streamlit-webrtc`.
5. **Two-Tier Safety Rejection (`not_food` + Threshold):**
   - If the model identifies `not_food` (empty plates, hands, trays, tables, cutlery), or if confidence is below the safety threshold (default `60%`), the system displays `"No food / Unknown food"` (red overlay).
   - **Financial Protection:** The item is strictly blocked from being added to the customer's bill.
6. **Explainable AI (Grad-CAM):** Displays an activation heatmap over the dish to verify whether the AI focused on the food texture (e.g., golden crust of samosa, crispy lace of dosa) rather than the plate background.
7. **ONNX Runtime Optimization:** Supports both PyTorch (`.pth`) and ONNX (`.onnx`) runtimes with sub-30ms inference on consumer laptops and edge POS hardware.
8. **Human-in-the-Loop Active Feedback:** Cashiers can log misclassified photos to `feedback/<class>/` for active learning and future retraining.

### The 16 Supported Categories
`burger`, `chai`, `chole_bhature`, `french_fries`, `fried_rice`, `ice_cream`, `idli`, `jalebi`, `masala_dosa`, `momos`, `not_food`, `pav_bhaji`, `pizza`, `poha`, `samosa`, `vada_pav`

*(Note: `not_food` has Price ₹0 and 0 kcal, and can never be added to a bill).*

---

## 📂 Project Architecture & Directory Structure

```text
smart-canteen/
├── README.md                      # Comprehensive guide, setup manual & viva handbook (this file)
├── requirements.txt               # Laptop dependencies (PyTorch installed via cu128 wheel)
├── .gitignore                     # Ignores weights, datasets, .env, caches, outputs
├── .env.example                   # Template for local Kaggle credentials
├── notebooks/
│   └── capstone_training.ipynb    # Complete Google Colab notebook (GPU training, 14 Classes)
├── scripts/
│   └── download_data_local.py     # Optional script to inspect Kaggle dataset locally
├── app/
│   ├── app.py                     # Streamlit web app with Upload, Snapshot & WebRTC Live tabs
│   ├── live_demo.py               # Fast OpenCV desktop live camera app (GPU/CPU, HUD & keys)
│   ├── predictor.py               # Shared prediction, preprocessing, smoothing & menu logic
│   ├── menu.csv                   # 14 Classes with realistic prices (INR) & calories
│   └── model/
│       ├── classes.json           # Ordered JSON list of the 16 category names
│       └── README.txt             # Place downloaded mobilenetv3.pth / food_model.onnx here
├── feedback/                      # Stores misclassified photos submitted via the app
│   └── .gitkeep
└── outputs/                       # Screenshots saved from the live camera demo
```

---

## 💻 Section A: One-Time Setup on Laptop

> [!IMPORTANT]  
> **NVIDIA RTX 5050 GPU (CUDA 12.8) & Python 3.12:**  
> PyTorch and torchvision are **not** listed as standard lines in `requirements.txt`. They must be installed first using the official PyTorch CUDA 12.8 wheel URL before installing the remaining packages.

### Step 1: Open your terminal
On Windows, open PowerShell or Command Prompt, and navigate to the project directory:
```powershell
cd C:\Projects\smart-canteen
```

### Step 2: Create and activate a Python 3.12 virtual environment
```powershell
# Create virtual environment named 'venv' with Python 3.12
py -3.12 -m venv venv

# Activate virtual environment on Windows
venv\Scripts\activate
```
*(You will see `(venv)` appear at the beginning of your command line prompt).*

### Step 3: Install PyTorch with CUDA 12.8 support (for NVIDIA RTX 5050)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
*(If you do not have an NVIDIA GPU or want CPU-only, standard `pip install torch torchvision` can also be used).*

### Step 4: Install the remaining project dependencies
```powershell
pip install -r requirements.txt
```
This installs `streamlit`, `opencv-python`, `streamlit-webrtc`, `av`, `onnxruntime`, `grad-cam`, `pillow`, `pandas`, `numpy`, and `matplotlib`.

---

## 🔑 Section B: Kaggle API Token Setup

To allow automated dataset downloading without hardcoding sensitive keys into git or code:

### How to get your Kaggle Token
1. Go to [https://www.kaggle.com](https://www.kaggle.com) and sign in.
2. Click your profile avatar in the top-right corner $\rightarrow$ **Settings**.
3. Scroll down to the **API** section and click **Create New Token**.
4. A token file `kaggle.json` will be downloaded containing your username and key.

### B1. Setting it up on Google Colab (Recommended)
1. Open `notebooks/capstone_training.ipynb` in Google Colab.
2. In the left vertical toolbar, click the **Secrets** icon (🔑 Key symbol).
3. Click **Add new secret**:
   - **Name:** `KAGGLE_API_TOKEN`
   - **Value:** Paste your Kaggle API key or token string.
4. Toggle **Notebook access** to **ON**.
5. The notebook reads this automatically using `google.colab.userdata`.

### B2. Setting it up Locally for `scripts/download_data_local.py` (Optional)
1. Copy `.env.example` to `.env`:
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` and paste your token:
   ```env
   KAGGLE_API_TOKEN=your_actual_token_here
   ```
3. `.gitignore` ensures `.env` is never committed.

---

## 🚀 Section C: Training on Google Colab (Free T4 GPU)

All heavy computation (dataset downloading, image cleaning, merging 14 Classes, 70/15/15 splitting, CNN training, and ONNX conversion) is automated inside `notebooks/capstone_training.ipynb`.

### Step-by-Step Colab Instructions:
1. **Upload Notebook:** Go to [Google Colab](https://colab.research.google.com/), click **Upload**, and select `notebooks/capstone_training.ipynb`.
2. **Enable GPU:**
   - In Colab's menu, click **Runtime** $\rightarrow$ **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** and click **Save**.
3. **Upload Custom Photos (`own_photos.zip`):**
   - In your Google Drive, create a folder named `capstone`.
   - Take custom photos for `vada_pav`, `poha`, and `not_food` (empty plates, hands, tables, trays, cutlery, backgrounds; 60–100 photos each).
   - Zip them into `own_photos.zip` (containing `vada_pav/`, `poha/`, and `not_food/` subfolders).
   - Upload to `/content/drive/MyDrive/capstone/own_photos.zip`.
4. **Run All Cells Sequentially:**
   - Click **Runtime** $\rightarrow$ **Run all** (or execute cells one by one from top to bottom).
   - Authorize Google Drive when prompted.

### Artifacts generated in your Google Drive (`/MyDrive/capstone/`):
- `mobilenetv3.pth` — Trained weights of MobileNetV3-Large (14 Classes).
- `classes.json` — The 16 class index-to-name mapping.
- `food_model.onnx` — High-performance ONNX model.
- `food_model_int8.onnx` — Dynamically quantized INT8 model for embedded hardware.
- `dataset_split.zip` — The 70/15/15 dataset archive (cached for fast reruns).
- `training_curves.png` — Loss and accuracy plots.
- `confusion_matrix_mobilenetv3-large.png` — High-resolution confusion matrix.
- `model_comparison.csv` — Parameters, file size, accuracy, and latency benchmarks.
- `gradcam_samples.png` — Visual Explainable AI overlays.

---

## 🖥️ Section D: Moving Trained Weights & Running the Demos

Once training is complete in Colab:

### Step 1: Download weights from Google Drive
Open Google Drive in your web browser, navigate to the `capstone` folder, and download:
1. `mobilenetv3.pth`
2. `classes.json` (and optionally `food_model.onnx`)

### Step 2: Copy to `app/model/`
Place the downloaded files into:
```text
smart-canteen/app/model/
├── README.txt
├── classes.json
├── mobilenetv3.pth
└── food_model.onnx      # (optional for ONNX backend)
```

---

### Demo 1: Streamlit Interactive Web Application
Run the Streamlit app with photo upload, snapshot camera, and browser live stream:
```powershell
streamlit run app/app.py
```
Your browser will open `http://localhost:8501`.
- **📁 Upload Photo tab:** Browse and test food photos from your laptop.
- **📸 Webcam Snapshot tab:** Built-in snapshot camera for instant point-and-click capture.
- **🎥 Live Camera Stream tab:** Real-time in-browser video classification via `streamlit-webrtc`.
- **Safety check:** If non-food or confidence is below threshold, shows `"No food / Unknown food"` and prevents billing.
- **Explainability:** Expand **Grad-CAM Heatmap** to see the neural attention map.

---

### Demo 2: Fast OpenCV Live Camera Demo (GPU Accelerated)
For maximum FPS, low latency, and direct NVIDIA GPU acceleration:

#### Run with PyTorch (uses NVIDIA RTX 5050 GPU via CUDA):
```powershell
python app/live_demo.py --backend torch
```

#### Run with ONNX Runtime:
```powershell
python app/live_demo.py --backend onnx
```

#### Additional options:
```powershell
# Custom confidence threshold (e.g. 70%) and external camera index 1:
python app/live_demo.py --backend torch --threshold 0.70 --camera 1
```

#### Keyboard Controls in Live Demo:
| Key | Action |
| :---: | :--- |
| **`[SPACE]`** | **Add current food item to bill** (blocked if `not_food` or below threshold) |
| **`[C]`** | **Clear running bill** back to ₹0 |
| **`[S]`** | **Save screenshot** to `outputs/` folder |
| **`[Q]` / `[ESC]`** | **Quit live demo** and cleanly release webcam |

---

## 📷 Section E: How to Photograph Custom Canteen Data

To train regional canteen dishes and the negative non-food class:

1. **Required Categories & Quantities:**
   - `vada_pav`: 60–100 photos
   - `poha`: 60–100 photos
   - `not_food`: 60–100 photos (empty plates, cafeteria tables, steel trays, hands, cutlery, people, cups, wallets, cafeteria background)
2. **Local Generalization Test Set:**
   - Keep 15–20 separate photos per class in `local_test/` (never seen during training) to verify cafeteria lighting generalization.
3. **Diversity is Key:**
   - **Plates & Packaging:** Steel thalis, paper plates, disposable wraps, plastic trays.
   - **Angles:** 90° top-down, 45° perspective, close-ups.
   - **Lighting:** Cafeteria fluorescent lights, sunny outdoor tables, counter shadows.
4. **Folder Organization before zipping:**
   ```text
   own_photos/
   ├── vada_pav/     # 60-100 .jpg images
   ├── poha/         # 60-100 .jpg images
   └── not_food/     # 60-100 .jpg images (empty plates, hands, tables, trays, backgrounds)
   ```
   Zip into `own_photos.zip` and upload to `/content/drive/MyDrive/capstone/own_photos.zip`.

---

## 🎓 Section F: Viva Voce Preparation (10 Q&As)

Here are the 10 most common external examiner viva questions with simple, accurate answers:

### Q1: What is Transfer Learning and why did you use it?
> **Answer:** Transfer learning takes a model trained on a massive dataset (ImageNet with 1.4 million images across 1,000 categories) and repurposes its learned visual features (edges, textures, shapes) for a new task. Because food datasets in a college lab are relatively small (500 images/class), training a deep CNN from scratch leads to severe overfitting. Transfer learning enables 85%+ accuracy with fast convergence and minimal data.

### Q2: Why did you choose MobileNetV3 and EfficientNet-B0 over ResNet-50 or VGG-16?
> **Answer:** VGG-16 (~138M params, 520MB) and ResNet-50 (~25M params, 100MB) are too heavy for low-cost canteen point-of-sale machines. MobileNetV3-Large (~5.4M params, ~20MB) and EfficientNet-B0 (~5.3M params, ~21MB) utilize **depthwise separable convolutions** and **squeeze-and-excitation blocks**, providing comparable accuracy with 80% fewer parameters and sub-30ms CPU latency.

### Q3: What is the two-phase training strategy (Warmup vs Fine-Tuning)?
> **Answer:**  
> - **Phase 1 (Warmup):** We freeze the pretrained feature extractor backbone and train only the newly added classification head for 5 epochs with $lr = 10^{-3}$. This prevents large, random gradients in the new head from destroying pretrained filters.  
> - **Phase 2 (Fine-Tuning):** We unfreeze all layers and train end-to-end for 12 epochs with a much smaller learning rate ($lr = 10^{-4}$) to gently adjust edge and texture filters to Indian dishes.

### Q4: Why did you use Label Smoothing in Cross-Entropy Loss?
> **Answer:** Standard cross-entropy encourages the model to be infinitely confident ($p \rightarrow 1.0$) on hard one-hot targets. Label smoothing ($0.1$) softens the target vector (e.g., $0.9$ for the true class and small probabilities for the rest). This regularizes the network and prevents overconfidence on visually ambiguous foods like poha vs fried rice.

### Q5: What is Top-3 Accuracy and why is it important for food recognition?
> **Answer:** Top-1 requires the #1 predicted class to match exactly. Top-3 accuracy checks whether the true class is within the top 3 highest probabilities. In food recognition, dishes like *vada pav* and *burger*, or *chole bhature* and *puri bhaji* share similar textures. Top-3 accuracy (>95%) provides cashier validation while maintaining robust usability.

### Q6: How does the system prevent billing mistakes when non-food items appear?
> **Answer:** We implement a **two-tier safety defense**:
> 1. **16th Explicit `not_food` Class:** Trained on empty plates, cafeteria tables, steel trays, cutlery, hands, and backgrounds. If predicted as `not_food`, price is ₹0 and adding to bill is strictly prohibited.
> 2. **Confidence Safety Threshold (Default 60%):** If maximum softmax probability is below 60%, the app rejects the prediction as `"No food / Unknown food"` and disables billing.

### Q7: What is Grad-CAM and what does it show?
> **Answer:** **Grad-CAM (Gradient-weighted Class Activation Mapping)** calculates the gradients of the target class score with respect to the feature maps of the final convolutional layer. It generates a 2D coarse heatmap highlighting which spatial regions of the image influenced the neural network's decision. It proves our model is looking at the food (e.g. samosa crust) rather than the plate or cafeteria tablecloth.

### Q8: What is ONNX and why convert the PyTorch model to it?
> **Answer:** **ONNX (Open Neural Network Exchange)** is an open format for representing machine learning models. Exporting PyTorch to ONNX decouples the model from Python/PyTorch dependencies and allows execution in optimized runtimes like ONNX Runtime, OpenVINO, or TensorRT, yielding 1.5x–3x faster CPU inference.

### Q9: What is INT8 Dynamic Quantization?
> **Answer:** Standard neural network weights use 32-bit floating point numbers (FP32). Dynamic quantization converts weights to 8-bit integers (INT8) during storage and matrix multiplications. This reduces model file size by ~4x (from ~20MB down to ~5MB) and speeds up CPU computation on low-power devices with negligible loss in accuracy (<0.5%).

### Q10: How does your system handle continuous improvement (Active Learning)?
> **Answer:** The app includes a **"Prediction Wrong?"** human-in-the-loop feedback mechanism. If the cashier or customer sees a wrong prediction, they select the correct dish and click submit. The image is timestamped and saved into `feedback/<true_class>/`. Periodically, these hard examples are merged into the training set for model fine-tuning.

---

## 👥 Section G: Team Work Distribution

| Team Member | Roll No. | Core Responsibilities |
| :--- | :--- | :--- |
| **Atharva Korde** | Roll 08 | **Dataset collection & preprocessing:** Downloading Kaggle Indian food dataset, Food-101 subset curation, image integrity verification (PIL), merging & capping at 500 images/class, 70/15/15 split, collecting `not_food` background negative samples. |
| **Dhruv Khedkar** | Roll 13 | **Model development & training:** Baseline CNN from scratch, MobileNetV3-Large & EfficientNet-B0 transfer learning pipelines for 14 Classes, warmup & fine-tuning learning rate schedules, checkpointing. |
| **Divyam Sharma** | Roll 14 | **Evaluation, metrics & local dataset:** Top-1/Top-3 benchmarking, Macro-F1 computation, confusion matrix heatmaps, inference latency benchmarks (GPU vs CPU), collecting & evaluating `local_test` canteen photos. |
| **Divyatwa Kiranpure** | Roll 15 | **Application deployment & live camera:** Streamlit interactive web application with WebRTC, standalone OpenCV GPU live camera demo (`live_demo.py`), shared predictor module (`predictor.py`), running billing table & calorie calculator, Grad-CAM visualization. |

---

## 🏆 Project Deliverables Checklist (20 Marks)
- [x] Functional Colab Notebook (`notebooks/capstone_training.ipynb`) with GPU acceleration (14 Classes)
- [x] 3 Models trained and compared: Baseline CNN vs MobileNetV3 vs EfficientNet-B0
- [x] 16 Categories: 15 Indian/International food classes + dedicated `not_food` background class
- [x] Evaluation metrics: Top-1, Top-3, Macro-F1, Confusion Matrix, and Training Curves
- [x] Interactive Streamlit App (`app/app.py`) with Upload, Webcam Snapshot, and WebRTC Live Camera
- [x] Standalone OpenCV Live Camera Demo (`app/live_demo.py`) with GPU acceleration, 10-frame smoothing, and hotkeys
- [x] Shared Predictor Module (`app/predictor.py`) eliminating code duplication
- [x] Safety Rejection Threshold (<60% confidence alert & `not_food` billing block)
- [x] Explainable AI (Grad-CAM heatmap expander)
- [x] ONNX Model Export & Benchmark
- [x] Active Learning Feedback logger (`feedback/`)
- [x] Menu rates & nutritional data (`app/menu.csv`)
