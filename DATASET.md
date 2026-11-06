# Smart Canteen Billing Assistant — Dataset Documentation & Provenance

This document details the complete data acquisition, directory mapping, custom canteen collection, preprocessing, and train/validation/test splitting methodology for the **14 classes** used in the Smart Canteen Billing Assistant.

---

## 1. Dataset Overview

The dataset consists of **3,738 food and background images** across 14 target classes (13 Indian and canteen food dishes + 1 background rejection class):

| Target Class | Source | Category Type | Image Count |
| :--- | :--- | :--- | :---: |
| `burger` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `chai` | Kaggle (`indian-food-classification`) | Beverage / Dish | ~300–400 |
| `chole_bhature` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `fried_rice` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `idli` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `jalebi` | Kaggle (`indian-food-classification`) | Dessert / Dish | ~300–400 |
| `masala_dosa` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `momos` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `pav_bhaji` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `pizza` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `samosa` | Kaggle (`indian-food-classification`) | Dish | ~300–400 |
| `vada_pav` | Self-Collected / Custom | Regional Dish | 60 |
| `poha` | Self-Collected / Custom | Regional Dish | 60 |
| `not_food` | Self-Collected / Custom | Negative Class | ~50 |
| **Total** | **Combined Dataset** | **14 Classes** | **3,738** |

---

## 2. Public Benchmark Source: Kaggle Indian Food Classification

- **Dataset Name**: Indian Food Classification
- **Kaggle Slug**: `l33tc0d3r/indian-food-classification`
- **URL**: [https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification](https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification)
- **License**: Open Database License (ODbL) / Public Domain for academic research

### Download Command (Kaggle CLI)
```bash
kaggle datasets download -d l33tc0d3r/indian-food-classification -p data/indian_raw --unzip
```

### Folder Name Mapping
The Kaggle archive contains multi-word folder names that are normalized to snake_case in our pipeline:

| Kaggle Source Subfolder | Normalized Internal Class | Notes |
| :--- | :--- | :--- |
| `burger` / `Burger` | `burger` | Capped up to 500 images |
| `chai` / `Chai` | `chai` | Hot tea in cups / glasses |
| `chole bhature` / `chole_bhature` | `chole_bhature` | Bhature with chole curry |
| `fried rice` / `fried_rice` | `fried_rice` | Indo-Chinese fried rice |
| `idli` / `Idli` | `idli` | Steamed rice cakes |
| `jalebi` / `Jalebi` | `jalebi` | Spiral fried dessert |
| `masala dosa` / `masala_dosa` | `masala_dosa` | Crispy crepe with potato filling |
| `momos` / `Momos` | `momos` | Steamed / fried dumplings |
| `pav bhaji` / `pav_bhaji` | `pav_bhaji` | Spiced vegetable mash with buns |
| `pizza` / `Pizza` | `pizza` | Baked sliced pizza |
| `samosa` / `Samosa` | `samosa` | Triangular fried pastry |

---

## 3. Custom Canteen & Local Photos (`own_photos.zip`)

To support regional campus canteen favourites (`vada_pav`, `poha`) and reject non-food background clutter (`not_food`), the team gathered custom photos.

### Provenance & Attribution
The photos for the classes `vada_pav`, `poha`, and `not_food` were taken by our team in the college canteen and the nearby area. All other classes originate from the public Kaggle Indian Food Classification dataset described above.

### Own Photos Structure:
```text
own_photos/
├── vada_pav/     # 60 images of fresh and packaged vada pav
├── poha/         # 60 images of breakfast poha with sev / coriander
└── not_food/     # ~50 images: empty plates, canteen tables, cash counter, hands, cutlery, cups
```

### Building `own_photos.zip` with `src/prepare_own_photos.py`
To clean, standardize, and zip your custom photos before uploading to Kaggle:

1. Place your raw camera photos in your local Downloads folder under:
   ```text
   C:\Users\<username>\Downloads\own photos\vada_pav\
   C:\Users\<username>\Downloads\own photos\poha\
   C:\Users\<username>\Downloads\own photos\not_food\
   ```
2. Run the preparation script:
   ```powershell
   python src/prepare_own_photos.py
   ```
3. The script:
   - Sanitizes and converts images (including iPhone HEIC / AVIF if `pillow-heif` is installed) to standard JPEG.
   - Resizes long edge to a maximum of 1024 px for fast uploading.
   - Verifies each class has at least 20 valid images.
   - Packages them into `Downloads/own_photos.zip` with class folders at the root.

---

## 4. Train / Validation / Test Splitting Methodology

- **Split Ratio**: 70% Train / 15% Validation / 15% Test
- **Seed**: `42` (fixed deterministic seed for reproducibility)
- **Stratification**: Class-balanced proportional sampling

### Exact Split Partition Counts:
- **Total Dataset**: 3,738 images across 14 classes
- **Training Set (70%)**: 2,611 images
- **Validation Set (15%)**: 555 images
- **Held-Out Test Set (15%)**: 572 images

The held-out 572-image test split is never seen during model training or hyperparameter tuning. All reported benchmark metrics (Top-1, Top-3, Macro-F1, confusion matrices) are computed strictly on this 572-image test set.

---

## 5. Augmentation & Input Standardization Pipeline

All images are preprocessed through PyTorch `torchvision.transforms`:

### Training Transformations:
1. `transforms.RandomResizedCrop(224, scale=(0.8, 1.0))` — simulates variation in camera-to-plate distance.
2. `transforms.RandomHorizontalFlip(p=0.5)` — accounts for plate orientation.
3. `transforms.RandomRotation(degrees=15)` — simulates handheld angle tilts.
4. `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — simulates diverse canteen lighting conditions (fluorescent, daylight, shadows).
5. `transforms.ToTensor()` — converts pixel range [0, 255] to [0.0, 1.0].
6. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` — ImageNet distribution alignment.

### Validation & Test Transformations:
1. `transforms.Resize(256)`
2. `transforms.CenterCrop(224)`
3. `transforms.ToTensor()`
4. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

