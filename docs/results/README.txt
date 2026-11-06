================================================================================
SMART CANTEEN - EVALUATION & EXPERIMENTAL RESULTS
================================================================================

This directory contains evaluation benchmarks, confusion matrices, training plots,
and real-world photo detection outputs generated from the deep learning pipeline.

Expected training evaluation files (from Kaggle Tesla T4 training notebook):
1. `training_curves.png`
   - Loss and accuracy curves across training and validation epochs (warmup and fine-tuning).
2. `confusion_matrix_mobilenetv3-large.png`
   - Normalized confusion matrix across all 14 classes on the 572-image held-out test split.
3. `gradcam_samples.png`
   - Grid of Grad-CAM heatmaps across test dishes explaining model decision boundaries.
4. `model_comparison.csv`
   - CSV benchmarking Baseline CNN vs MobileNetV3-Large vs EfficientNet-B0 on parameters,
     size, top-1/top-3 accuracy, macro-F1, and CPU/GPU inference latency.

Real canteen photo detection results (generated via tests/evaluate_real_photos.py):
5. `annotated_samosa_chai_01.jpg`
   - Production bounding box detection overlay on real photo with Chai and Samosa.
6. `debug_samosa_chai_01.jpg`
   - Candidate proposal inspection view showing all proposals, status, and reject reasons.
================================================================================

