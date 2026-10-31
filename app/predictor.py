"""
Smart Canteen Billing Assistant: Food Image Recognition
Shared Predictor Module (app/predictor.py)

Handles:
- Loading PyTorch (.pth) and ONNX (.onnx) models
- GPU auto-detection (NVIDIA CUDA if available, else CPU)
- Standard ImageNet preprocessing (Center-crop, 224x224, Normalization)
- Softmax probability calculation & 10-frame temporal smoothing
- Threshold rejection & 14th class ("not_food") handling (14 classes: 13 dishes + not_food)
- Menu lookup for pricing (INR) and calories (kcal)
- Grad-CAM explainability computation
"""

import os
import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

# Directories
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MENU_FILE = APP_DIR / "menu.csv"
CLASSES_FILE = MODEL_DIR / "classes.json"

# Standard ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================================
# MENU & CLASS NAMES HELPERS
# ==============================================================================
def load_menu(menu_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads the canteen menu CSV with prices (INR) and calories (kcal)."""
    target = Path(menu_path) if menu_path else MENU_FILE
    if target.exists():
        df = pd.read_csv(target)
        df["display_name"] = df["class"].str.replace("_", " ").str.title()
        return df
    # Fallback default 14-class menu if file is missing
    fallback_data = [
        {"class": "burger", "price": 65, "kcal": 420},
        {"class": "chai", "price": 15, "kcal": 80},
        {"class": "chole_bhature", "price": 80, "kcal": 550},
        {"class": "fried_rice", "price": 80, "kcal": 380},
        {"class": "idli", "price": 35, "kcal": 160},
        {"class": "jalebi", "price": 40, "kcal": 300},
        {"class": "masala_dosa", "price": 60, "kcal": 320},
        {"class": "momos", "price": 50, "kcal": 280},
        {"class": "not_food", "price": 0, "kcal": 0},
        {"class": "pav_bhaji", "price": 70, "kcal": 450},
        {"class": "pizza", "price": 120, "kcal": 680},
        {"class": "poha", "price": 30, "kcal": 220},
        {"class": "samosa", "price": 20, "kcal": 260},
        {"class": "vada_pav", "price": 25, "kcal": 290},
    ]
    df = pd.DataFrame(fallback_data)
    df["display_name"] = df["class"].str.replace("_", " ").str.title()
    return df


def load_class_names(classes_path: Optional[Path] = None) -> List[str]:
    """Loads ordered class names from classes.json or menu.csv."""
    target = Path(classes_path) if classes_path else CLASSES_FILE
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
        except Exception:
            pass

    # Fallback to menu classes
    menu_df = load_menu()
    return menu_df["class"].tolist()


# ==============================================================================
# PYTORCH MODEL ARCHITECTURE BUILDER
# ==============================================================================
def build_pytorch_architecture(arch_name: str, num_classes: int):
    """
    Constructs MobileNetV3-Large or EfficientNet-B0 matching Colab training.
    Replaces classifier layer to output num_classes (default 14).
    """
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as e:
        raise ImportError(
            "PyTorch or torchvision is not installed. For NVIDIA RTX 5050 (CUDA 12.8), run:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ) from e

    if arch_name == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    elif arch_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unsupported architecture '{arch_name}'. Use 'mobilenetv3' or 'efficientnet_b0'.")

    return model, target_layer


# ==============================================================================
# UNIFIED PREPROCESSING (MATCHES COLAB TRAINING & EVALUATION)
# ==============================================================================
def preprocess_image(
    image_input: Any,
    is_bgr: bool = False,
    return_debug_image: bool = False
) -> Tuple[np.ndarray, Optional[Image.Image]]:
    """
    Unified preprocessing function used by BOTH uploaded photos and live webcam frames.
    Exactly replicates the training/validation pipeline from notebooks/capstone_training.ipynb:
        Resize(256) -> CenterCrop(224) -> ToTensor() -> ImageNet Normalization
        mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]

    Args:
        image_input: PIL Image (RGB) or numpy array (BGR if is_bgr=True, else RGB).
        is_bgr: Set to True when passing raw OpenCV frames (cv2 reads BGR by default).
        return_debug_image: If True, returns a 224x224 PIL Image reconstructed from the
                            exact normalized tensor seen by the neural network.

    Returns:
        tensor_np: Float32 array of shape [1, 3, 224, 224] for PyTorch or ONNX forward pass.
        debug_pil: Reconstructed 224x224 PIL Image for visual debugging (or None).
    """
    import cv2
    from PIL import ImageOps

    # Step 1: Ensure RGB PIL Image with proper EXIF orientation applied
    if isinstance(image_input, Image.Image):
        pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if is_bgr:
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            rgb_arr = image_input
        # Handle read-only NumPy arrays from newer Pillow versions
        if not rgb_arr.flags.writeable:
            rgb_arr = rgb_arr.copy()
        pil_img = ImageOps.exif_transpose(Image.fromarray(rgb_arr)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}. Expected PIL.Image or np.ndarray.")

    # Step 2: Resize preserving aspect ratio so smaller edge is 256
    # Matches torchvision.transforms.Resize(256)
    w, h = pil_img.size
    if h < w:
        new_h = 256
        new_w = max(224, int(round(w * (256.0 / h))))
    else:
        new_w = 256
        new_h = max(224, int(round(h * (256.0 / w))))
    resized_pil = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Step 3: Center crop to 224x224
    # Matches torchvision.transforms.CenterCrop(224)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    right = left + 224
    bottom = top + 224
    cropped_pil = resized_pil.crop((left, top, right, bottom))

    # Step 4: Convert to float32 [0.0, 1.0] and apply ImageNet mean/std normalization
    arr = np.array(cropped_pil, dtype=np.float32) / 255.0  # shape [224, 224, 3]
    norm_arr = (arr - IMAGENET_MEAN) / IMAGENET_STD         # shape [224, 224, 3]
    chw = np.transpose(norm_arr, (2, 0, 1))                 # shape [3, 224, 224]
    tensor_np = np.expand_dims(chw, axis=0)                 # shape [1, 3, 224, 224]

    # Step 5: Optional debug image reconstruction (denormalized to [0, 255] uint8)
    debug_pil = None
    if return_debug_image:
        denorm = (norm_arr * IMAGENET_STD + IMAGENET_MEAN) * 255.0
        denorm_uint8 = np.clip(denorm, 0, 255).astype(np.uint8)
        debug_pil = Image.fromarray(denorm_uint8)

    return tensor_np, debug_pil


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D or 2D numpy arrays."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# ==============================================================================
# GRAD-CAM EXPLAINABILITY
# ==============================================================================
def compute_gradcam(model, target_layer, image_pil: Image.Image) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Computes Grad-CAM heatmap overlay for PyTorch models.
    Guarantees gradients are enabled even when inference was run under torch.inference_mode().
    Returns (heatmap_rgb_uint8, error_message).
    """
    try:
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from PIL import ImageOps

        # Transpose EXIF and convert to 224x224 RGB
        rgb_pil = ImageOps.exif_transpose(image_pil).convert("RGB").resize((224, 224))
        rgb_float = np.float32(rgb_pil) / 255.0

        # Build tensor
        tensor_img = (rgb_float - IMAGENET_MEAN) / IMAGENET_STD
        tensor_img = np.transpose(tensor_img, (2, 0, 1))
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0).float()

        # If model is on CUDA, pass tensor to CUDA
        device = next(model.parameters()).device
        tensor_img = tensor_img.to(device)

        # Explicitly enable gradients so GradCAM backprop works seamlessly
        with torch.enable_grad():
            tensor_img.requires_grad_(True)
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=tensor_img, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
            return visualization, None
    except Exception as e:
        return None, str(e)


# ==============================================================================
# UNIFIED FOOD PREDICTOR CLASS
# ==============================================================================
class FoodPredictor:
    """
    Shared predictor supporting PyTorch (.pth) and ONNX (.onnx) backends.
    Includes temporal probability smoothing and safety rejection logic.
    """

    def __init__(
        self,
        backend: str = "torch",
        arch_name: str = "mobilenetv3",
        model_path: Optional[str] = None,
        threshold: float = 0.60,
        window_size: int = 10,
        device_str: Optional[str] = None,
    ):
        self.backend = backend.lower().strip()
        self.arch_name = arch_name.lower().strip()
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.class_names = load_class_names()
        self.menu_df = load_menu()
        self.num_classes = len(self.class_names)

        # Sliding window deque for temporal smoothing across frames
        self.prob_history = deque(maxlen=self.window_size)

        # Device setup
        self.device_str = device_str
        self.torch_device = None
        self.model = None
        self.target_layer = None
        self.ort_session = None

        # Resolve model weights path
        self.model_path = self._resolve_model_path(model_path)

        # Initialize requested backend
        if self.backend == "torch":
            self._init_pytorch()
        elif self.backend == "onnx":
            self._init_onnx()
        else:
            raise ValueError(f"Invalid backend '{self.backend}'. Choose 'torch' or 'onnx'.")

    def _resolve_model_path(self, custom_path: Optional[str]) -> Path:
        """Finds model path or auto-detects from app/model/."""
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                return p

        # Auto-detect default names
        if self.backend == "torch":
            preferred = MODEL_DIR / f"{self.arch_name}.pth"
            if preferred.exists():
                return preferred
            # Any .pth in MODEL_DIR
            alt = list(MODEL_DIR.glob("*.pth"))
            if alt:
                return alt[0]
            return preferred
        else:
            preferred = MODEL_DIR / "food_model.onnx"
            if preferred.exists():
                return preferred
            alt = list(MODEL_DIR.glob("*.onnx"))
            if alt:
                return alt[0]
            return preferred

    def _init_pytorch(self):
        """Loads PyTorch model on CUDA if available, else CPU with optimized threading."""
        import torch

        # Device selection: use GPU if available
        if self.device_str:
            self.torch_device = torch.device(self.device_str)
        else:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure CPU threads to prevent thread contention and high CPU latency
        if self.torch_device.type == "cpu":
            num_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
            try:
                torch.set_num_threads(num_threads)
            except Exception:
                pass

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PyTorch weights not found at: '{self.model_path}'\n"
                f"Please ensure 'mobilenetv3.pth' is inside 'app/model/'."
            )

        model, target_layer = build_pytorch_architecture(self.arch_name, self.num_classes)
        state_dict = torch.load(self.model_path, map_location=self.torch_device)
        model.load_state_dict(state_dict)
        model.to(self.torch_device)
        model.eval()

        self.model = model
        self.target_layer = target_layer

    def _init_onnx(self):
        """Loads ONNX model using onnxruntime with automatic fallback to PyTorch if missing."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            # Fall back to PyTorch if onnxruntime is not installed
            print(f"[FoodPredictor] onnxruntime not available ({e}). Falling back to PyTorch...")
            self.backend = "torch"
            self.model_path = self._resolve_model_path(None)
            self._init_pytorch()
            return

        if not self.model_path.exists():
            # If food_model.onnx is missing, smoothly fall back to PyTorch .pth
            print(f"[FoodPredictor] ONNX model '{self.model_path}' not found. Falling back to PyTorch...")
            self.backend = "torch"
            self.model_path = self._resolve_model_path(None)
            self._init_pytorch()
            return

        # Configure ONNX Runtime execution providers
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers and self.device_str != "cpu":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        self.ort_session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)

    def reset_smoothing(self):
        """Clears the temporal probability sliding window."""
        self.prob_history.clear()

    def _forward(self, input_tensor_np: np.ndarray) -> np.ndarray:
        """Executes forward pass through PyTorch or ONNX and returns 1D softmax array."""
        if self.backend == "torch":
            import torch
            tensor = torch.from_numpy(input_tensor_np).to(self.torch_device).float()
            # torch.inference_mode is faster than torch.no_grad in PyTorch >= 1.9
            inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
            with inference_ctx():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return probs
        else:
            # ONNX Runtime forward pass
            input_name = self.ort_session.get_inputs()[0].name
            raw_outputs = self.ort_session.run(None, {input_name: input_tensor_np})
            logits = raw_outputs[0][0]
            return softmax(logits)

    def predict(
        self,
        image_input: Any,
        is_bgr: bool = False,
        smooth: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Unified prediction pipeline for BOTH uploaded photos and webcam frames.
        1. Preprocesses using exact Resize(256) -> CenterCrop(224) -> ImageNet normalization.
        2. Executes neural network forward pass (PyTorch inference_mode or ONNX).
        3. Applies temporal probability smoothing (10-frame sliding window deque).
        4. Checks confidence safety threshold (60%) and rejection of 'not_food'.
        5. Looks up prices and calories from menu.csv.
        6. Extracts top-3 predictions with probabilities.
        """
        # Step 1: Preprocess to [1, 3, 224, 224] using the shared transform
        tensor_np, debug_pil = preprocess_image(
            image_input,
            is_bgr=is_bgr,
            return_debug_image=debug
        )

        # Step 2: Forward pass -> raw softmax probabilities
        raw_probs = self._forward(tensor_np)

        # Step 3: Temporal smoothing over the last N frames
        if smooth:
            self.prob_history.append(raw_probs)
            smoothed_probs = np.mean(self.prob_history, axis=0)
        else:
            smoothed_probs = raw_probs

        # Step 4: Top-1 index and confidence
        top_idx = int(np.argmax(smoothed_probs))
        top_conf = float(smoothed_probs[top_idx])
        top_class_raw = self.class_names[top_idx]

        # Safety & Not-Food Checks:
        # A prediction is valid food ONLY if class != 'not_food' AND confidence >= threshold
        is_not_food = (top_class_raw == "not_food")
        is_confident = (top_conf >= self.threshold)
        is_valid_food = (not is_not_food) and is_confident

        # Step 5: Menu lookup for price and calories
        if is_valid_food:
            match_row = self.menu_df[self.menu_df["class"] == top_class_raw]
            if not match_row.empty:
                display_name = str(match_row.iloc[0]["display_name"])
                price = int(match_row.iloc[0]["price"])
                kcal = int(match_row.iloc[0]["kcal"])
            else:
                display_name = top_class_raw.replace("_", " ").title()
                price = 40
                kcal = 250
            status_text = display_name
        else:
            display_name = "No food / Unknown food"
            price = 0
            kcal = 0
            status_text = "No food / Unknown food"

        # Step 6: Top-3 predictions for transparent multi-class display
        top_k = min(3, len(self.class_names))
        top_indices = np.argsort(smoothed_probs)[::-1][:top_k]
        top_predictions = []
        for idx in top_indices:
            c_raw = self.class_names[idx]
            match_r = self.menu_df[self.menu_df["class"] == c_raw]
            c_price = int(match_r.iloc[0]["price"]) if not match_r.empty else 0
            c_kcal = int(match_r.iloc[0]["kcal"]) if not match_r.empty else 0
            top_predictions.append({
                "class_raw": c_raw,
                "class_name": c_raw.replace("_", " ").title(),
                "confidence": float(smoothed_probs[idx]),
                "price": c_price,
                "kcal": c_kcal,
            })

        result = {
            "class_raw": top_class_raw,
            "class_name": display_name,
            "status_text": status_text,
            "confidence": top_conf,
            "is_food": is_valid_food,
            "is_not_food": is_not_food,
            "is_confident": is_confident,
            "price": price,
            "kcal": kcal,
            "all_probs": smoothed_probs,
            "raw_probs": raw_probs,
            "top_predictions": top_predictions,
            "debug_image": debug_pil,
            "tensor_input": tensor_np,
        }

        return result

    def predict_cv2(
        self,
        frame_bgr: np.ndarray,
        smooth: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Convenience method for OpenCV BGR camera frames."""
        return self.predict(frame_bgr, is_bgr=True, smooth=smooth, debug=debug)

    def predict_pil(
        self,
        image_pil: Image.Image,
        smooth: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Convenience method for PIL images (used by Streamlit upload/snapshot)."""
        return self.predict(image_pil, is_bgr=False, smooth=smooth, debug=debug)

    def predict_array(
        self,
        rgb_array: np.ndarray,
        smooth: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Convenience method for RGB numpy arrays."""
        return self.predict(rgb_array, is_bgr=False, smooth=smooth, debug=debug)
