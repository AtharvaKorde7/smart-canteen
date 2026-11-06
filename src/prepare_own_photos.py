"""
Smart Canteen Billing Assistant: Own Photos Preprocessing Utility (src/prepare_own_photos.py)
=============================================================================================

Purpose:
--------
Standardizes self-collected canteen dish photographs (`vada_pav`, `poha`, and `not_food`)
by converting varied phone formats (HEIC, AVIF, PNG, WebP) to clean RGB JPEGs with
normalized dimensions (max 1024x1024). Packages the clean dataset into `own_photos.zip`
with top-level class folders, ready for direct upload to Kaggle/Colab training pipelines.

Inputs and Outputs:
-------------------
- Expected Input:
    `C:\\Users\\<username>\\Downloads\\own photos\\vada_pav\\`
    `C:\\Users\\<username>\\Downloads\\own photos\\poha\\`
    `C:\\Users\\<username>\\Downloads\\own photos\\not_food\\`
- Output:
    `C:\\Users\\<username>\\Downloads\\own_photos.zip` (clean zip archive)

Usage:
------
    python src/prepare_own_photos.py
"""

import shutil
import zipfile
from pathlib import Path
from typing import Dict
from PIL import Image

# Register HEIF/AVIF image opener for mobile photos if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("[Note] pillow-heif not installed: .heic/.avif files will be skipped "
          "(pip install pillow-heif to include them).")

# Default directory paths
DOWNLOADS = Path.home() / "Downloads"
SRC = DOWNLOADS / "own photos"              # Folder that contains poha / vada_pav / not_food
DST = DOWNLOADS / "own_photos_clean"        # Temporary clean output folder
ZIP = DOWNLOADS / "own_photos.zip"          # Final zip to upload to Kaggle / Google Drive


def normalize_class_name(folder_name: str) -> str:
    """
    Normalizes arbitrary folder casing, spaces, and punctuation to match canonical class keys.

    Args:
        folder_name (str): Raw directory name from filesystem.

    Returns:
        str: Lowercase alphanumeric string for dictionary lookup.
    """
    return folder_name.lower().replace("_", "").replace(" ", "").replace("-", "")


CLASSES = {
    "vadapav": "vada_pav",
    "poha": "poha",
    "notfood": "not_food"
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".bmp"}


def process_and_package_photos():
    """
    Scans raw source folders, cleans and standardizes each image to JPEG,
    and packages them into a top-level zip archive.
    """
    if not SRC.exists():
        raise SystemExit(f"Source folder not found: {SRC}\n"
                         f"Please place your collected photos under: {SRC}")

    if DST.exists():
        shutil.rmtree(DST)  # Start clean so re-runs never duplicate images
    DST.mkdir(parents=True)

    counts: Dict[str, int] = {}

    # Traverse all folders under SRC to find matching class folders
    for folder in SRC.rglob("*"):
        if folder.is_dir() and normalize_class_name(folder.name) in CLASSES:
            cls = CLASSES[normalize_class_name(folder.name)]
            out = DST / cls
            out.mkdir(exist_ok=True)
            n = counts.get(cls, 0)
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                try:
                    # Convert to standard RGB JPEG and bound dimensions to max 1024x1024
                    img = Image.open(f).convert("RGB")
                    img.thumbnail((1024, 1024))
                    n += 1
                    img.save(out / f"{cls}_{n:03d}.jpg", quality=90)
                except Exception as e:
                    print("Skipped:", f.name, "->", e)
            counts[cls] = n

    print("\nProcessed images per class:", counts)
    for c in CLASSES.values():
        if counts.get(c, 0) < 20:
            print(f"WARNING: {c} has fewer than 20 images! The training notebook expects at least 20 images per class.")

    # Build the zip archive with class folders directly at the top level
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in DST.rglob("*.jpg"):
            z.write(p, p.relative_to(DST).as_posix())  # Top-level class folders, POSIX path separators

    print("Created:", ZIP, "| size MB:", round(ZIP.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    process_and_package_photos()
