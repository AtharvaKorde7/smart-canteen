"""
Smart Canteen Billing Assistant
Optional Local Dataset Downloader for Inspection

NOTE:
This script is OPTIONAL. The complete model training runs on Google Colab (with a free T4 GPU).
You only need to run this on your laptop if you want to inspect raw images from the Kaggle dataset locally.
Do NOT download Food-101 locally, as it is over 5 GB.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env file
# Looks for .env in current folder or parent project folder
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"[INFO] Loaded environment settings from {env_path}")
else:
    load_dotenv()
    print("[INFO] No .env file found at project root; checking system environment variables...")

# 2. Check Kaggle credentials
kaggle_token = os.environ.get("KAGGLE_API_TOKEN")
if not kaggle_token or kaggle_token.strip() == "paste_your_token_here":
    print("\n[ERROR] KAGGLE_API_TOKEN is missing or not set!")
    print("Please follow these steps:")
    print("1. Log in to https://www.kaggle.com and go to Account / Settings.")
    print("2. Click 'Create New Token' to download your Kaggle token.")
    print("3. Copy .env.example to .env and paste your token inside.")
    print("4. Re-run this script.\n")
    sys.exit(1)

# Set Kaggle environment variable for the kaggle library
os.environ["KAGGLE_API_TOKEN"] = kaggle_token.strip()

# 3. Check available disk space (require at least 10 GB)
target_dir = Path(__file__).resolve().parent.parent / "data" / "indian_raw"
target_dir.parent.mkdir(parents=True, exist_ok=True)

# Check disk space on target drive
total, used, free = shutil.disk_usage(target_dir.parent)
free_gb = free / (1024 ** 3)
print(f"[INFO] Free disk space: {free_gb:.2f} GB")

if free_gb < 10.0:
    print(f"\n[WARNING] You have only {free_gb:.2f} GB free disk space.")
    print("Downloading and unzipping the dataset requires at least 10 GB.")
    choice = input("Do you still want to proceed? (y/N): ").strip().lower()
    if choice != "y":
        print("[INFO] Download cancelled by user.")
        sys.exit(0)

# 4. Download dataset using Kaggle API
DATASET_NAME = "l33tc0d3r/indian-food-classification"
print(f"\n[INFO] Starting download of Kaggle dataset: {DATASET_NAME}")
print(f"[INFO] Destination folder: {target_dir}")

target_dir.mkdir(parents=True, exist_ok=True)

try:
    # Import kaggle library
    import kaggle
    kaggle.api.authenticate()
    print("[INFO] Kaggle authentication successful!")
    
    # Download and unzip directly
    kaggle.api.dataset_download_files(DATASET_NAME, path=str(target_dir), unzip=True)
    print("\n[SUCCESS] Download and extraction completed successfully!")
except ImportError:
    print("\n[ERROR] The 'kaggle' python package is not installed.")
    print("Please run: pip install kaggle")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] An error occurred while downloading: {e}")
    print("Tip: If using the Kaggle API CLI, you can also run:")
    print(f"kaggle datasets download -d {DATASET_NAME} -p \"{target_dir}\" --unzip")
    sys.exit(1)

# 5. Inspect folder names and count images
print("\n" + "=" * 50)
print("DATASET INSPECTION: FOLDER NAMES & IMAGE COUNTS")
print("=" * 50)

valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
category_counts = {}

for root, dirs, files in os.walk(target_dir):
    # Only count files in leaf subfolders
    image_files = [f for f in files if Path(f).suffix.lower() in valid_extensions]
    if image_files:
        rel_folder = os.path.relpath(root, target_dir)
        category_counts[rel_folder] = len(image_files)

if category_counts:
    total_images = sum(category_counts.values())
    for folder, count in sorted(category_counts.items()):
        print(f"  • {folder:35s}: {count:4d} images")
    print("-" * 50)
    print(f"Total categories found: {len(category_counts)}")
    print(f"Total images found:     {total_images}")
else:
    print("[WARNING] No image files found. Check extraction folder.")
print("=" * 50)
print("\n[DONE] You can inspect the images locally, but remember to run the full training in Google Colab!")
