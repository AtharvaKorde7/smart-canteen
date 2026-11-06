#!/usr/bin/env python3
"""
Smart Canteen Billing Assistant: Capstone Submission Packager
============================================================

Purpose:
--------
Automates packaging of the Smart Canteen capstone project repository into
the university-mandated submission archive format:
    [GROUP_ID]_V-B.ZIP

Exclusion Rules:
----------------
- Python virtual environments (venv/, .venv/, env/)
- Git internal directories (.git/)
- Python bytecaches (__pycache__/, *.pyc, *.pyo, *.pyd)
- Local environment secrets (.env, .env.local)
- Raw dataset dumps (data/, indian_raw/, own_raw/, *.zip)
- Any temporary runtime outputs (outputs/, *.log)
- Any individual file exceeding 100 MB
- The generated zip archive itself

Mandatory Inclusion Verification:
---------------------------------
- app/model/mobilenetv3.pth (Production model weights)
- app/model/classes.json (14 classes list)
- notebooks/capstone_training_kaggle.ipynb (Kaggle training notebook)
- DATASET.md & README.md
- requirements.txt & packages.txt

Usage:
------
    python scripts/make_submission_zip.py --group-id <GROUP_ID>
    Example: python scripts/make_submission_zip.py --group-id G12
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# Excluded directory names (exact match)
EXCLUDED_DIR_NAMES = {
    "venv",
    ".venv",
    "env",
    "ENV",
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
    "data",
    "indian_raw",
    "food-101",
    "own_raw"
}

# Excluded file extensions (temporary files and auto-downloaded YOLO weights)
EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
    ".pt"
}

# Max file size limit: 100 MB
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


def should_exclude(file_path: Path, output_zip_path: Path) -> bool:
    """
    Determines whether a file should be excluded from the submission zip.
    """
    rel_path = file_path.relative_to(REPO_ROOT)

    # Exclude the generated zip file itself or any other zip archives
    if file_path.resolve() == output_zip_path.resolve():
        return True
    if file_path.suffix.lower() == ".zip":
        return True

    # Exclude .env secrets (preserve .env.example)
    if file_path.name == ".env" or (file_path.name.startswith(".env.") and file_path.name != ".env.example"):
        return True

    # Exclude bytecode extensions
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    # Exclude prohibited directories in path parts
    for part in rel_path.parts[:-1]:
        if part in EXCLUDED_DIR_NAMES:
            return True

    # Exclude files larger than 100 MB
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            print(f"[SKIP] Excluding large file (>100MB): {rel_path} ({file_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
    except OSError:
        return True

    return False


def build_submission_zip(group_id: str, output_dir: Path = REPO_ROOT) -> Path:
    """
    Builds the submission zip archive [GROUP_ID]_V-B.ZIP and validates its contents.
    """
    clean_group_id = group_id.strip().upper()
    zip_filename = f"{clean_group_id}_V-B.ZIP"
    output_zip_path = output_dir / zip_filename

    print("=" * 70)
    print(f"PACKAGING CAPSTONE SUBMISSION ARCHIVE: {zip_filename}")
    print("=" * 70)
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Output Archive  : {output_zip_path}")

    # Remove existing zip if present
    if output_zip_path.exists():
        output_zip_path.unlink()

    packed_files = []

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(REPO_ROOT):
            # Prune excluded directories in-place to avoid unnecessary traversal
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]

            for file in files:
                full_path = Path(root) / file
                if not should_exclude(full_path, output_zip_path):
                    rel_path = full_path.relative_to(REPO_ROOT)
                    # Use POSIX forward slashes for cross-platform archive compatibility
                    arcname = rel_path.as_posix()
                    zipf.write(full_path, arcname)
                    packed_files.append((arcname, full_path.stat().st_size))

    # Verification: check mandatory files
    archive_namelist = set()
    with zipfile.ZipFile(output_zip_path, "r") as zipf:
        archive_namelist = set(zipf.namelist())

    mandatory_files = [
        "app/model/mobilenetv3.pth",
        "app/model/classes.json",
        "notebooks/capstone_training_kaggle.ipynb",
        "README.md",
        "DATASET.md",
        "requirements.txt",
        "packages.txt",
        "app/app.py",
        "app/billing.py",
        "app/detector.py",
        "app/predictor.py"
    ]

    missing = [m for m in mandatory_files if m not in archive_namelist]

    total_size_mb = output_zip_path.stat().st_size / (1024 * 1024)

    print("\nArchived File Manifest:")
    print("-" * 70)
    for name, size in sorted(packed_files):
        print(f"  • {name:<50s} ({size / 1024:8.1f} KB)")
    print("-" * 70)
    print(f"Total files packed : {len(packed_files)}")
    print(f"Archive total size : {total_size_mb:.2f} MB")
    print("-" * 70)

    if missing:
        print(f"\n[ERROR] Missing mandatory submission file(s) in archive: {missing}")
        sys.exit(1)

    print("\n[VERIFIED] All mandatory files and model weights successfully verified inside archive!")
    print(f"[SUCCESS] Submission archive created at: {output_zip_path}\n")
    return output_zip_path


def main():
    parser = argparse.ArgumentParser(description="Build Smart Canteen Capstone Submission ZIP.")
    parser.add_argument(
        "--group-id",
        type=str,
        required=True,
        help="University assigned Group ID (e.g. G12)."
    )
    args = parser.parse_args()
    build_submission_zip(args.group_id)


if __name__ == "__main__":
    main()
