# Converts your own photos to clean JPGs and builds own_photos.zip for the Colab notebook.
#
# Expected input (what you already have):
#   C:\Users\<you>\Downloads\own photos\poha\       (photos of poha)
#   C:\Users\<you>\Downloads\own photos\vada_pav\   (photos of vada pav)
#   C:\Users\<you>\Downloads\own photos\not_food\   (empty plates, hands, tables, objects...)
#
# Output: C:\Users\<you>\Downloads\own_photos.zip  (class folders at the TOP level)
#
# Run from the project root:  python src/prepare_own_photos.py
import shutil, zipfile
from pathlib import Path
from PIL import Image

try:
    import pillow_heif                      # only needed for iPhone HEIC / AVIF photos
    pillow_heif.register_heif_opener()
except ImportError:
    print("[Note] pillow-heif not installed: .heic/.avif files will be skipped "
          "(pip install pillow-heif to include them).")

DOWNLOADS = Path.home() / "Downloads"
SRC = DOWNLOADS / "own photos"              # folder that contains poha / vada_pav / not_food
DST = DOWNLOADS / "own_photos_clean"        # temporary clean output folder
ZIP = DOWNLOADS / "own_photos.zip"          # final zip to upload to Colab / Google Drive

def norm(s): return s.lower().replace("_", "").replace(" ", "").replace("-", "")
CLASSES = {"vadapav": "vada_pav", "poha": "poha", "notfood": "not_food"}
EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".bmp"}

if not SRC.exists():
    raise SystemExit(f"Source folder not found: {SRC}")

if DST.exists():
    shutil.rmtree(DST)                      # start clean so re-runs never duplicate images
DST.mkdir(parents=True)

counts = {}
for folder in SRC.rglob("*"):
    if folder.is_dir() and norm(folder.name) in CLASSES:
        cls = CLASSES[norm(folder.name)]
        out = DST / cls
        out.mkdir(exist_ok=True)
        n = counts.get(cls, 0)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            try:
                img = Image.open(f).convert("RGB")
                img.thumbnail((1024, 1024))
                n += 1
                img.save(out / f"{cls}_{n:03d}.jpg", quality=90)
            except Exception as e:
                print("Skipped:", f.name, "->", e)
        counts[cls] = n

print("\nImages per class:", counts)
for c in CLASSES.values():
    if counts.get(c, 0) < 20:
        print(f"WARNING: {c} has fewer than 20 images! The notebook will refuse to run.")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in DST.rglob("*.jpg"):
        z.write(p, p.relative_to(DST).as_posix())   # class folders at top level, Linux-safe
print("Created:", ZIP, "| size MB:", round(ZIP.stat().st_size / 1e6, 1))
