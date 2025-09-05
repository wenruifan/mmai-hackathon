import os
import glob
import pandas as pd
from pathlib import Path
from PIL import Image

# ---- Configure your dataset root ----
DATA_PATH = r"D:\MMAI'25 Hackathon"
FILES_PATH = os.path.join(
    DATA_PATH,
    "mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0",
    "files",
)

# -----------------------------
# 1) Build paths from metadata
# -----------------------------
def get_cxr_paths(base_path: str, csv_path: str | None = None):
    """Return DataFrame with a resolved `path` to each JPG.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    # Auto-find metadata CSV if not provided
    if csv_path is None:
        candidates = []
        for pat in [
            os.path.join(DATA_PATH, "**", "*metadata*.csv"),
            os.path.join(DATA_PATH, "**", "mimic-cxr*-metadata*.csv"),
            os.path.join(DATA_PATH, "**", "mimic-cxr-2.0.0-metadata.csv"),
        ]:
            candidates.extend(glob.glob(pat, recursive=True))
        if not candidates:
            raise FileNotFoundError(
                "Could not auto-find a metadata CSV. Please pass csv_path explicitly."
            )
        csv_path = min(candidates, key=len)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Detect dicom id column
    col_candidates = [c for c in df.columns]
    lower_map = {c.lower(): c for c in col_candidates}
    id_col = None
    for key in ("dicom_id", "dicom", "image_id"):
        if key in lower_map:
            id_col = lower_map[key]
            break
    if id_col is None:
        raise KeyError(
            "No suitable ID column found. Expected one of: 'dicom_id', 'dicom', 'image_id'."
        )

    # Scan all JPGs once
    jpg_map = {p.stem: p for p in base.rglob("*.jpg")}

    # Map id -> jpg path
    df["path"] = df[id_col].astype(str).str.strip().map(lambda x: str(jpg_map.get(x, "")))

    # Keep only matches
    before = len(df)
    df = df[df["path"] != ""].copy()
    after = len(df)
    print(f"Matched {after}/{before} rows using ID column '{id_col}'.")

    return df

# -----------------------------
# 2) Load a single image
# -----------------------------
def load_cxr_image(path: str, to_gray: bool = True):
    """Load a single CXR image as PIL Image (grayscale by default)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path)
    return img.convert("L") if to_gray else img.convert("RGB")

# ---------
# Example
# ---------
if __name__ == "__main__":
    # Option A: Let the function auto-discover your metadata CSV under DATA_PATH
    df = get_cxr_paths(FILES_PATH)

    # Option B: Provide the exact CSV path if you know it
    # csv_file = os.path.join(DATA_PATH, "mimic-cxr-2.0.0-metadata.csv")
    # df = get_cxr_paths(FILES_PATH, csv_file)

    print(df.head())
    if not df.empty:
        img = load_cxr_image(df.iloc[0]["path"])  # PIL Image
        # img.show()  # uncomment to preview
