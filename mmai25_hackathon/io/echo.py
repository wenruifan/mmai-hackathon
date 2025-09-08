import os
from pathlib import Path

import pandas as pd

# ---- Configure your dataset root ----
DATA_PATH = r"your_data_path_here"
ECHO_DIR = "mimic-iv-echo-0.1.physionet.org"
FILES_PATH = os.path.join(DATA_PATH, ECHO_DIR)


# -----------------------------
# 1) Build paths from echo-record-list.csv
# -----------------------------
def get_echo_paths(base_path: str, csv_path: str):
    """
    Return a DataFrame with resolved ECHO DICOM paths.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    df = pd.read_csv(csv_path)
    if "dicom_filepath" not in df.columns:
        raise KeyError("CSV must contain a 'dicom_filepath' column (e.g., files/.../133.dcm)")

    df["dicom_filepath"] = df["dicom_filepath"].astype(str).str.strip()
    df["dcm_path"] = df["dicom_filepath"].map(lambda rp: str(base / rp))

    before = len(df)
    df = df[df["dcm_path"].map(os.path.exists)].copy()
    after = len(df)
    print(f"Matched {after}/{before} DICOM files present on disk.")
    return df


# -----------------------------
# 2) Load a single ECHO DICOM
# -----------------------------
def load_echo_dicom(dcm_path: str):
    """
    Load an ECHO DICOM (supports multi-frame cine) using pydicom.

    Returns:
      frames: np.ndarray with shape (T, H, W) for multi-frame, or (1, H, W) for single image
      meta:   dict with handy fields (Rows, Columns, NumberOfFrames, FrameTime, CineRate, etc.)

    Requires: pip install pydicom
    """
    if not dcm_path or not os.path.exists(dcm_path):
        raise FileNotFoundError(f"DICOM not found: {dcm_path}")

    try:
        import pydicom
    except ImportError as e:
        raise ImportError("pydicom is required. Install with: pip install pydicom") from e

    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array

    if arr.ndim == 2:
        arr = arr[None, ...]

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = (arr * slope + inter).astype(arr.dtype)

    meta = {
        "Rows": int(getattr(ds, "Rows", arr.shape[-2])),
        "Columns": int(getattr(ds, "Columns", arr.shape[-1])),
        "NumberOfFrames": int(getattr(ds, "NumberOfFrames", arr.shape[0])),
        "FrameTime_ms": float(getattr(ds, "FrameTime", 0.0)) if hasattr(ds, "FrameTime") else None,
        "CineRate": int(getattr(ds, "CineRate", 0)) if hasattr(ds, "CineRate") else None,
        "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", None),
        "BitsAllocated": int(getattr(ds, "BitsAllocated", 0)) if hasattr(ds, "BitsAllocated") else None,
        "PixelRepresentation": int(getattr(ds, "PixelRepresentation", 0))
        if hasattr(ds, "PixelRepresentation")
        else None,
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
        "Modality": getattr(ds, "Modality", None),
        "SeriesDescription": getattr(ds, "SeriesDescription", None),
    }
    return arr, meta


# ---------
# Example
# ---------
if __name__ == "__main__":
    # Point to your echo-record-list.csv
    csv_file = os.path.join(FILES_PATH, "echo-record-list.csv")

    df = get_echo_paths(FILES_PATH, csv_file)
    print(df.head())

    if not df.empty:
        frames, meta = load_echo_dicom(df.iloc[0]["dcm_path"])
        print("Frames shape:", frames.shape)
        print(
            "Meta:",
            {
                k: meta[k]
                for k in (
                    "NumberOfFrames",
                    "Rows",
                    "Columns",
                    "FrameTime_ms",
                    "CineRate",
                )
            },
        )
