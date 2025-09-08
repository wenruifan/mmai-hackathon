import os
import pandas as pd
from pathlib import Path

# ---- Configure your dataset root ----
DATA_PATH = r"your_data_path_here"
ECG_DIR = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
FILES_PATH = os.path.join(DATA_PATH, ECG_DIR)

# -----------------------------
# 1) Build paths from record_list
# -----------------------------
def get_ecg_paths(base_path: str, csv_path: str):
    """
    Return a DataFrame with resolved ECG file paths from record_list.csv.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise KeyError("CSV must contain a 'path' column.")

    abs_heas, abs_dats = [], []
    for rel in df["path"].astype(str):
        abs_heas.append(str(base / f"{rel}.hea"))
        abs_dats.append(str(base / f"{rel}.dat"))

    df["hea_path"] = abs_heas
    df["dat_path"] = abs_dats

    before = len(df)
    df = df[df["hea_path"].map(os.path.exists) & df["dat_path"].map(os.path.exists)].copy()
    after = len(df)
    print(f"Matched {after}/{before} records with both .hea and .dat present.")

    return df

# -----------------------------
# 2) Load a single ECG record
# -----------------------------
def load_ecg_record(hea_path: str):
    """Load an ECG record given a .hea file path using wfdb."""
    if not hea_path or not os.path.exists(hea_path):
        raise FileNotFoundError(f".hea not found: {hea_path}")

    try:
        import wfdb
    except ImportError as e:
        raise ImportError("Install wfdb with: pip install wfdb") from e

    rec = os.path.splitext(hea_path)[0]  # drop extension
    signals, fields = wfdb.rdsamp(rec)
    return signals, fields

# ---------
# Example
# ---------
if __name__ == "__main__":
    csv_file = os.path.join(FILES_PATH, "record_list.csv")
    df = get_ecg_paths(FILES_PATH, csv_file)
    print(df.head())

    if not df.empty:
        sig, meta = load_ecg_record(df.iloc[0]["hea_path"])
        print("Signals shape:", sig.shape)
        print("Sampling freq:", meta.get("fs"))
