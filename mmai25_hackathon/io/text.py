import os
import pandas as pd
from pathlib import Path
from typing import Literal, Optional, Tuple

# ---- Configure your dataset root ----
DATA_PATH = r"your_data_path_here"
TEXT_DIR  = "mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note"
NOTE_PATH = os.path.join(DATA_PATH, TEXT_DIR)

# -----------------------------
# 1) Load notes (radiology or discharge)
# -----------------------------
def get_text_notes(
    base_note_path: str,
    subset: Literal["radiology", "discharge"] = "radiology",
    include_detail: bool = False,
    keep_cols: Optional[Tuple[str, ...]] = ("note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"),
) -> pd.DataFrame:
    """
    Load free-text clinical notes.

    Parameters
    ----------
    base_note_path : str
        Folder that contains the 4 CSVs (radiology.csv, radiology_detail.csv, discharge.csv, discharge_detail.csv).
    subset : {'radiology','discharge'}
        Which note family to load.
    include_detail : bool
        If True, left-join the corresponding *_detail.csv on ['note_id','subject_id'] and add detail columns.
    keep_cols : tuple of str | None
        Columns to keep from the main notes CSV. If None, keep all columns.

    Returns
    -------
    pd.DataFrame
        Notes DataFrame. If include_detail=True, extra columns from *_detail are merged (field_name, field_value, field_ordinal).
    """
    base = Path(base_note_path)
    if not base.exists():
        raise FileNotFoundError(f"Notes folder not found: {base}")

    main_name = f"{subset}.csv"
    detail_name = f"{subset}_detail.csv"

    main_csv= base / main_name
    detail_csv = base / detail_name

    if not main_csv.exists():
        raise FileNotFoundError(f"Missing main notes CSV: {main_csv}")

    df = pd.read_csv(main_csv)

    # Ensure required ID columns exist
    required_ids = {"note_id", "subject_id"}
    if not required_ids.issubset(set(df.columns)):
        raise KeyError(f"{main_name} must contain columns: {required_ids}")

    # Keep requested columns if provided and present
    if keep_cols is not None:
        cols_present = [c for c in keep_cols if c in df.columns]
        # Guarantee IDs stay even if not in keep_cols
        for col in ["note_id", "subject_id"]:
            if col not in cols_present and col in df.columns:
                cols_present.insert(0, col)
        df = df[cols_present].copy()

    # Optionally join detail
    if include_detail:
        if not detail_csv.exists():
            raise FileNotFoundError(f"Requested detail join but missing: {detail_csv}")
        det = pd.read_csv(detail_csv)
        # minimal check
        if not required_ids.issubset(set(det.columns)):
            raise KeyError(f"{detail_name} must contain columns: {required_ids}")
        # Typical detail columns: field_name, field_value, field_ordinal
        df = df.merge(det, how="left", on=["note_id", "subject_id"])

    # Report simple stats
    total = len(df)
    has_text = "text" in df.columns
    if has_text:
        nonempty = (df["text"].astype(str).str.strip() != "").sum()
        print(f"Loaded {total} {subset} notes ({nonempty} with non-empty text).")
    else:
        print(f"Loaded {total} {subset} note rows (no 'text' column in selection).")

    return df

# -----------------------------
# 2) Fetch text for a given note_id
# -----------------------------
def load_text_note(df: pd.DataFrame, note_id: int, return_meta: bool = False):
    """
    Retrieve the free-text for a given note_id from a DataFrame returned by get_text_notes().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by get_text_notes().
    note_id : int
        Note identifier to look up.
    return_meta : bool
        If True, return (text, row_dict). Otherwise return text only.

    Returns
    -------
    text : str | (str, dict)
        The note text; optionally with the note's metadata as a dictionary.
    """
    if "note_id" not in df.columns:
        raise KeyError("DataFrame must contain 'note_id' column.")
    rows = df[df["note_id"] == note_id]
    if rows.empty:
        raise KeyError(f"note_id {note_id} not found.")

    row = rows.iloc[0]
    if "text" not in df.columns:
        raise KeyError("The DataFrame does not include a 'text' column. Re-load with keep_cols including 'text'.")
    txt = str(row["text"])
    if return_meta:
        meta = row.to_dict()
        return txt, meta
    return txt

# ---------
# Example
# ---------
if __name__ == "__main__":
    # Radiology notes
    radi_df = get_text_notes(NOTE_PATH, subset="radiology", include_detail=False)
    print(radi_df.head(2))

    if not radi_df.empty:
        sample_text = load_text_note(radi_df, note_id=int(radi_df.iloc[0]["note_id"]))
        print("Radiology sample text (truncated):", sample_text[:200], "...")

    # Discharge notes
    disc_df = get_text_notes(NOTE_PATH, subset="discharge", include_detail=True)  # include detail join
    print(disc_df.head(2))

    if not disc_df.empty:
        sample_text = load_text_note(disc_df, note_id=int(disc_df.iloc[0]["note_id"]))
        print("Discharge sample text (truncated):", sample_text[:200], "...")
