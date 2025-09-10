from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import pandas as pd

# ---- Configure your dataset root ----
DATA_PATH = r"your_data_path_here"
HOSP_DIR = "mimic-iv-3.1/hosp"
ICU_DIR = "mimic-iv-3.1/icu"

# --------------------------------------
# 1) File maps for HOSP and ICU
# --------------------------------------
HOSP_TABLES = [
    "admissions.csv",
    "diagnoses_icd.csv",
    "drgcodes.csv",
    "emar.csv",
    "emar_detail.csv",
    "hcpcsevents.csv",
    "labevents.csv",
    "microbiologyevents.csv",
    "omr.csv",
    "patients.csv",
    "pharmacy.csv",
    "poe.csv",
    "poe_detail.csv",
    "prescriptions.csv",
    "procedures_icd.csv",
    "provider.csv",
    "services.csv",
    "transfers.csv",
    "d_hcpcs.csv",
    "d_icd_diagnoses.csv",
    "d_icd_procedures.csv",
    "d_labitems.csv",
]

ICU_TABLES = [
    "caregiver.csv",
    "chartevents.csv",
    "d_items.csv",
    "datetimeevents.csv",
    "icustays.csv",
    "ingredientevents.csv",
    "inputevents.csv",
    "outputevents.csv",
    "procedureevents.csv",
]


# --------------------------------------
# 2) Helper to read CSV (with filtering)
# --------------------------------------
def _read_csv(
    filepath: Path,
    keep_cols: Optional[Tuple[str, ...]] = None,
    dtypes: Optional[Dict[str, str]] = None,
    filters: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """
    Wrapper around pd.read_csv with optional column selection and filtering.
    """
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    df = pd.read_csv(filepath, usecols=keep_cols, dtype=dtypes)

    if filters:
        for col, allowed in filters.items():
            df = df[df[col].isin(allowed)]

    return df


# --------------------------------------
# 3) Loader
# --------------------------------------
def get_tabular_mimic(
    base_path: str,
    domain: Literal["hosp", "icu", "both"] = "hosp",
    tables: Optional[Tuple[str, ...]] = None,
    keep_cols: Optional[Dict[str, Tuple[str, ...]]] = None,
    dtypes: Optional[Dict[str, Dict[str, str]]] = None,
    filters: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load tabular data from MIMIC-IV-3.1.

    Parameters
    ----------
    base_path : str
        Root folder containing `hosp` and/or `icu` subfolders.
    domain : {'hosp','icu','both'}
        Which domain to load from.
    tables : tuple of str | None
        Which CSV files to load. If None, load all in the domain(s).
    keep_cols, dtypes : dict
        Per-table settings for column selection and dtypes.
    filters : dict
        Row filters, e.g., {"icustays": {"subject_id": [123]}}.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dict of {table_name: DataFrame}.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Base path not found: {base}")

    selected_domains = []
    if domain in ["hosp", "both"]:
        selected_domains.append((HOSP_DIR, HOSP_TABLES))
    if domain in ["icu", "both"]:
        selected_domains.append((ICU_DIR, ICU_TABLES))

    out = {}
    for dom, file_list in selected_domains:
        dom_path = base / dom
        for fname in file_list:
            table_name = fname.replace(".csv", "")
            if tables and table_name not in tables:
                continue

            filepath = dom_path / fname
            cols = keep_cols.get(table_name) if keep_cols else None
            dtmap = dtypes.get(table_name) if dtypes else None
            fltrs = filters.get(table_name) if filters else None

            df = _read_csv(filepath, keep_cols=cols, dtypes=dtmap, filters=fltrs)
            print(f"Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns.")
            out[table_name] = df

    return out


# --------------------------------------
# 4) Merge helper
# --------------------------------------
def merge_tables(
    frames: Dict[str, pd.DataFrame],
    how: str = "inner",
    on: Optional[Tuple[str, ...]] = None,
    plan: Optional[Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = None,
) -> pd.DataFrame:
    """
    Merge multiple tables.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
    how : str
        Merge type.
    on : tuple of str
        Keys for merge if same for all.
    plan : sequence of (left, right, keys)
        Explicit multi-step merge plan.

    Returns
    -------
    pd.DataFrame
    """
    if plan:
        df = frames[plan[0][0]]
        for left, right, keys in plan:
            df = df.merge(frames[right], how=how, on=keys)
        return df
    else:
        # Simple reduce-style merge
        keys = on or ("subject_id",)
        dfs = list(frames.values())
        df = dfs[0]
        for other in dfs[1:]:
            df = df.merge(other, how=how, on=keys)
        return df


# ---------
# Example
# ---------
if __name__ == "__main__":
    # Example: load ICU stays + admissions
    data = get_tabular_mimic(
        DATA_PATH,
        domain="both",
        tables=("icustays", "admissions"),
        keep_cols={
            "icustays": ("subject_id", "hadm_id", "stay_id"),
            "admissions": ("subject_id", "hadm_id"),
        },
    )

    merged = merge_tables(data, how="inner", on=("subject_id", "hadm_id"))
    print(merged.head())
