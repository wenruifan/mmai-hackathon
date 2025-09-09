"""
Multi-omics preprocessing (linear pipeline)
===========================================

What this script does
---------------------
Given:
  1) A *label table* (CSV/TSV) with sample IDs as the first column (index)
     and at least one column containing class labels.
  2) One or more *omics tables* (CSV/TSV), each with features as rows and
     samples as columns (common in bio datasets).

It will:
  - Load & filter labels (drop NA; optionally keep a subset of label values).
  - Load omics, transpose to [samples x features], align to the label index.
  - Drop labels that are missing from **all** modalities.
  - Clean missing values per modality (drop columns with >=10% NA, then mean-impute).
  - (Optional) Min–max normalize features per modality independently.
  - (Optional) Remove features below a variance threshold (per modality).
  - Map string labels → integers 0..K-1.
  - Save:
      save_path/label.csv            (mapped integer labels; see NOTE below)
      save_path/{name}_names.csv     (feature names for each modality)
      save_path/{name}_feat.csv      (feature matrix per modality)

IMPORTANT shape assumptions
---------------------------
- Label table: index = sample IDs (unique), column `label_column_name` holds labels.
- Each omics file: index = feature IDs, columns = sample IDs BEFORE transpose.
  After loading we call `.transpose()` so final shape is [samples x features].

File format notes
-----------------
- All readers use `index_col=0`. Your first column must be the sample (for labels)
  or feature (for omics) index.
- `sep` defaults to tab (`"\t"`). Change `sep` if you have commas.

Saving notes (be careful!)
--------------------------
- This script currently writes labels as a Series *without* index or header
  (see `labels.to_csv(..., index=False, header=False)`). That **discards sample IDs**.
  Keep this if your downstream code expects a plain vector. If you need the sample
  IDs preserved, change to `index=True, header=True` and update downstream.

Quick start (example)
---------------------
data_paths = [
    ("/path/to/mRNA.tsv",        "mRNA"),
    ("/path/to/methylation.tsv", "methylation"),
    ("/path/to/miRNA.tsv",       "miRNA"),
]
run_preprocessing_pipeline(
    label_path="/path/to/labels.tsv",
    data_paths=data_paths,
    save_path="./BRCA/processed/",
    label_column_name="PAM50Call_RNAseq",
    label_column_values=None,
    clean_missing=True,
    normalize=True,
    var_threshold=[None, 1e-6, 1e-6]  # one per modality (or None to skip)
)
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------
# Small, reusable helper funcs
# -----------------------------


def load_label_table(
    label_path: str,
    sep: str = "\t",
    label_column_name: Optional[str] = None,
    label_column_values: Optional[List[str]] = None,
) -> pd.Series:
    """
    Load label table and return a 1D Series of labels indexed by sample ID.

    Parameters
    ----------
    label_path : str
        Path to the label file (CSV/TSV). The first column must be sample IDs
        (will be used as the index).
    sep : str, default="\t"
        Field separator used in the file.
    label_column_name : str
        Name of the column that contains class labels. Required.
    label_column_values : list[str] | None
        If provided, keep only rows whose label ∈ this set (useful to focus on
        selected classes, e.g., five PAM50 subtypes).

    Returns
    -------
    labels : pd.Series
        1D Series (index = sample IDs, values = label strings), sorted by index.

    Behavior
    --------
    - Drops rows with missing labels (NA in `label_column_name`).
    - Optionally filters labels to a user-specified subset.
    - Prints class counts for sanity-checking.

    Common pitfalls
    ---------------
    - If your sample IDs are not in the first column, set `index_col=0` will be wrong.
      Fix the file or change the reader accordingly.
    """
    label_df = pd.read_csv(label_path, sep=sep, index_col=0)
    if label_column_name is None:
        raise ValueError("label_column_name must be provided for label filtering.")

    # Keep only samples with non-null labels
    label_df = label_df[label_df[label_column_name].notnull()]
    labels = label_df[label_column_name]

    print(f"Labels size after removing missing values: {labels.shape}")

    # (Optional) filter label values to a given subset
    if label_column_values is not None:
        labels = labels[labels.isin(label_column_values)]
        print(f"Labels size after value filtering: {labels.shape}")

    # Sort by index to keep stable order
    labels = labels.sort_index(axis=0)
    print(f"\nInput class labels:\n{labels.value_counts(dropna=True)}\n")
    return labels


def load_single_omics(path: str, name: str, sep: str = "\t") -> Tuple[str, pd.DataFrame]:
    """
    Load one omics modality.

    Steps
    -----
    - Reads a matrix with `index_col=0` (features as rows, samples as columns).
    - Transposes to shape [samples x features] (machine-learning friendly).
    - Assigns index/column names for readability.

    Parameters
    ----------
    path : str
        Path to the omics file (CSV/TSV). First column must be feature IDs.
    name : str
        Short identifier for this modality (e.g., "mRNA", "methylation").
    sep : str
        Field separator.

    Returns
    -------
    (name, df) : tuple[str, pd.DataFrame]
        Name and the processed DataFrame (index = sample IDs, columns = feature IDs).
    """
    df = pd.read_csv(path, sep=sep, index_col=0).transpose()
    df.index.names = ["sample"]
    df.columns.names = ["feature"]

    # Sanity checks (non-fatal; change to asserts if you want strictness)
    # - Ensure no duplicate sample IDs
    if not df.index.is_unique:
        print(f"[WARN] Duplicate sample IDs found in {name}. Consider deduplication.")

    # - Ensure no duplicate feature names
    if not df.columns.is_unique:
        print(f"[WARN] Duplicate feature names found in {name}. Consider deduplication.")

    return name, df


def remove_missing_labels(df: pd.DataFrame, valid_sample_index: pd.Index) -> pd.DataFrame:
    """
    Keep only rows (samples) present in `valid_sample_index`, then sort rows.

    Why this matters
    ----------------
    Ensures every sample in each modality has a corresponding label entry and
    aligns row order for consistent saving/merging downstream.
    """
    df = df[df.index.isin(valid_sample_index)]
    return df.sort_index(axis=0)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values per modality.

    Procedure
    ---------
    1) Drop columns (features) with ≥10% missing values (keep columns with >=90% non-NA).
    2) Fill any remaining NA by column mean (simple imputation).
    3) Error if any NA remains (guards against silent failures).

    Tunables
    --------
    - Adjust the 10% threshold by changing `min_non_na` if needed.
    """
    min_non_na = int(len(df.index) * 0.9)  # keep columns with >=90% observed
    df = df.dropna(axis=1, thresh=min_non_na)
    df = df.fillna(df.mean())

    if df.isna().any().any():
        raise ValueError("The modality contains missing values. Please handle them before proceeding.")
    return df


def normalize_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min–max normalize each feature independently: (x - min) / (max - min).

    Notes
    -----
    - If a column is constant (max == min), denominator = 0.
      We temporarily replace the 0 denom with NA and then fill NA with 0.0,
      effectively leaving that feature as all zeros (safe default).

    When to use
    -----------
    - Useful when features have different scales. If your downstream method
      is scale-invariant (e.g., tree models), normalization may be optional.
    """
    col_min = df.min()
    col_max = df.max()
    denom = (col_max - col_min).replace(0, pd.NA)
    df_norm = (df - col_min) / denom
    # Replace NA/inf from constant columns with 0.0
    df_norm = df_norm.fillna(0.0)
    return df_norm


def remove_low_variance_features(df: pd.DataFrame, threshold: Optional[float]) -> pd.DataFrame:
    """
    Optionally drop columns with variance < threshold.

    Parameters
    ----------
    threshold : float | None
        If None, do nothing. Otherwise, retain only features whose variance >= threshold.
        Typical tiny thresholds: 1e-8, 1e-6, etc.

    Tip
    ---
    - Use per-modality thresholds to reflect different numeric scales before normalization.
    """
    if threshold is None:
        return df
    return df.loc[:, df.var() >= threshold]


def print_omics_shapes(omics: List[Tuple[str, pd.DataFrame]], title: str):
    """
    Utility to log modality shapes for quick inspection.
    """
    print(f"\nOmic modality shape ({title}):")
    for name, df in omics:
        print(f" - {name} shape: {df.shape}")
    print()


def check_label_indices_availability(
    labels: pd.Series, omics: List[Tuple[str, pd.DataFrame]]
) -> Tuple[bool, List[str]]:
    """
    Check whether each sample in labels is present in at least one modality.

    Returns
    -------
    is_all_available : bool
        True if every labeled sample is present in ≥1 modality.
    labels_to_remove : list[str]
        Labeled sample IDs that do not appear in any modality (to be dropped).

    Why this matters
    ----------------
    Keeps the label vector consistent with the available data. Otherwise you'd
    carry labels for samples you never actually feed to a model.
    """
    combined_indices = set()
    for _, df in omics:
        combined_indices.update(df.index)

    not_found = [sid for sid in labels.index if sid not in combined_indices]
    print(f"Number of labels not in combined omics indices: {len(not_found)}")
    return (len(not_found) == 0), not_found


def map_labels_to_int(labels: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Map unique label strings to integers 0..K-1 (order = first appearance order).

    Returns
    -------
    mapped : pd.Series
        Same index as `labels`, values are ints in [0, K-1], name="Class".
    mapping : dict[str, int]
        Dictionary of {original_label -> int_code} for reproducibility/logging.

    Notes
    -----
    - If you need a deterministic order (e.g., alphabetical), change the
      enumeration order to `for i, label in enumerate(sorted(unique))`.
    """
    unique = labels.unique()
    mapping = {label: i for i, label in enumerate(unique)}
    mapped = labels.map(mapping).rename("Class")
    print(f"Mapped class labels: {mapping}")
    print(f"\nMapped class labels:\n{mapped.value_counts(dropna=False)}\n")
    return mapped, mapping


def save_processed(
    omics: List[Tuple[str, pd.DataFrame]],
    labels: Optional[pd.Series],
    data_paths: List[Tuple[str, str]],
    label_path: Optional[str],
    save_label: bool = True,
    save_path: Optional[str] = None,
):
    """
    Save processed outputs to disk.

    Parameters
    ----------
    omics : list[(name, df)]
        Each df is [samples x features], aligned to the final label index.
    labels : pd.Series | None
        Integer-mapped labels (index = samples). If None or `save_label=False`,
        the label file will not be saved.
    data_paths : list[(orig_path, name)]
        Original (path, name) pairs, used here only to keep naming consistent.
    label_path : str | None
        Kept for parity with the class interface (not used for writing).
    save_label : bool
        Whether to write `label.csv`.
    save_path : str | None
        Directory to save files. Defaults to current directory.

    Writes
    ------
    - label.csv
      CURRENTLY written **without** index/header to keep compatibility with
      some pipelines that expect a raw vector. Change to include index if needed.
    - {name}_names.csv
      1D list of feature names (as they appear in the df.columns) *excluding*
      the 1st column if you later decide to prepend something. Right now it
      simply dumps the columns[1:], mirroring your original behavior.
      (See NOTE below.)
    - {name}_feat.csv
      The full feature matrix [samples x features], written without index/header.

    NOTE about `{name}_names.csv`
    -----------------------------
    Your original code uses `names = list(df.columns[1:])`. That drops the first
    feature name. Keep as-is if this is intentional for downstream compatibility.
    If not intentional, change to `names = list(df.columns)`.

    Tip
    ---
    Add versioning to `save_path` (e.g., include a timestamp or config hash).
    """
    print("Saving the processed data...")
    if save_path is None:
        save_path = "./"

    # ---- Save labels (optional) ----
    if save_label and labels is not None and label_path is not None:
        label_out = os.path.join(save_path, "label.csv")
        os.makedirs(os.path.dirname(label_out), exist_ok=True)
        # WARNING: This drops sample IDs and the column name.
        labels.to_csv(label_out, index=False, header=False)

    # ---- Save each modality ----
    for (orig_path, name), (_, df) in zip(data_paths, omics):
        name_out_path = os.path.join(save_path, f"{name}_names.csv")
        feat_out_path = os.path.join(save_path, f"{name}_feat.csv")
        os.makedirs(os.path.dirname(name_out_path), exist_ok=True)

        # NOTE: Preserve your original slicing behavior.
        names = list(df.columns[1:])
        name_df = pd.DataFrame(names)
        name_df.to_csv(name_out_path, index=False, header=False)

        # Save feature matrix (no index/header) for compact downstream loading
        df.to_csv(feat_out_path, index=False, header=False)


# -----------------------------
# Linear pipeline
# -----------------------------


def run_preprocessing_pipeline(
    label_path: str,
    data_paths: List[Tuple[str, str]],
    save_path: str,
    label_column_name: str = "PAM50Call_RNAseq",
    label_column_values: Optional[List[str]] = None,
    clean_missing: bool = True,
    normalize: bool = True,
    var_threshold: Optional[List[Optional[float]]] = None,
):
    """
    Run the full preprocessing pipeline as a linear script.

    Parameters
    ----------
    label_path : str
        Path to the label file (CSV/TSV). Must have sample IDs in the first column.
    data_paths : list[(path, name)]
        List of (omics_file_path, modality_name) tuples.
        Each omics file must have features as rows and samples as columns (pre-transpose).
    save_path : str
        Directory to write processed outputs.
    label_column_name : str
        Column in the label file that contains class labels to predict.
    label_column_values : list[str] | None
        If provided, restrict to these label classes (others are dropped).
    clean_missing : bool
        If True, drop high-NA features and mean-impute remaining NA per modality.
    normalize : bool
        If True, min–max normalize features per modality.
    var_threshold : list[float|None] | None
        Per-modality variance thresholds. Must match `len(data_paths)` if provided.
        Use None to skip variance filtering for a modality.

    Output files
    ------------
    - save_path/label.csv
    - save_path/{name}_names.csv
    - save_path/{name}_feat.csv

    Logging
    -------
    Prints shapes and class counts at each key step for traceability.

    Notes
    -----
    - The order of saved rows follows the sorted label index.
    - If you later join modalities, ensure consistent row order across files.
    """
    # ----- User-configurable inputs (same semantics as your class version) -----
    sep = "\t"  # adjust if using CSV with commas

    # ----------------- Step 1: Load labels -----------------
    labels = load_label_table(
        label_path,
        sep=sep,
        label_column_name=label_column_name,
        label_column_values=label_column_values,
    )

    # ----------------- Step 2: Load omics ------------------
    omics: List[Tuple[str, pd.DataFrame]] = []
    for path, name in data_paths:
        omics.append(load_single_omics(path, name, sep=sep))
    print_omics_shapes(omics, "Raw/Input omic modalities")

    # ----------------- Step 3: Remove samples w/ missing labels -----------------
    # Align each modality to the (possibly filtered) label index
    omics = [(name, remove_missing_labels(df, labels.index)) for name, df in omics]
    print_omics_shapes(omics, "After missing label removal")

    # ----------------- Step 4: Drop labels not present in ANY modality ----------
    is_ok, labels_to_remove = check_label_indices_availability(labels, omics)
    if labels_to_remove:
        # If a sample has a label but appears in none of the modalities, drop it.
        labels = labels.drop(labels_to_remove)
    print(f"Are all labels available in omic modalities? {is_ok}\n")

    # After dropping such labels, re-align all modalities again
    omics = [(name, remove_missing_labels(df, labels.index)) for name, df in omics]

    # ----------------- Step 5: Clean missing values (per modality) --------------
    if clean_missing:
        omics = [(name, clean_missing_values(df)) for name, df in omics]
        print_omics_shapes(omics, "After missing value removal")

    # ----------------- Step 6: Normalize (per modality) -------------------------
    if normalize:
        omics = [(name, normalize_minmax(df)) for name, df in omics]

    # ----------------- Step 7: Remove low-variance features ---------------------
    if var_threshold is not None:
        if len(var_threshold) != len(omics):
            raise ValueError("var_threshold must be a list with the same length as data_paths.")
        pruned = []
        for (name, df), thr in zip(omics, var_threshold):
            pruned.append((name, remove_low_variance_features(df, thr)))
        omics = pruned
        print_omics_shapes(omics, "After low-variance feature removal")

    # ----------------- Step 8: Map labels to integers ---------------------------
    labels_mapped, mapping = map_labels_to_int(labels)
    # Consider saving `mapping` as JSON alongside outputs if reproducibility is critical.

    # ----------------- Step 9: Save outputs ------------------------------------
    save_processed(
        omics=omics,
        labels=labels_mapped,
        data_paths=data_paths,
        label_path=label_path,
        save_label=True,
        save_path=save_path,
    )

    print("Done.")
