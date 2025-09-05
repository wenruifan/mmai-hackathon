import os
import pandas as pd
from typing import List, Tuple, Optional, Dict

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
    - Keeps only rows where label_column_name is non-null
    - (Optional) keeps only rows whose label value ∈ label_column_values
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
    Load one omics modality:
    - Reads table with index_col=0
    - Transposes to shape [samples x features]
    - Sets neat index/column names
    Returns (name, df)
    """
    df = pd.read_csv(path, sep=sep, index_col=0).transpose()
    df.index.names = ["sample"]
    df.columns.names = ["feature"]
    return name, df


def remove_missing_labels(df: pd.DataFrame, valid_sample_index: pd.Index) -> pd.DataFrame:
    """Keep only rows whose sample IDs appear in the label index; sort rows."""
    df = df[df.index.isin(valid_sample_index)]
    return df.sort_index(axis=0)


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Drop columns (features) with ≥10% missing values (keep cols with >=90% non-NA).
    2) Fill any remaining NA by column mean.
    3) Assert no missing remains.
    """
    min_non_na = int(len(df.index) * 0.9)
    df = df.dropna(axis=1, thresh=min_non_na)
    df = df.fillna(df.mean())

    if df.isna().any().any():
        raise ValueError("The modality contains missing values. Please handle them before proceeding.")
    return df


def normalize_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min–max normalize each feature independently: (x - min) / (max - min).
    Note: If a column is constant, denominator becomes 0 → will yield NaN/inf.
    We replace those with 0.0 safely.
    """
    col_min = df.min()
    col_max = df.max()
    denom = (col_max - col_min).replace(0, pd.NA)
    df_norm = (df - col_min) / denom
    # Replace NA/inf from constant columns with 0.0
    df_norm = df_norm.fillna(0.0)
    return df_norm


def remove_low_variance_features(df: pd.DataFrame, threshold: Optional[float]) -> pd.DataFrame:
    """Optionally drop columns with variance < threshold."""
    if threshold is None:
        return df
    return df.loc[:, df.var() >= threshold]


def print_omics_shapes(omics: List[Tuple[str, pd.DataFrame]], title: str):
    print(f"\nOmic modality shape ({title}):")
    for name, df in omics:
        print(f" - {name} shape: {df.shape}")
    print()


def check_label_indices_availability(
    labels: pd.Series,
    omics: List[Tuple[str, pd.DataFrame]]
) -> Tuple[bool, List[str]]:
    """
    Check whether each sample in labels is present in at least one modality.
    Returns:
      - is_all_available: bool
      - labels_to_remove: list of sample IDs to drop from labels
    """
    combined_indices = set()
    for _, df in omics:
        combined_indices.update(df.index)

    not_found = [sid for sid in labels.index if sid not in combined_indices]
    print(f"Number of labels not in combined omics indices: {len(not_found)}")
    return (len(not_found) == 0), not_found


def map_labels_to_int(labels: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """Map unique label strings to 0..K-1 integers, return (mapped_series, mapping_dict)."""
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
    print("Saving the processed data...")
    omics_names = [os.path.basename(p[1]) for p in data_paths]
    if save_path is None:
        save_path = "./"

    if save_label and labels is not None and label_path is not None:
            
        label_out = os.path.join(save_path, "label.csv")
        os.makedirs(os.path.dirname(label_out), exist_ok=True)
        # print(f" - label path: {label_out}, label shape: {labels.shape}")
        # labels.drop([labels.columns[0]], axis=0, inplace=True)
        labels.to_csv(label_out, index=False, header=False)

    for (orig_path, name), (_, df) in zip(data_paths, omics):
        name_out_path = os.path.join(save_path, f"{name}_names.csv")
        feat_out_path = os.path.join(save_path, f"{name}_feat.csv")
        os.makedirs(os.path.dirname(name_out_path), exist_ok=True)
        names = list(df.columns[1:])
        name_df = pd.DataFrame(names)
        name_df.to_csv(name_out_path, index=False, header=False)
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
    var_threshold: Optional[List[Optional[float]]] = None
    ):
    """
    Run the full preprocessing pipeline as a linear script.
    This mirrors the steps in the MultiOmicsPreprocessor class but is structured
    as a single script for clarity.
    """
    # ----- User-configurable inputs (same semantics as your class version) -----
    
    num_omics = len(data_paths)  # not strictly needed but kept for parity
    sep = "\t"



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
    omics = [(name, remove_missing_labels(df, labels.index)) for name, df in omics]
    print_omics_shapes(omics, "After missing label removal")

    # ----------------- Step 4: Drop labels not present in ANY modality ----------
    is_ok, labels_to_remove = check_label_indices_availability(labels, omics)
    if labels_to_remove:
        labels = labels.drop(labels_to_remove)
    print(f"Are all labels available in omic modalities? {is_ok}\n")

    # After dropping labels, ensure omics use the same (now reduced) label set
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

    # ----------------- Step 9: Save outputs ------------------------------------
    save_processed(
        omics=omics,
        labels=labels_mapped,
        data_paths=data_paths,
        label_path=label_path,
        save_label=True,
        save_path=save_path,
    )