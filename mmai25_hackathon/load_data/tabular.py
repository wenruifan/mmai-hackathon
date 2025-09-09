"""
Compact utilities for reading CSVs and aggregating tables.

Exposes `read_tabular(paths, index_cols=None, join='outer')`, which either
concatenates inputs column-wise (no keys) or merges by overlapping join keys
until disjoint groups remain. Non-key column collisions get filename-based
suffixes; columns that only differ by those suffixes are blended back into a
single column **only if their values agree row-wise** (otherwise suffixes are
kept to preserve provenance).
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, overload, Tuple, Union

import pandas as pd


def _blend_suffixed_columns_inplace(df: pd.DataFrame, suffix_tokens: List[str]) -> None:
    """
    Coalesces any columns that are identical up to a filename-based suffix.

    Columns like "caregiver_id_emar" and "caregiver_id_chartevents" (where the
    suffixes come from filename stems) will be examined together with their
    unsuffixed base column "caregiver_id" (if present). If all overlapping
    non-null values across the variants are equal for every row, those variants
    are blended into a single unsuffixed base column; otherwise all suffixed
    columns are kept intact.

    This operation is applied to **all** columns with known suffix tokens, not
    just the declared join keys.

    Args:
        df (pd.DataFrame): The DataFrame to modify in place.
        suffix_tokens (List[str]): Known suffix tokens (e.g., ["_chartevents", "_emar"])
            that were used during merges, typically derived from filename stems.

    Returns:
        None: The operation mutates `df` in place.

    Examples:
        If df has columns ["caregiver_id_emar", "caregiver_id_chartevents"] and
        their values agree row-wise wherever both are non-null, they become a
        single "caregiver_id" column. If any disagreement exists, both suffixed
        columns are retained.
    """
    # 1) Group columns by their base name using the known suffix tokens.
    base_to_cols: Dict[str, List[str]] = {}

    # First pass: find suffixed variants and map them to their base names.
    for col in list(df.columns):
        for token in suffix_tokens:
            if col.endswith(token):
                base = col[: -len(token)]
                if base:  # avoid empty base
                    base_to_cols.setdefault(base, []).append(col)
        # Do not add unsuffixed columns here; we add them in a second pass
        # only if we found at least one suffixed variant for that base.

    # Second pass: include unsuffixed base columns when appropriate.
    for base in list(base_to_cols.keys()):
        if base in df.columns and base not in base_to_cols[base]:
            base_to_cols[base].append(base)

    # 2) For each base, try to blend if all overlapping values are equal.
    for base, cols in base_to_cols.items():
        # Only act if there are at least two variants (suffix and/or base)
        if len(cols) <= 1:
            continue

        # Ensure deterministic order: prefer the unsuffixed base first if present
        cols_sorted = sorted(cols, key=lambda c: (0 if c == base else 1, c))

        # Check for conflicts across all pairs on rows where both are non-null
        conflict = False
        for i in range(len(cols_sorted)):
            s_i = df[cols_sorted[i]]
            for j in range(i + 1, len(cols_sorted)):
                s_j = df[cols_sorted[j]]
                mask = s_i.notna() & s_j.notna() & (s_i != s_j)
                if mask.any():
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            # Keep variants (and suffixes) as-is to preserve provenance.
            continue

        # No conflicts: coalesce to an unsuffixed base column using first-non-null
        blended = None
        for col in cols_sorted:
            blended = df[col] if blended is None else blended.combine_first(df[col])

        if base in df.columns:
            # Overwrite existing base column; drop only the non-base variants.
            df[base] = blended
            to_drop = [c for c in cols_sorted if c != base]
        else:
            # Insert a new base column near the first variant; drop all variants.
            insert_at = df.columns.get_loc(cols_sorted[0])
            df.insert(loc=insert_at, column=base, value=blended)
            to_drop = cols_sorted

        if to_drop:
            df.drop(columns=to_drop, inplace=True)


@overload
def read_tabular(
    paths: Union[List[str], str, List[Path]],
    index_cols: None = ...,
    join: str = "outer",
) -> pd.DataFrame:
    """
    Overload: when `index_cols` is None or empty, returns a single concatenated DataFrame.
    """
    ...


@overload
def read_tabular(
    paths: Union[List[str], str, List[Path]],
    index_cols: List[str],
    join: str = "outer",
) -> Union[
    Tuple[()],  # empty when no frames contain any of the keys
    Tuple[Tuple[str, ...], pd.DataFrame],  # single component (unpacked)
    Tuple[Tuple[Tuple[str, ...], pd.DataFrame], ...],  # multiple components
]:
    """
    Overload: when `index_cols` is provided, returns key-partitioned components.
    """
    ...


def read_tabular(
    paths: Union[str, List[str], List[Path]],
    index_cols: Optional[List[str]] = None,
    join: str = "outer",
) -> Union[
    pd.DataFrame,
    Tuple[()],
    Tuple[Tuple[str, ...], pd.DataFrame],
    Tuple[Tuple[Tuple[str, ...], pd.DataFrame], ...],
]:
    """
    Reads and aggregates tabular CSV files by concatenation or merges.

    Behavior:
      • If `index_cols` is falsy: read all CSVs and concatenate column-wise.
      • Else: group CSVs by the exact subset of provided keys they contain,
        merge within each subset, then greedily merge groups that share keys
        until only disjoint key-components remain.
      • During merges, non-key column collisions receive filename-based suffixes.
        After each merge, **all columns** that differ only by those suffixes are
        blended back into a single base column **only if their overlapping values
        are exactly equal**; otherwise the suffixed variants are kept.

    Args:
        paths (Union[List[str], str]): A list of CSV paths or a single CSV path.
        index_cols (Optional[List[str]]): Candidate join keys. If None/empty,
            inputs are concatenated along columns. Default: None.
        join (str): Merge strategy: one of {"outer", "inner", "left", "right"}.
            Default: "outer".

    Returns:
        Union[
            pd.DataFrame,
            Tuple[()],
            Tuple[Tuple[str, ...], pd.DataFrame],
            Tuple[Tuple[Tuple[str, ...], pd.DataFrame], ...],
        ]:
            - If `index_cols` is falsy: the concatenated DataFrame.
            - Else:
                * `()` if no CSV contains any requested key.
                * `(keys_tuple, DataFrame)` for a single disjoint component.
                * `((keys_tuple, DataFrame), ...)` for multiple components.

    Raises:
        ValueError: If `join` is not one of {"outer", "inner", "left", "right"}.

    Examples:
        >>> read_tabular(["a.csv", "b.csv"])
        DataFrame(...)

        >>> read_tabular(["df1.csv", "df2.csv", "df3.csv"], index_cols=["id", "site"])
        ((('id', 'site'), DataFrame), (('subject',), DataFrame), ...)
    """
    valid_joins = {"outer", "inner", "left", "right"}
    if join not in valid_joins:
        raise ValueError(f"`join` must be one of {valid_joins}, got {join!r}")

    # Normalize and load
    if isinstance(paths, str):
        paths = [paths]
    dataframes: List[pd.DataFrame] = [pd.read_csv(path) for path in paths]
    filename_stems: List[str] = [Path(path).stem for path in paths]
    suffix_tokens: List[str] = [f"_{stem}" for stem in filename_stems]

    # Concatenate mode
    if not index_cols:
        concatenated = pd.concat(dataframes, axis="columns", join=join)
        # Post-process: also blend any identical suffixed columns in concat mode
        _blend_suffixed_columns_inplace(concatenated, suffix_tokens)
        return concatenated

    # Group frames by the exact subset of keys they contain
    frames_by_exact_keys: Dict[Tuple[str, ...], List[Tuple[pd.DataFrame, str]]] = {}
    for df, stem in zip(dataframes, filename_stems):
        key_subset = tuple(col for col in index_cols if col in df.columns)
        if key_subset:
            frames_by_exact_keys.setdefault(key_subset, []).append((df, stem))

    if not frames_by_exact_keys:
        return tuple()  # No frames had any requested keys

    # Merge within each exact key-subset
    merged_groups: List[Tuple[FrozenSet[str], pd.DataFrame, str]] = []
    for key_subset, frames_with_stems in frames_by_exact_keys.items():
        (acc_df, first_stem), *rest = frames_with_stems
        left_suffix = f"_{first_stem}"
        for df, stem in rest:
            acc_df = acc_df.merge(df, on=list(key_subset), how=join, suffixes=(left_suffix, f"_{stem}"))
            # Blend any identical suffixed columns (keys and non-keys)
            _blend_suffixed_columns_inplace(acc_df, suffix_tokens)
            left_suffix = f"_{stem}"
        merged_groups.append((frozenset(key_subset), acc_df, left_suffix))

    # Greedy pairwise merges across groups until disjoint (prefer larger overlaps)
    while True:
        best_pair = None
        best_score = None  # (-overlap_size, combined_cells)
        n = len(merged_groups)

        for i in range(n):
            keys_i, df_i, sfx_i = merged_groups[i]
            for j in range(i + 1, n):
                keys_j, df_j, sfx_j = merged_groups[j]
                shared = keys_i & keys_j
                if not shared:
                    continue
                score = (-len(shared), df_i.size + df_j.size)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (i, j, sorted(shared))

        if best_pair is None:
            break

        i, j, join_cols = best_pair
        keys_i, df_i, sfx_i = merged_groups[i]
        keys_j, df_j, sfx_j = merged_groups[j]

        merged_df = df_i.merge(df_j, on=join_cols, how=join, suffixes=(sfx_i, sfx_j))
        # Blend any identical suffixed columns (keys and non-keys)
        _blend_suffixed_columns_inplace(merged_df, suffix_tokens)

        merged_groups[i] = (keys_i | keys_j, merged_df, sfx_j)
        del merged_groups[j]

    # Assemble output
    output = tuple(
        (tuple(sorted(keys)), df) for (keys, df, _) in sorted(merged_groups, key=lambda g: tuple(sorted(g[0])))
    )
    return output[0] if len(output) == 1 else output


if __name__ == "__main__":
    import argparse

    # Example script (assuming folder mimic-iv-3.1 is in the current directory)
    # python -m mmai25_hackathon.io.tabular mimic-iv-3.1 --index-cols subject_id hadm_id charttime --join outer
    # NOTE: Expect increase in row count given we are doing outer join and each dataframes may or will have
    #       different relational structures (i.e., admissions to icustays in MIMIC-IV has one-to-many
    #       relationship w.r.t. subject_id and hadm_id)

    parser = argparse.ArgumentParser(description="Read and aggregate tabular CSV files.")
    parser.add_argument("base_path", help="Base path for the CSV files.")
    parser.add_argument("--index-cols", nargs="+", default=None, help="Columns to use as index.")
    parser.add_argument("--join", default="outer", help="Join type for merging DataFrames.")
    args = parser.parse_args()

    # Recursive glob for CSV files
    csv_files = list(Path(args.base_path).rglob("*.csv"))
    stems = [f.stem for f in csv_files]

    print(f"Merging {len(csv_files)} CSV files including:")
    for f in csv_files:
        print(f" - {f}")

    merged_indices, df = read_tabular(csv_files, index_cols=args.index_cols, join=args.join)

    print(df)
