"""
Compact utilities for reading CSVs and aggregating tables.

Exposes `read_tabular(paths, index_cols=None, join='outer')`, which either
concatenates inputs column-wise (no keys) or merges by overlapping join keys
until disjoint groups remain, preserving column provenance via filename-based
suffixes. Returns a concatenated DataFrame or a tuple of (keys, DataFrame) pairs
(single pair is unpacked).
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import pandas as pd


def read_tabular(
    paths: Union[List[str], str],
    index_cols: Optional[List[str]] = None,
    join: str = "outer",
) -> Union[
    pd.DataFrame,
    Tuple[Tuple[Tuple[str, ...], pd.DataFrame], ...],
    Tuple[Tuple[str, ...], pd.DataFrame],
    Tuple[()],
]:
    """
    Reads and aggregates tabular CSV files by concatenation or merges.

    If `index_cols` is falsy, all CSVs are concatenated column-wise. Otherwise,
    DataFrames that share one or more of the provided `index_cols` are merged
    greedily until only disjoint key-components remain. Column name collisions
    are disambiguated using suffixes derived from filename stems.

    Args:
        paths (Union[List[str], str]): A list of CSV paths or a single CSV path.
        index_cols (Optional[List[str]]): Candidate join keys. If None/empty,
            inputs are concatenated along columns. Default: None.
        join (str): Merge strategy: one of {"outer", "inner", "left", "right"}.
            Default: "outer".

    Returns:
        Union[
            pd.DataFrame,
            Tuple[Tuple[Tuple[str, ...], pd.DataFrame], ...],
            Tuple[Tuple[str, ...], pd.DataFrame],
            Tuple[()],
        ]:
            - If `index_cols` is falsy: the concatenated DataFrame.
            - Else: a tuple of `(keys_tuple, DataFrame)` pairs, where `keys_tuple`
              is the sorted union of keys in that merged component. If exactly one
              component exists, that single pair is returned directly (unpacked).
              If no CSV contains any of `index_cols`, returns `()`.

    Raises:
        ValueError: If `join` is invalid.

    Examples:
        >>> read_tabular(["a.csv", "b.csv"])
        DataFrame(...)
        >>> read_tabular(["df1.csv", "df2.csv", "df3.csv"], index_cols=["id", "site"])
        ((('id', 'site'), DataFrame), (('subject',), DataFrame), ...)
    """
    valid_joins = {"outer", "inner", "left", "right"}
    if join not in valid_joins:
        raise ValueError(f"`join` must be one of {valid_joins}, got {join!r}")

    # Normalize input list and load CSVs (track stems for suffix provenance)
    if isinstance(paths, str):
        paths = [paths]

    dataframes: List[pd.DataFrame] = [pd.read_csv(path) for path in paths]
    filename_stems: List[str] = [Path(path).stem for path in paths]

    # Concatenate mode (preserve column names)
    if not index_cols:
        return pd.concat(dataframes, axis="columns", join=join)

    # Group each DataFrame by the exact subset of keys it actually contains
    frames_by_exact_keys: Dict[Tuple[str, ...], List[Tuple[pd.DataFrame, str]]] = {}
    for df, stem in zip(dataframes, filename_stems):
        key_subset = tuple(col for col in index_cols if col in df.columns)
        if key_subset:
            frames_by_exact_keys.setdefault(key_subset, []).append((df, stem))

    if not frames_by_exact_keys:
        return tuple()  # no frames contained any join keys

    # Merge within each exact key-subset first (deterministic, cheap)
    # Structure: (keys_frozenset, merged_df, last_suffix_used)
    merged_groups: List[Tuple[FrozenSet[str], pd.DataFrame, str]] = []
    for key_subset, frames_with_stems in frames_by_exact_keys.items():
        (merged_df, first_stem), *rest = frames_with_stems
        left_suffix = f"_{first_stem}"
        for df, stem in rest:
            merged_df = merged_df.merge(df, on=list(key_subset), how=join, suffixes=(left_suffix, f"_{stem}"))
            left_suffix = f"_{stem}"
        merged_groups.append((frozenset(key_subset), merged_df, left_suffix))

    # Greedy pairwise merging across groups until disjoint (prefer larger overlaps)
    while True:
        best_pair = None
        best_pair_score = None  # (-overlap_size, combined_cells)
        group_count = len(merged_groups)

        for i in range(group_count):
            keys_i, df_i, suffix_i = merged_groups[i]
            for j in range(i + 1, group_count):
                keys_j, df_j, suffix_j = merged_groups[j]
                shared_keys = keys_i & keys_j
                if not shared_keys:
                    continue
                score = (-len(shared_keys), df_i.size + df_j.size)
                if best_pair_score is None or score < best_pair_score:
                    best_pair_score = score
                    best_pair = (i, j, sorted(shared_keys))

        if best_pair is None:
            break

        i, j, join_columns = best_pair
        keys_i, df_i, suffix_i = merged_groups[i]
        keys_j, df_j, suffix_j = merged_groups[j]

        merged_df = df_i.merge(df_j, on=join_columns, how=join, suffixes=(suffix_i, suffix_j))
        merged_groups[i] = (keys_i | keys_j, merged_df, suffix_j)
        del merged_groups[j]

    # Assemble output as a tuple of (sorted_keys_tuple, DataFrame) pairs
    output = tuple(
        (tuple(sorted(keys)), df) for (keys, df, _suffix) in sorted(merged_groups, key=lambda g: tuple(sorted(g[0])))
    )
    return output[0] if len(output) == 1 else output
