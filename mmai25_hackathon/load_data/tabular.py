"""
Tabular data utilities for reading, merging, and graph conversion.

Functions:
    - read_tabular: Load a single CSV file into a DataFrame.
    - merge_multiple_dataframes: Merge DataFrames by join keys, handling column collisions with suffixes.
    - tabular_to_graph: Convert a DataFrame to a graph using row/sample similarity.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params


@validate_params(
    {
        "path": [Path, str],
        "subset_cols": [None, list, str],
        "index_cols": [None, list, str],
        "filter_rows": [None, dict],
        "sep": [str],
        "raise_errors": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def read_tabular(
    path: Union[str, Path],
    subset_cols: Optional[Union[List[str], str]] = None,
    index_cols: Optional[Union[List[str], str]] = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
    sep: str = ",",
    raise_errors: bool = True,
) -> pd.DataFrame:
    """
    Reads a single tabular text file into a DataFrame.

    If `subset_cols` and/or `index_cols` are provided, will select only those columns
    that exist in the DataFrame. The order will be `index_cols` followed by `subset_cols`.

    Args:
        path (Union[str, Path]): Path to the tabular text file.
        subset_cols (Optional[Union[List[str], str]]): If provided, will select only these columns if they exist in the DataFrame.
            Default: None.
        index_cols (Optional[Union[List[str], str]]): If provided, will select these columns as the index of the DataFrame.
            Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): If provided, will filter the rows
            based on the specified column values. The keys are column names and the values are sequences
            of acceptable values for filtering. Will be ignored if not found in the DataFrame. Default: None.
        sep (str): Value separator in the tabular text file. Default: ",".
        raise_errors (bool): If True, will raise an error if none of the specified `subset_cols`
            or `index_cols` are found in the DataFrame. Default: True.

    Returns:
        pd.DataFrame: The loaded DataFrame with complete or partial columns selected.

    Raises:
        ValueError: If `raise_errors` is True and none of the specified `subset_cols`
            or `index_cols` are found in the DataFrame.

    Examples:
        >>> df = read_tabular("data.csv")
        >>> print(df.head())
    """
    # Just provide thin wrapper around pd.read_csv for function availability.
    df = pd.read_csv(path, sep=sep)

    if isinstance(subset_cols, str):
        subset_cols = [subset_cols]

    if isinstance(index_cols, str):
        index_cols = [index_cols]

    selected_index_cols = pd.Index([])
    if index_cols is not None:
        # preserves the order of index_cols as provided when sort=False
        selected_index_cols = df.columns.intersection(index_cols, sort=False)

    selected_subset_cols = df.columns.difference(selected_index_cols, sort=False)
    if subset_cols is not None:
        selected_subset_cols = df.columns.intersection(subset_cols, sort=False)

    if raise_errors and subset_cols is not None and len(selected_subset_cols) == 0:
        raise ValueError(f"No valid subset_cols found in DataFrame for: {subset_cols}")

    if raise_errors and index_cols is not None and len(selected_index_cols) == 0:
        raise ValueError(f"No valid index_cols found in DataFrame for: {index_cols}")

    # Reorder the dataframe to have index_cols first then subset_cols
    selected_cols = selected_index_cols.union(selected_subset_cols, sort=False)

    df = df if len(selected_cols) == 0 else df[df.columns.intersection(selected_cols, sort=False)]

    for col, valid_vals in (filter_rows or {}).items():
        if col in df.columns:
            df = df[df[col].isin(valid_vals)]

    return df


@validate_params(
    {
        "dfs": ["array-like"],
        "dfs_name": [None, "array-like"],
        "index_cols": [None, list, str],
        "join": [StrOptions({"outer", "inner", "left", "right"})],
    },
    prefer_skip_nested_validation=True,
)
def merge_multiple_dataframes(
    dfs: Sequence[pd.DataFrame],
    dfs_name: Optional[Sequence[str]] = None,
    index_cols: Optional[List[str]] = None,
    join: Literal["outer", "inner", "left", "right"] = "outer",
) -> List[Tuple[Tuple[str, ...], pd.DataFrame]]:
    """
    Merge a sequence of DataFrames by shared keys until disjoint components remain.

    - If `index_cols` is None/empty, return a single component with all frames concatenated
      column-wise: [((), concat_df)].
    - Otherwise: (1) merge frames that share the same subset of `index_cols`, then
      (2) greedily merge groups whose key sets overlap (prefer larger overlaps, then
      smaller combined size). Column collisions get suffixes from `dfs_name`
      (or `_df{i}` if not provided).

    Args:
        dfs (Sequence[pd.DataFrame]): Sequences of dataframes to merge.
        dfs_name (Optional[Sequence[str]]): Optional names (same length as `dfs`) used to derive merge suffixes.
        index_cols (Optional[List[str]]): Key columns to use for merging. If None/empty,
            will concatenate all frames. Default: None.
        subset_cols (Optional[List[str]]): If provided, will pre select these columns from each dataframe
            if any of them exist in the dataframe. Default: None.
        join (Literal["outer", "inner", "left", "right"]): Dataframe merging strategy. Default: "outer".

    Returns:
        List[Tuple[Tuple[str, ...], pd.DataFrame]]: A list of tuples where each tuple contains:
            - A tuple of key column names used for merging that component.
            - The merged DataFrame for that component.
            - Accounts for column collisions by adding suffixes and disjoining index columns.

    Examples:
        >>> df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        >>> df2 = pd.DataFrame({"id": [1, 2], "b": [0.1, 0.2]})
        >>> df3 = pd.DataFrame({"site": ["A", "B"], "c": [5, 6]})
        >>> comps = merge_multiple_dataframes(
        ...     [df1, df2, df3],
        ...     dfs_name=["X", "Y", "Z"],
        ...     index_cols=["id", "site"],
        ... )
        >>> [keys for keys, _ in comps]
        [('id',), ('site',)]
        >>> # The first component merges df1 & df2 on 'id'; non-key collisions would get
        >>> # suffixes '_X' and '_Y'. The second component is just df3 keyed by 'site'.
    """
    valid_joins = {"outer", "inner", "left", "right"}
    if join not in valid_joins:
        raise ValueError(f"`join` must be one of {valid_joins}, got {join!r}")

    if dfs_name is not None and len(dfs_name) != len(dfs):
        raise ValueError(
            f"Length of `dfs_name` must match length of `dfs`. Found {len(dfs_name)} and {len(dfs)} respectively."
        )

    if not dfs:
        return []

    # Concatenate-only mode
    if not index_cols:
        return [((), pd.concat(list(dfs), axis="columns", join=join))]

    # Prepare suffix labels
    labels = [f"_{name}" for name in (dfs_name or [f"df{i}" for i in range(len(dfs))])]

    # Bucket frames by the exact subset of keys they actually contain
    df_by_subset = {}
    for df, label in zip(dfs, labels):
        subset = tuple(col for col in index_cols if col in df.columns)
        if subset:
            df_by_subset.setdefault(subset, []).append((df, label))

    if not df_by_subset:
        return []

    # Merge within each exact key-subset first
    groups = []  # (keys, df, last_suffix)
    for subset, items in df_by_subset.items():
        (merged_df, left_suffix), *rest = items
        for df, right_suffix in rest:
            merged_df = merged_df.merge(
                df,
                on=list(subset),
                how=join,
                suffixes=(left_suffix, right_suffix),
            )
            left_suffix = right_suffix
        groups.append((frozenset(subset), merged_df, left_suffix))

    # Greedy pairwise merging across groups until no overlaps remain
    while True:
        best = None
        best_score = None  # (-overlap_size, combined_cells)
        n = len(groups)

        for i in range(n):
            keys_i, df_i, sfx_i = groups[i]
            for j in range(i + 1, n):
                keys_j, df_j, sfx_j = groups[j]
                overlap = keys_i & keys_j
                if not overlap:
                    continue
                score = (-len(overlap), df_i.size + df_j.size)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (i, j, sorted(overlap))

        if best is None:
            break

        i, j, on_cols = best
        keys_i, df_i, sfx_i = groups[i]
        keys_j, df_j, sfx_j = groups[j]
        merged_df = df_i.merge(df_j, on=on_cols, how=join, suffixes=(sfx_i, sfx_j))
        groups[i] = (keys_i | keys_j, merged_df, sfx_j)
        del groups[j]

    # Materialize as (sorted_keys_tuple, DataFrame), sorted by keys for consistency
    return [(tuple(sorted(keys)), df) for (keys, df, _) in sorted(groups, key=lambda g: tuple(sorted(g[0])))]


if __name__ == "__main__":
    # separate function for loading tabular data
    # 1. read_tabular (only single csv)
    # 2. merge multiple dataframes (optimize the query greedily)
    # 3. concat multiple dataframes

    import argparse

    # Example script (assuming folder mimic-iv-3.1 is in the current directory)
    # python -m mmai25_hackathon.load_data.tabular mimic-iv-3.1 --index-cols subject_id hadm_id charttime --join outer
    # NOTE: Expect increase in row count given we are doing outer join and each dataframes may or will have
    #       different relational structures (i.e., admissions to icustays in MIMIC-IV has one-to-many
    #       relationship w.r.t. subject_id and hadm_id)

    parser = argparse.ArgumentParser(description="Read and aggregate tabular CSV files.")
    parser.add_argument("base_path", help="Base path for the CSV files.")
    parser.add_argument("--index-cols", nargs="+", default=None, help="Columns to use as index.")
    parser.add_argument("--subset-cols", nargs="+", default=None, help="Columns to subset.")
    parser.add_argument("--join", default="outer", help="Join type for merging DataFrames.")
    args = parser.parse_args()

    # Recursive glob for CSV files
    csv_files = list(Path(args.base_path).rglob("*.csv"))

    # Load multiple dataframes
    dfs = [read_tabular(f, index_cols=args.index_cols, subset_cols=args.subset_cols) for f in csv_files]
    df_names = [f.stem for f in csv_files]

    # Merge dataframes
    components = merge_multiple_dataframes(dfs, dfs_name=df_names, index_cols=args.index_cols, join=args.join)

    for keys, comp_df in components:
        print(f"Component keys: {keys}")
        print(comp_df.head())
        print()
