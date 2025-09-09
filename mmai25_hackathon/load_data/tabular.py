"""
Tabular data utilities for reading, merging, and graph conversion.

Functions:
    - read_tabular: Load a single CSV file into a DataFrame.
    - merge_multiple_dataframes: Merge DataFrames by join keys, handling column collisions with suffixes.
    - tabular_to_graph: Convert a DataFrame to a graph using row/sample similarity.
"""

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_kernels
from torch_geometric.data import Data

from ..utils import find_global_cutoff, symmetrize_matrix


def read_tabular(path: Union[str, Path], sep: str = ",") -> pd.DataFrame:
    """
    Reads a single tabular text file into a DataFrame.

    Args:
        path (Union[str, Path]): Path to the tabular text file.
        sep (str): Value separator in the tabular text file. Default: ",".

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Examples:
        >>> df = read_tabular("data.csv")
        >>> print(df.head())
    """
    # Just provide thin wrapper around pd.read_csv for function availability.
    return pd.read_csv(path, sep=sep)


def merge_multiple_dataframes(
    dfs: Sequence[pd.DataFrame],
    dfs_name: Optional[Sequence[str]] = None,
    index_cols: Optional[List[str]] = None,
    join: str = "outer",
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
        dfs: Input DataFrames.
        dfs_name: Optional names (same length as `dfs`) used to derive merge suffixes.
        index_cols: Candidate join keys.
        join: One of {"outer", "inner", "left", "right"}.

    Returns:
        List of components as (sorted_keys_tuple, merged_dataframe).

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
    frames_by_subset: Dict[Tuple[str, ...], List[Tuple[pd.DataFrame, str]]] = {}
    for df, label in zip(dfs, labels):
        subset = tuple(col for col in index_cols if col in df.columns)
        if subset:
            frames_by_subset.setdefault(subset, []).append((df, label))

    if not frames_by_subset:
        return []

    # Merge within each exact key-subset first
    groups: List[Tuple[FrozenSet[str], pd.DataFrame, str]] = []  # (keys, df, last_suffix)
    for subset, items in frames_by_subset.items():
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

    # Materialize as (sorted_keys_tuple, DataFrame)
    components: List[Tuple[Tuple[str, ...], pd.DataFrame]] = [
        (tuple(sorted(keys)), df) for (keys, df, _sfx) in sorted(groups, key=lambda g: tuple(sorted(g[0])))
    ]
    return components


def tabular_to_graph(
    df: pd.DataFrame, edge_per_node: int = 10, metric: str = "cosine", threshold: Optional[float] = None
) -> Data:
    """
    Convert tabular data from DataFrame into a graph representation. The nodes are derived from the rows
    or samples in the DataFrame, and edges are created based on similarity between the rows.

    Args:
        df (pd.DataFrame): DataFrame containing the features. Assumes the subsets (i.e., train/test/val)
            separation is done separately and `df` contains all samples.
        edge_per_node (int): Number of edges to create per node based on similarity. Default: 10.
        metric (str): Metric to compute similarity between rows. Default: "cosine".
        threshold (Optional[float]): Predefined thresholds for edge creation. If None, it will be estimated
            from the data given `edge_per_node`. Default: None.

    Returns:
        torch_geometric.data.Data: The resulting graph representation containing:
            - `x`: Node feature matrix with shape [num_nodes, num_node_features].
            - `edge_index`: Graph connectivity in COO format with shape [2, num_edges].
            - `edge_attr`: Edge feature matrix with shape [num_edges, num_edge_features].
            - `num_nodes`: Number of nodes in the graph.
            - `num_edges`: Number of edges in the graph.
            - `feature_names`: List of feature names corresponding to columns in `df`.
            - `metric`: The metric used for similarity computation.
            - `threshold`: The threshold used for edge creation.
    """
    # Assumes all columns can be casted to float32
    features = df.to_numpy(dtype=np.float32)
    similarity_matrix = pairwise_kernels(features, metric=metric)
    if threshold is None:
        threshold = find_global_cutoff(similarity_matrix, edge_per_node)

    # We do not need to remove self-loops as they will be removed in GNN layers if needed
    adjacency_matrix = symmetrize_matrix(similarity_matrix >= threshold, method="maximum")
    # Get edges where we have shape [2, num_edges]
    edge_index = np.vstack(np.nonzero(adjacency_matrix)).astype(np.int64)
    # We use the sparsified similarity value as the edge weights
    edge_weight = similarity_matrix[adjacency_matrix]

    # Cast to torch tensors
    x = torch.from_numpy(features)
    edge_index = torch.from_numpy(edge_index)
    edge_weight = torch.from_numpy(edge_weight)

    return Data(
        x,
        edge_index,
        edge_weight=edge_weight,
        feature_names=df.columns.to_list(),
        metric=metric,
        threshold=threshold,
    )


if __name__ == "__main__":
    # separate function for loading tabular data
    # 1. read_tabular (only single csv)
    # 2. merge multiple dataframes (optimize the query greedily)
    # 3. concat multiple dataframes

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

    # Load multiple dataframes
    dfs = [pd.read_csv(f) for f in csv_files]
    df_names = [f.stem for f in csv_files]

    # Merge dataframes
    components = merge_multiple_dataframes(dfs, dfs_name=df_names, index_cols=args.index_cols, join=args.join)

    for keys, comp_df in components:
        print(f"Component keys: {keys}")
        print(comp_df.head())
        print()

    # Quick test for tabular_to_graph
    from sklearn.datasets import make_classification

    # Create a synthetic dataset
    X, _ = make_classification(n_samples=100, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    graph = tabular_to_graph(df, edge_per_node=5, metric="cosine")
    print(graph)
