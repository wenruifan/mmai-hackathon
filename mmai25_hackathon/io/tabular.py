"""
Provides a function to read and aggregate tabular data from various sources.
"""

from typing import List, Optional, Union

import pandas as pd


def read_tabular(
    dfs: Union[List[Union[pd.DataFrame, str]], pd.DataFrame, str],
    index_cols: Optional[List[str]] = None,
    join: str = "outer",
) -> pd.DataFrame:
    """
    Reads and concatenates a sequence of tabular DataFrames.

    Args:
        dfs (Union[List[Union[pd.DataFrame, str]], pd.DataFrame, str]): A DataFrame, a file path to a CSV file,
            or a list of DataFrames/CSV file paths.
        index_cols (Optional[List[str]]): A list of column names to use as the index. Default: None.
        join (str): The type of join to perform when concatenating DataFrames. Default: "outer".

    Returns:
        pd.DataFrame: A single DataFrame containing the concatenated data.
    """
    if isinstance(dfs, (pd.DataFrame, str)):
        dfs = [dfs]

    # Load the csv if not already a DataFrame
    for i, df in enumerate(dfs):
        if isinstance(df, str):
            dfs[i] = pd.read_csv(df)

    # Assumes all dataframes are already aligned by the row count
    # we simply concatenate
    if index_cols is None:
        return pd.concat(dfs, axis="columns")

    # We do a nested memorizing search to aggregate the DataFrames
    df_agg = dfs[0]
    memo_index = set()
    for i in range(1, len(dfs)):
        for j in range(1, len(dfs)):
            # Skip if already merged
            if i in memo_index or j in memo_index:
                continue

            # Find any intersecting columns, skip if none
            intersects = df_agg.columns.intersection(index_cols)
            intersects = intersects.intersection(dfs[j].columns).tolist()
            if not intersects:
                continue

            # Combine by the available intersecting columns
            # Include the index of the current DataFrame if
            # we successfully merged the DataFrames
            df_agg = df_agg.merge(dfs[j], on=intersects, how=join)
            memo_index.add(j)

    return df_agg
