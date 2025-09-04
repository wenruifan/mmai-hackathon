"""
Provides label handling utilities fetching labels.
For now, we limit it to supporting CSV and DataFrame inputs.
"""

from typing import Union, Sequence
import pandas as pd


def fetch_labels_from_dataframe(
    df: pd.DataFrame, label_col: Union[str, Sequence[str]], index_col: str = None
) -> pd.DataFrame:
    """
    Fetches labels from a DataFrame or CSV file. Will read the CSV if a path is provided.

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        label_col (Union[str, Sequence[str]]): Column name or sequence of column names for labels.
        index_col (str, optional): Column to set as index. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the labels with name `"label"` if a single column is provided,
                      or the original column names if multiple columns are provided.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    if index_col is not None:
        df = df.set_index(index_col)

    if isinstance(label_col, Sequence) and len(label_col) > 1:
        return df[list(label_col)]

    return df[label_col].to_frame("label")
