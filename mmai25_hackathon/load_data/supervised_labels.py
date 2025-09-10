"""
Labels handling utilities for supervised learning.

This module provides functions to fetch supervision labels from CSV files or pandas DataFrames, supporting both single-column and multi-column labels for regression or classification tasks. It also includes utilities for one-hot encoding categorical labels.

Functions:
    - fetch_supervised_labels_from_dataframe: Fetch labels from a DataFrame or CSV file, supporting index columns and single/multi-column labels.
    - one_hot_encode_labels: One-hot encode categorical labels in a DataFrame, supporting single or multiple columns.

Examples:
    >>> df = pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]})
    >>> labels = fetch_supervised_labels_from_dataframe(df, label_col="label", index_col="id")
    >>> one_hot_labels = one_hot_encode_labels(labels)
"""

from typing import Sequence, Union

import numpy as np
import pandas as pd


def fetch_supervised_labels_from_dataframe(
    df: Union[pd.DataFrame, str],
    label_col: Union[str, Sequence[str]],
    index_col: str = None,
) -> pd.DataFrame:
    """
    Fetches supervision labels from a DataFrame or CSV file. Will read the CSV if a path is provided.

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        label_col (Union[str, Sequence[str]]): Column name or sequence of column names for labels.
        index_col (str, optional): Column to set as index. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the labels with name `"label"` if a single column is provided,
                      or the original column names if multiple columns are provided.

    Examples:
        >>> df = pd.DataFrame(
        ...     {"id": [1, 2, 3], "feature1": [0.5, 0.6, 0.7], "feature2": [1.5, 1.6, 1.7], "label": [0, 1, 0]}
        ... )
        >>> labels = fetch_supervised_labels_from_dataframe(df, label_col="label", index_col="id")
        >>> print(labels)
            label
        id
        1      0
        2      1
        3      0
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


def one_hot_encode_labels(labels: pd.DataFrame, columns: Union[Sequence[str], str] = "label") -> pd.DataFrame:
    """
    One-hot encodes categorical labels in a DataFrame.

    Args:
        labels (pd.DataFrame): DataFrame containing the labels to be one-hot encoded.
        columns (Union[Sequence[str], str]): Column name or sequence of column names to be one-hot encoded. Default: "label".

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded labels.

    Examples:
        >>> df = pd.DataFrame({"label": ["cat", "dog", "cat", "mouse"]})
        >>> one_hot_labels = one_hot_encode_labels(df)
        >>> print(one_hot_labels)
           label_cat  label_dog  label_mouse
        0          1          0            0
        1          0          1            0
        2          1          0            0
        3          0          0            1
    """
    if isinstance(columns, str):
        columns = [columns]
    return pd.get_dummies(labels, columns=columns, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    # Example script: python -m mmai25_hackathon.load_data.supervised_labels dataset.csv

    parser = argparse.ArgumentParser(description="Process supervision labels for regression/classification.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing supervision labels.")
    args = parser.parse_args()

    # Take from Peizhen's csv file for DrugBAN training
    df = fetch_supervised_labels_from_dataframe(args.csv_path, label_col="Y")
    for i, label in enumerate(df["label"].head(5), 1):
        print(i, label)

    one_hot_df = one_hot_encode_labels(df, columns="label")
    print(one_hot_df.head(5))
