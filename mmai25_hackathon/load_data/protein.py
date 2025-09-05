"""
Provides utility functions for handling protein sequences, including reading from
dataframes or CSV files and converting sequences to integer-encoded representations.
"""

from typing import Union

import numpy as np
import pandas as pd

# Generate character set for protein sequences between A-Z (except J)
CHARPROTSET = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) != "J"]
# Zero is used for padding or unknown characters hence increment from one.
CHARPROTSET = {letter: idx for idx, letter in enumerate(CHARPROTSET, 1)}


def fetch_protein_sequences_from_dataframe(
    df: Union[pd.DataFrame, str], prot_seq_col: str, index_col: str = None
) -> pd.DataFrame:
    """
    Fetches protein sequences from a DataFrame or CSV file. Will read the CSV if a path is provided.

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        prot_seq_col (str): Column name for protein sequences.
        index_col (str, optional): Column to set as index. Default: None.

    Returns:
        pd.DataFrame: A single column DataFrame containing the protein sequences with name `"protein_sequence"`.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if prot_seq_col not in df.columns:
        raise ValueError(f"Column '{prot_seq_col}' not found in DataFrame.")

    if index_col is not None:
        df = df.set_index(index_col)

    return df[prot_seq_col].to_frame("protein_sequence")


def protein_sequence_to_integer_encoding(sequence: str, max_length: int = 1200) -> np.ndarray:
    """
    Converts a protein sequence into an integer-encoded representation.

    Args:
        sequence (str): The protein sequence to encode.
        max_length (int): The maximum length of the output array.

    Returns:
        np.ndarray: An array of shape (max_length,) containing the integer-encoded representation.
    """
    # Initialize an array of zeros
    encoded_sequence = np.zeros(max_length, dtype=np.uint64)
    for i, char in enumerate(sequence[:max_length]):
        # If character is not in CHARPROTSET, it will be skipped and assumed to be unknown
        encoded_sequence[i] = CHARPROTSET.get(char, 0)
    return encoded_sequence
