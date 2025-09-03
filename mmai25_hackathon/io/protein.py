import logging

import numpy as np

# Generate character set for protein sequences
CHARPROTSET = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) != "J"]
# Zero is used for padding or unknown characters
CHARPROTSET = {letter: idx + 1 for idx, letter in enumerate(CHARPROTSET)}


def encode_protein_sequence(sequence: str, max_length: int = 1200) -> np.ndarray:
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
        try:
            encoded_sequence[i] = CHARPROTSET[char]
        except KeyError:
            logging.warning(
                (
                    f"Character '{char}' does not exists in sequence category encoding. "
                    "Will be skipped and treated as padding with index 0."
                )
            )
    return encoded_sequence
