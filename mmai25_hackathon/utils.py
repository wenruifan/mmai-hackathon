"""
General utilities for different use cases.

Functions:
    - find_global_cutoff: Find a cutoff for similarity matrices to ensure k neighbors per row.
    - symmetrize_matrix: Symmetrize a square matrix using various aggregation methods.
"""

import numpy as np

from sklearn.utils.validation import check_symmetric
from sklearn.utils._param_validation import validate_params, Interval, Integral, StrOptions


VALID_SYMMETRIZATIONS = {"average", "maximum", "minimum", "lower", "upper"}


@validate_params(
    {
        "similarity_matrix": ["array-like"],
        "k": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def find_global_cutoff(similarity_matrix: np.ndarray, k: int) -> float:
    """
    Find an approximately correct global cutoff such that each row has at least `k`
    neighbors above the cutoff.

    Args:
        similarity_matrix (np.ndarray): A 2D numpy array representing the similarity matrix.
        k (int): The minimum number of neighbors each row should have above the cutoff.
            If k is outside [1, num_rows - 1], k will be clipped to this range.

    Returns:
        float: The global cutoff value.

    Raises:
        ValueError: If the similarity matrix is not symmetric.

    Examples:
        >>> from sklearn.datasets import make_swiss_roll
        >>> from sklearn.metrics import pairwise_kernels
        >>> X, _ = make_swiss_roll(n_samples=100, noise=0.1, random_state=42)
        >>> similarity_matrix = pairwise_kernels(X, metric="cosine")
        >>> cutoff = find_global_cutoff(similarity_matrix, k=5)
        >>> print(round(cutoff, 4))
        0.9676
    """
    # Validate the similarity matrix is symmetric
    similarity_matrix = check_symmetric(similarity_matrix, raise_exception=True)
    # Get number of samples and clip k to valid range
    num_samples = len(similarity_matrix)

    # No valid edges can be formed
    if num_samples < 2:
        return np.inf

    k = np.clip(k, 1, num_samples - 1)

    # Expect flattened array of shape (num_samples * (num_samples - 1) / 2,)
    # which contains all pairwise similarities without duplicates
    triu_sim_matrix = similarity_matrix[np.triu_indices(num_samples, k=1)]
    finite_sim_matrix = triu_sim_matrix[np.isfinite(triu_sim_matrix)]

    # Fetch targeted number of undirected edges
    # Given initial value (num_samples * k) // 2
    # We will clip it to be at least 1 and at most len(finite_sim_matrix)
    # Subtracting 1 to convert it to a zero-based index
    edge_target = np.clip((num_samples * k) // 2, 1, len(finite_sim_matrix))

    # Get the cutoff value
    index = len(finite_sim_matrix) - edge_target
    return np.partition(finite_sim_matrix, index)[index]


@validate_params(
    {"matrix": ["array-like"], "method": [StrOptions(VALID_SYMMETRIZATIONS)]},
    prefer_skip_nested_validation=True,
)
def symmetrize_matrix(matrix: np.ndarray, method: str = "average") -> np.ndarray:
    """
    Symmetrizes a square matrix using the specified method.

    Args:
        matrix (np.ndarray): A square numpy array to be symmetrized.
        method (str): The method to use for symmetrization. Options are:
            - "sum": matrix + matrix.T
            - "average": (matrix + matrix.T) / 2
            - "maximum": np.maximum(matrix, matrix.T)
            - "minimum": np.minimum(matrix, matrix.T)
            - "lower": np.tril(matrix) + np.tril(matrix, -1).T
            - "upper": np.triu(matrix) + np.triu(matrix, 1).T

    Returns:
        np.ndarray: The symmetrized matrix.

    Raises:
        ValueError: If the input matrix is not square or if an invalid method is provided.

    Examples:
        >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> symmetrize_matrix(mat, method="average")
        array([[1. , 3. , 5. ],
               [3. , 5. , 7. ],
               [5. , 7. , 9. ]])
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    if method == "maximum":
        return np.maximum(matrix, matrix.mT)
    if method == "minimum":
        return np.minimum(matrix, matrix.mT)
    if method == "lower":
        return np.tril(matrix) + np.tril(matrix, -1).mT
    if method == "upper":
        return np.triu(matrix) + np.triu(matrix, 1).mT

    matrix = matrix + matrix.mT

    # default to average
    return 0.5 * matrix if method == "average" else matrix


# Quick run of the function
if __name__ == "__main__":
    # Run with python -m mmai25_hackathon.utils

    from sklearn.datasets import make_swiss_roll
    from sklearn.metrics import pairwise_kernels

    # Create a synthetic dataset
    X, _ = make_swiss_roll(n_samples=100, noise=0.1, random_state=42)

    # Compute the similarity matrix and symmetrize it
    similarity_matrix = pairwise_kernels(X, metric="cosine")
    similarity_matrix = symmetrize_matrix(similarity_matrix)

    # Find the global cutoff
    cutoff = find_global_cutoff(similarity_matrix, k=5)
    print("Global cutoff:", round(cutoff, 4))
