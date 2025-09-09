"""
Molecular data utilities for handling SMILES strings and graph conversion.

Functions:
    - fetch_smiles_from_dataframe: Extract SMILES strings from DataFrame or CSV.
    - smiles_to_graph: Convert SMILES to a molecular graph (uses PyG's from_smiles).

Note:
    - The function smiles_to_graph is a wrapper for PyG's native from_smiles implementation, provided for clarity in this hackathon context.
"""

from typing import Union

import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils.smiles import from_smiles


def fetch_smiles_from_dataframe(df: Union[pd.DataFrame, str], smiles_col: str, index_col: str = None) -> pd.DataFrame:
    """
    Fetches SMILES strings from a DataFrame or CSV file. Will read the CSV if a path is provided.

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        smiles_col (str): Column name for SMILES representations.
        index_col (str, optional): Column to set as index. Default: None.

    Returns:
        pd.DataFrame: A single column DataFrame containing the SMILES strings with name `"smiles"`.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 2, 3],
        ...         "smiles": ["CCO", "C1=CC=CC=C1", "CC(=O)O"],
        ...     }
        ... )
        >>> smiles = fetch_smiles_from_dataframe(df, smiles_col="smiles", index_col="id")
        >>> print(smiles)
            smiles
        id
        1      CCO
        2  C1=CC=CC=C1
        3   CC(=O)O
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame.")

    if index_col is not None:
        df = df.set_index(index_col)

    return df[smiles_col].to_frame("smiles")


def smiles_to_graph(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> Data:
    """
    Converts a SMILES string to a molecular graph representation.

    Args:
        smiles (str): The SMILES string to convert.
        with_hydrogen (bool): Store hydrogens in the graph if True. Default: False
        kekulize (bool): Converts aromatic bonds to single/double bonds if True. Default: False

    Returns:
        torch_geometric.data.Data: The molecular graph representation containing:
            - `x`: Node feature matrix with shape [num_nodes, num_node_features].
            - `edge_index`: Graph connectivity in COO format with shape [2, num_edges].
            - `edge_attr`: Edge feature matrix with shape [num_edges, num_edge_features].
            - `num_nodes`: Number of nodes in the graph.
            - `num_edges`: Number of edges in the graph.
            - `smiles`: The original SMILES string as a graph attribute.

    Examples:
        >>> smiles = "CCO"
        >>> graph = smiles_to_graph(smiles)
        >>> print(graph)
        Data(x=[3, 9], edge_index=[2, 4], edge_attr=[4, 3], smiles='CCO')
    """
    return from_smiles(smiles, with_hydrogen, kekulize)


if __name__ == "__main__":
    import argparse

    # Example script: python -m mmai25_hackathon.io.molecule dataset.csv

    parser = argparse.ArgumentParser(description="Process SMILES strings.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing SMILES strings.")
    args = parser.parse_args()

    # Take from Peizhen's csv file for DrugBAN training
    df = fetch_smiles_from_dataframe(args.csv_path, smiles_col="SMILES")
    for i, smiles in enumerate(df["smiles"].head(5), 1):
        graph = smiles_to_graph(smiles)
        print(i, graph)
