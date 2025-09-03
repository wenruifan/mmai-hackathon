"""
Just a reference code for dataset class.

Can discuss later on how we arrange the base dataset.
"""

import pandas as pd
import torch
from torch.utils import data
from torch_geometric.utils.smiles import from_smiles

from .io.protein import encode_protein_sequence


class DTIDataset(data.Dataset):
    """
    Dataset class for DTI (Drug-Target Interaction) prediction given
    SMILES representations and protein sequences.

    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        smiles_col (str): Column name for SMILES representations.
        prot_seq_col (str): Column name for protein sequences.
        label_col (str): Column name for ground truth labels.
        max_atom_nodes (int): Maximum number of atom nodes in the molecule graph. Default: 290
        max_prot_seq_len (int): Maximum length of the protein sequence. Default: 1200
        with_hydrogen (bool): Whether to include hydrogen atoms in the molecule graph. Default: False
        kekulize (bool): Converts aromatic bonds to single/double bonds if True. Default: False
    """

    def __init__(
        self,
        csv_path: str,
        smiles_col: str,
        prot_seq_col: str,
        label_col: str,
        max_atom_nodes: int = 290,
        max_prot_seq_len: int = 1200,
        with_hydrogen: bool = False,
        kekulize: bool = False,
    ):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.smiles = df[smiles_col].tolist()
        self.protein_sequences = df[prot_seq_col].tolist()
        self.targets = df[label_col].tolist()
        self.max_atom_nodes = max_atom_nodes
        self.max_prot_seq_len = max_prot_seq_len
        self.with_hydrogen = with_hydrogen
        self.kekulize = kekulize

    def __getitem__(self, index):
        """
        Get a single data item from the dataset.

        Args:
            index (int): Index of the data item to retrieve.

        Returns:
            dict: A dictionary of features and labels containing:
                - molecule_graph (torch_geometric.data.Data): The graph representation of the molecule.
                - protein_encoding (torch.Tensor): The encoded representation of the protein sequence.
                - target (int): The ground truth label for the DTI interaction.
        """
        # Convert SMILES to PyG graph representation
        smiles = from_smiles(self.smiles[index])
        # Convert protein sequence to integer encoding and cast to tensor
        protein_encoding = encode_protein_sequence(self.protein_sequences[index], self.max_prot_seq_len)
        protein_encoding = torch.from_numpy(protein_encoding)
        # Fetch target/prediction label
        target = self.targets[index]

        return {"molecule_graph": smiles, "protein_encoding": protein_encoding, "target": target}

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """
        return len(self.smiles)

    def __repr__(self):
        """
        Representation of the DTIDataset.
        """
        return (
            f"DTIDataset(num_samples={len(self)}, "
            f"max_atom_nodes={self.max_atom_nodes}, "
            f"max_prot_seq_len={self.max_prot_seq_len}, "
            f"with_hydrogen={self.with_hydrogen}, "
            f"kekulize={self.kekulize})"
        )
