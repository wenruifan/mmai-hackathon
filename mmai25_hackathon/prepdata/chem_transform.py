import torch
from rdkit import Chem

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def smiles_to_graph(smiles: str, max_drug_nodes: int) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object representing the molecular graph.

    Args:
        smiles (str): The SMILES representation of the molecule.
        max_drug_nodes (int): The maximum number of nodes (atoms) in the graph.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object containing:
            - x (torch.Tensor): Node feature matrix of shape [num_nodes, num_node_features].
            - edge_index (torch.Tensor): Graph connectivity in COO format of shape [2, num_edges].
            - edge_attr (torch.Tensor): Edge feature matrix of shape [num_edges, num_edge_features].
            - num_nodes (int): The number of nodes in the graph.
    """
    molecule = Chem.MolFromSmiles(smiles)

    # Feature extraction per-atom/node
    atom_features = [
        [
            atom.GetAtomicNum(),  # Atomic number - essential
            atom.GetDegree(),
            atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
        ]
        for atom in molecule.GetAtoms()
    ]
    atom_features = torch.tensor(atom_features, dtype=torch.float)

    # Feature extraction per-bond/edge
    edge_indices = []
    edge_features = []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_indices += [[start, end], [end, start]]
        edge_features += [[bond_type] * 2]

    # Cast to tensors
    num_nodes = len(atom_features)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # Add self-loops
    edge_indices, edge_features = add_self_loops(edge_indices, edge_features, fill_value=0, num_nodes=num_nodes)

    # Ensure edge_features is a 2D tensor
    if edge_features.dim() == 1:
        edge_features = edge_features.unsqueeze(-1)

    # Pad or truncate by max_drug_nodes
    if num_nodes < max_drug_nodes:
        padding = torch.zeros(max_drug_nodes - num_nodes, atom_features.size(1), dtype=torch.float)
        atom_features = torch.cat((padding, atom_features), 0)
    elif num_nodes > max_drug_nodes:
        atom_features = atom_features[:max_drug_nodes]

    assert atom_features.dim() == 2, f"x must be a 2D tensor, got {atom_features.dim()}D."
    assert edge_indices.dim() == 2, f"edge_index must be a 2D tensor, got {edge_indices.dim()}D."
    assert edge_indices.size(0) == 2, f"edge_index must have shape [2, num_edges], got {edge_indices.size()}."
    assert edge_features.dim() == 2, f"edge_attr must be a 2D tensor, got {edge_features.dim()}D."

    return Data(x=atom_features, edge_index=edge_indices, edge_attr=edge_features, num_nodes=max_drug_nodes)
