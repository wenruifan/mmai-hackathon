# Quick Demo: Molecule Utilities

This demo shows how to use the molecule utilities to read SMILES strings from a DataFrame and convert them to graph representations using PyTorch Geometric.

## 1. Read SMILES from DataFrame

```python
import pandas as pd
from mmai25_hackathon.io.molecule import fetch_smiles_from_dataframe, smiles_to_graph

data = {
    "id": [1, 2, 3],
    "smiles": ["CCO", "C1=CC=CC=C1", "CC(=O)O"],
}
df = pd.DataFrame(data)
smiles_df = fetch_smiles_from_dataframe(df, smiles_col="smiles", index_col="id")
print(smiles_df)
```

### Output

```
   smiles
id
1      CCO
2  C1=CC=CC=C1
3   CC(=O)O
```

## 2. Convert SMILES to Graph

```python
for smiles in smiles_df["smiles"]:
    graph = smiles_to_graph(smiles)
    print(graph)
```

### Output

```
Data(x=[3, 9], edge_index=[2, 4], edge_attr=[4, 3], smiles='CCO')
Data(x=[6, 9], edge_index=[2, 12], edge_attr=[12, 3], smiles='C1=CC=CC=C1')
Data(x=[4, 9], edge_index=[2, 6], edge_attr=[6, 3], smiles='CC(=O)O')
```
