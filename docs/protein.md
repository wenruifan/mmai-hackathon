# Quick Demo: Protein Utilities

This demo shows how to use the protein utilities to read protein sequences from a DataFrame and convert them to integer-encoded representations.

## 1. Read Protein Sequences from DataFrame

```python
import pandas as pd
from mmai25_hackathon.io.protein import fetch_protein_sequences_from_dataframe, protein_sequence_to_integer_encoding

data = {
    "id": [1, 2, 3],
    "protein_sequence": ["MKTAYIAKQRQISFVKSH", "GAVLILLLV", "TTPSYVAFTDTER"],
}
df = pd.DataFrame(data)
prot_df = fetch_protein_sequences_from_dataframe(df, prot_seq_col="protein_sequence", index_col="id")
print(prot_df)
```

### Output

```
      protein_sequence
id
1  MKTAYIAKQRQISFVKSH
2         GAVLILLLV
3      TTPSYVAFTDTER
```

## 2. Convert Protein Sequence to Integer Encoding

```python
for seq in prot_df["protein_sequence"]:
    encoded = protein_sequence_to_integer_encoding(seq, max_length=10)
    print(encoded)
```

### Output

```
[13 11 20  1 25  9  1 11 17 18]
[7 1 22 12 9 12 12 12 22  0]
[20 20 16 19 25 22  1  6 20  4]
```
