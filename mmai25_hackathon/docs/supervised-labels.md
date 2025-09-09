# Quick Demo: Supervised Labels Utilities

This demo shows how to use the supervised labels utilities to fetch labels from a DataFrame and perform one-hot encoding for classification tasks.

## 1. Fetch Labels from DataFrame

```python
import pandas as pd
from mmai25_hackathon.io.supervised_labels import fetch_supervised_labels_from_dataframe, one_hot_encode_labels

data = {
    "id": [1, 2, 3, 4],
    "label": ["cat", "dog", "cat", "mouse"],
}
df = pd.DataFrame(data)
labels_df = fetch_supervised_labels_from_dataframe(df, label_col="label", index_col="id")
print(labels_df)
```

### Output

```
   label
id
1   cat
2   dog
3   cat
4 mouse
```

## 2. One-Hot Encode Labels

```python
one_hot_df = one_hot_encode_labels(labels_df, columns="label")
print(one_hot_df)
```

### Output

```
   label_cat  label_dog  label_mouse
id
1        1.0        0.0         0.0
2        0.0        1.0         0.0
3        1.0        0.0         0.0
4        0.0        0.0         1.0
```
