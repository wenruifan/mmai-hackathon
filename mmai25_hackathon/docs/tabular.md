# Quick Demo: Tabular Utilities

This demo shows how to use the tabular utilities to read CSV files, merge DataFrames by join keys, and convert tabular data to a graph.

## 1. Read a Single CSV File

```python
import pandas as pd
from mmai25_hackathon.io.tabular import read_tabular

df = read_tabular("data.csv")
print(df.head())
```

### Output

```
<first few rows of your CSV file>
```

## 2. Merge Multiple DataFrames by Join Keys

```python
from mmai25_hackathon.io.tabular import merge_multiple_dataframes

df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
df2 = pd.DataFrame({"id": [1, 2], "b": [0.1, 0.2]})
df3 = pd.DataFrame({"site": ["A", "B"], "c": [5, 6]})

comps = merge_multiple_dataframes(
		[df1, df2, df3],
		dfs_name=["X", "Y", "Z"],
		index_cols=["id", "site"],
		join="outer"
)
for keys, comp_df in comps:
		print(f"Component keys: {keys}")
		print(comp_df)
```

### Output

```
Component keys: ('id',)
	 id   a    b
0   1  10  0.1
1   2  20  0.2
Component keys: ('site',)
	site  c
0    A  5
1    B  6
```

## 3. Convert Tabular DataFrame to Graph

```python
from mmai25_hackathon.io.tabular import tabular_to_graph
import numpy as np

# Example DataFrame
df = pd.DataFrame(np.random.rand(5, 4), columns=[f"feature_{i}" for i in range(4)])
graph = tabular_to_graph(df, edge_per_node=2, metric="cosine")
print(graph)
```

### Output

```
Data(x=[5, 4], edge_index=[2, 10], edge_attr=[10], feature_names=['feature_0', 'feature_1', 'feature_2', 'feature_3'], metric='cosine', threshold=...)
```
