## Pull Request Template

### Summary
Provide a short summary of the changes in this PR.

---

### Dataset & DataLoader Design
- Describe the **design decisions** for your Dataset and DataLoader implementation.  
- Explain how your design addresses **scalability, flexibility, and modularity**.  
- Highlight any specific challenges and how you resolved them.  

---

### Handling Relationships Between Samples
- Explain how your DataLoader takes **relationships across samples** into account. For example: linking samples from different modalities that come from the **same subject**.  
- Describe how missing modalities or unaligned samples are handled.  

---

### Sample Usage Instructions
Provide example code showing how to use your Dataset/DataLoader:

```python
from load_data.dataset import MyDataset
from torch.utils.data import DataLoader

dataset = MyDataset(root="data/", ...)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

for batch in loader:
    print(batch.keys())
```

### Data Modalities Covered

List the modalities included in this PR (tick all that apply):

- [ ] Image
- [ ] Text
- [ ] Signals
- [ ] Tabular
- [ ] Other

### Extensibility

Explain how this implementation can be extended to support additional modalities.

- What should be modified or subclassed?
- What conventions should be followed?

### Checklist

- [ ] Code runs without errors
- [ ] Code is well-documented
- [ ] Tests are included and pass
- [ ] Follows project coding standards

### Additional Notes

Any additional information or context that reviewers should be aware of.