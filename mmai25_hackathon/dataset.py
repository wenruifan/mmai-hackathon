"""
The core of the hackathon is to create a multimodal dataset and dataloader
that can seamlessly handle and scale with multiple modalities such as
proteins, molecules, images, text, tabular, and time series data.

We provided a barebones `torch.utils.data.Dataset` implementation
named `BaseDataset`, which has the base methods mainly used when
building custom datasets.

The class serves as a template for creating custom datasets.

Optionally, we include an `__add__` as potential base idea to aggregate
multiple modalities into a single dataset. Read `__add__` docstring
for more details. You may consider other ways to achieve this that
work better for your use case.

We also provide `torch_geometric.data.DataLoader` as the default DataLoader
instead of `torch.utils.data.DataLoader` which we aliased as `BaseDataLoader`
for handling both graph and non-graph data seamlessly in-case there are some
potential ideas that needs overriding of the DataLoader.
"""

from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as PyGDataLoader


class BaseDataset(Dataset):
    """
    Base dataset class for creating custom datasets.

    The arguments and methods defined here can be customized as needed.

    The goal is to have easy to extend dataset class for various modalities that
    can also be combined to obtain multimodal datasets

    Args:
        *args: Positional arguments for dataset initialization.
        **kwargs: Keyword arguments for dataset initialization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("BaseDataset is an abstract class and cannot be instantiated directly.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__ method.")

    def __getitem__(self, idx: int):
        """Return a single sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__ method.")

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        """Return any extra information about the dataset."""
        return f"sample_size={len(self)}"

    def __add__(self, other):
        """
        Aggregate data with heterogenous modalities.

        For example, we may have:
        ```python
        dataset1 = ECGDataset(...)
        dataset2 = ImageDataset(...)
        ...
        datasetN = TextDataset(...)
        ```
        One way we imagine to combine them is by using the `+` operator,
        such that all we need to do is:
        ```python
        multimodal_dataset = dataset1 + dataset2 + ... + datasetN
        ```
        """
        raise NotImplementedError("Subclasses must implement __add__ method if needed.")


class BaseDataLoader(PyGDataLoader):
    """
    A base dataloader directly inheriting from `torch_geometric.data.DataLoader` without any
    modification. This is to ensure that both graph and non-graph data can be handled seamlessly.

    Args:
    dataset (BaseDataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load. Default: 1
    shuffle (bool, optional): If set to True, the data will be reshuffled at every epoch. Default: False
    follow_batch (List[str], optional): Creates assignment batch vectors for each key in the list. Default: None
    exclude_keys (List[str], optional): Will exclude each key in the list. Default: None
    **kwargs (optional): Additional arguments of torch.utils.data.DataLoader.
    """
