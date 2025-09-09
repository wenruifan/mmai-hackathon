# Multi-Omics Preprocessing and Graph Construction

We provide a **two-step pipeline** for preparing and loading multi-omics data into graph-structured datasets for graph neural networks (PyTorch Geometric).

- **Step 1: Preprocessing**
  Clean and normalize raw TCGA-style omics data tables (e.g., mRNA, methylation, miRNA) together with clinical labels.

- **Step 2: Graph Loading**
  Convert the processed CSVs into modality-specific graphs, where each node is a patient/sample, and edges represent sample similarity.

---

## 📦 Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [torch-sparse](https://github.com/rusty1s/pytorch_sparse)
- NumPy, Pandas

Install with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric torch-sparse
pip install numpy pandas
```

---

## 🚀 Workflow

### 1. Preprocess raw TCGA-BRCA files

```python
from prep.omics import run_preprocessing_pipeline

label_path = "/home/wenrui/Projects/embc/BRCA/raw/label"
save_path  = "/home/wenrui/Projects/embc/BRCA/processed"

data_paths = [
    ("/home/wenrui/Projects/embc/BRCA/raw/mRNA",        "mRNA"),
    ("/home/wenrui/Projects/embc/BRCA/raw/methylation", "Methylation"),
    ("/home/wenrui/Projects/embc/BRCA/raw/miRNA",       "miRNA"),
]

run_preprocessing_pipeline(
    label_path=label_path,
    data_paths=data_paths,
    save_path=save_path,
    label_column_name="PAM50Call_RNAseq",
    clean_missing=True,
    normalize=True,
    var_threshold=[0.04, 0.03, None],   # one threshold per modality
)
```

This will produce in `save_path`:

- `label.csv` — one column of integer labels (no header/index).
- `mRNA_feat.csv`, `Methylation_feat.csv`, `miRNA_feat.csv` — feature matrices `[samples × features]`.
- `mRNA_names.csv`, `Methylation_names.csv`, `miRNA_names.csv` — lists of feature names.

---

### 2. Load processed CSVs and build graphs

```python
from load_data.omics import load_multiomics

feature_csvs = [
    "/home/wenrui/Projects/embc/BRCA/processed/mRNA_feat.csv",
    "/home/wenrui/Projects/embc/BRCA/processed/Methylation_feat.csv",
    "/home/wenrui/Projects/embc/BRCA/processed/miRNA_feat.csv",
]
labels_csv = "/home/wenrui/Projects/embc/BRCA/processed/label.csv"
feature_names = [
    "/home/wenrui/Projects/embc/BRCA/processed/mRNA_names.csv",
    "/home/wenrui/Projects/embc/BRCA/processed/Methylation_names.csv",
    "/home/wenrui/Projects/embc/BRCA/processed/miRNA_names.csv",
]

train_data = load_multiomics(
    feature_csvs=feature_csvs,
    labels_csv=labels_csv,
    featname_csvs=feature_names,
    mode="train",
    num_classes=5,
    edge_per_node=15,  # average neighbors per node
)

train_data_list = train_data["data_list"]   # one Data per modality
fit_ref         = train_data["fit_"]        # thresholds to reuse on val/test
```

Each `Data` object contains:
- `x`: `[N, F_m]` feature matrix
- `y`: `[N]` labels (or `[N, C]` one-hot if `num_classes` specified)
- `edge_index`, `edge_weight`: graph structure
- `adj_t`: sparse adjacency (`torch_sparse.SparseTensor`)
- `feat_names`: feature name array
- `train_sample_weight`: (train only) sample weights

---

### 3. Apply the same thresholds to validation/test splits

```python
val_data = load_multiomics(
    feature_csvs=val_feature_csvs,
    labels_csv=val_labels_csv,
    featname_csvs=val_feature_names,
    mode="val",
    num_classes=5,
    edge_per_node=15,
    ref=fit_ref,   # reuse thresholds from train
)
```

---

## 📊 Graph construction details

- **Similarity**: cosine similarity between samples.
- **Thresholding**: global cutoff so each node has ~k neighbors.
- **Adjacency**: symmetrized (max), with self-loops, row-normalized.
- **Result**: one patient-similarity graph per modality.

---

## ⚠️ Notes & gotchas

- Raw label files are downloaded from [TCGA website](https://xenabrowser.net/datapages/) expected as **TSVs** (tab-separated). Preprocessing writes CSVs (comma-separated).
- Preprocessing saves `label.csv` **without sample IDs** (to match the loader).
  Modify `save_processed` if you want IDs preserved.
- `_sample_weight` currently uses **class frequency** weights.
  Switch to inverse-frequency if you need proper balancing.
