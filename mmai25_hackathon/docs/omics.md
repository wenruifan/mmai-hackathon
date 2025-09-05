# Multiomics data preprocessing and loading

## Preprocessing raw files from TCGA-BRCA database
```python
from prep.omics import run_preprocessing_pipeline

label_path = "/home/wenrui/Projects/embc/BRCA/raw/label"
save_path = "/home/wenrui/Projects/embc/BRCA/processed"
data_paths = [
    ("/home/wenrui/Projects/embc/BRCA/raw/mRNA", "mRNA"),
    ("/home/wenrui/Projects/embc/BRCA/raw/methylation", "Methylation"),
    ("/home/wenrui/Projects/embc/BRCA/raw/miRNA", "miRNA")]

label_column_name = "PAM50Call_RNAseq"

clean_missing = True
normalize = True
var_threshold = [0.04, 0.03, None] 
run_preprocessing_pipeline(
    label_path=label_path,
    data_paths=data_paths, 
    save_path=save_path,
    label_column_name=label_column_name,
    clean_missing=clean_missing,
    normalize=normalize,
    var_threshold=var_threshold
)
```
## Loading the preprocessed CSV files and Convert to the graph we want
```python
from load_data.omics import load_multiomics
feature_csvs = [
    "mRNA_feat.csv",
    "Methylation_feat.csv",
    "miRNA_feat.csv"
]
labels_csv = "label.csv"
feature_names = [
    "mRNA_names.csv",
    "Methylation_names.csv",
    "miRNA_names.csv"
]
train_data = load_multiomics(
    feature_csvs=feature_csvs,
    labels_csv=labels_csv,
    featname_csvs=feature_names,
    mode="train",
    num_classes=5,
    edge_per_node=15,
)
train_data_list = train_data["data_list"]
fit_ref = train_data["fit_"] # loading this reference transform to val and test dataset for consistency

val_data = load_multiomics(
    feature_csvs=val_feature_csvs,
    labels_csv=val_labels_csv,
    featname_csvs=val_feature_names,
    mode="val",
    num_classes=5,
    edge_per_node=15,
    ref=fit_ref, # loading reference transform
)
```