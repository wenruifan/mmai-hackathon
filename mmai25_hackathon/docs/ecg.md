# ECG Data I/O Module (`ecg.py`)

This module provides utilities to load diagnostic electrocardiograms (ECGs) from the **MIMIC-IV-ECG** dataset. It includes path resolution based on `record_list.csv` and loading support via the `wfdb` package. For this hackathon, we have used a subset of the dataset while preserving the original folder structure and file organisation.

---

## Data Description

### Electrocardiogram Waveforms

- 18 ten-second-long 12-lead ECGs across 18 unique subjects.
- ECGs are sampled at **500 Hz**.
- Linked to all the MIMIC Clinical Databases via `subject_id`.

### Directory Structure

The ECGs are grouped by `subject_id` and stored as WFDB records with `.hea` and `.dat` pairs. Folder structure:

```
files/
├── p100/
│   └── p101/
│       └── s133/
│           ├── 133.dat
│           └── 133.hea
├── p200/
│   └── p201/
│       ├── s247/
│       ├── s411/
│       └── s413/
```

Each `study_id` has a `.dat` and `.hea` pair as a WFDB record.

### Supporting CSVs

- `record_list.csv`: contains relative paths, subject IDs, and study IDs for all WFDB records.
- `machine_measurements.csv`: machine-generated measurements for each ECG.
- `machine_measurements_data_dictionary.csv`: description for each measurement column.

---

## Functional Overview

This module supports:
- Resolving full `.hea` and `.dat` paths from `record_list.csv`.
- Loading ECG signals and metadata using `wfdb.rdsamp`.

---

## Quick Example

### Step 1: Set the data path

Edit this in `ecg.py`:

```python
DATA_PATH = r"your_data_path_here"
```

Example:

```python
DATA_PATH = r"D:\Datasets\MIMIC-IV-ECG"
```

---

### Step 2: Load ECG paths

```python
from ecg import get_ecg_paths, load_ecg_record, FILES_PATH

csv_file = os.path.join(FILES_PATH, "record_list.csv")
df = get_ecg_paths(FILES_PATH, csv_file)
```

---

### Step 3: Load and view a single ECG

```python
if not df.empty:
    sig, meta = load_ecg_record(df.iloc[0]["hea_path"])
    print("Signals shape:", sig.shape)
    print("Sampling freq:", meta.get("fs"))
```

---

## 📂 Module Location

This module is located at:

```
mmai25_hackathon/load_data/ecg.py
```