# Echo Data I/O Module (`echo.py`)

This module provides utilities to load echocardiograms (ECHO) from the **MIMIC-IV-ECHO** dataset. It supports resolving `.dcm` DICOM file paths using `echo-record-list.csv` and reading the DICOMs using `pydicom`. For this hackathon, we have used a subset of the dataset while preserving the original folder structure and file organisation.

---

## Data Description

### Echocardiograms

- 18 echocardiographic DICOM files from 18 distinct patients.
- Linked to all the MIMIC Clinical Databases via `subject_id`.
- Stored as `.dcm` DICOM files, each containing a cine sequence (multi-frame view).

### Directory Structure

Each study is stored in its own subdirectory and follows this pattern:

```
files/
├── p100/
│   └── p101/
│       ├── s133/
│       │   └── 133.dcm
│       └── s231/
│           └── 231.dcm
└── p200/
    └── p201/
        └── s247/
            └── 247.dcm
```
---

## Supporting CSVs

- `echo-record-list.csv`: maps `dicom_filepath` to the actual `.dcm` file, includes `subject_id`, `study_id`, and `acquisition_datetime`.
- `echo-study-list.csv`: links `study_id` to corresponding cardiologist reports (`note_id`, `note_seq`, `note_charttime`).

---
## Functional Overview

This module supports:
- Resolving DICOM paths from `echo-record-list.csv`.
- Loading `.dcm` cine sequences using `pydicom`.
- Extracting key metadata such as number of frames, resolution, frame time, etc.

---

## Quick Example

### Step 1: Set the data path

Update this in `echo.py`:

```python
DATA_PATH = r"your_data_path_here"
```

Example:

```python
DATA_PATH = r"D:\Datasets\MIMIC-IV-ECHO"
```

---

### Step 2: Load DICOM paths

```python
from echo import get_echo_paths, load_echo_dicom, FILES_PATH

csv_file = os.path.join(FILES_PATH, "echo-record-list.csv")
df = get_echo_paths(FILES_PATH, csv_file)
```

---

### Step 3: Load a DICOM and extract metadata

```python
if not df.empty:
    frames, meta = load_echo_dicom(df.iloc[0]["dcm_path"])
    print("Frames shape:", frames.shape)
    print("Number of Frames:", meta["NumberOfFrames"])
    print("Resolution: {}x{}".format(meta["Rows"], meta["Columns"]))
```

---

## 📂 Module Location

This module is located at:

```
mmai25_hackathon/load_data/echo.py
```

---