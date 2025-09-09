# CXR Data I/O Module (`cxr.py`)

This module provides functions to load chest X-ray (CXR) data from the MIMIC-CXR-JPG v2.1.0 dataset.
For this hackathon, we have used a subset of the dataset while preserving the original folder structure and file organisation.
---

## Data Description

This dataset contains:

- A set of 3 folders, each with 3 sub-folders, where each sub-folder corresponds to a single patient's radiographic studies in JPG format.
- `mimic-cxr-2.0.0-metadata.csv.gz`: metadata including view position, patient orientation, and anonymized image acquisition time.
- `mimic-cxr-2.0.0-split.csv.gz`: recommended train/validation/test split.
- `mimic-cxr-2.0.0-chexpert.csv.gz`: CheXpert-generated labels.
- `mimic-cxr-2.0.0-negbio.csv.gz`: NegBio-generated labels.
- `mimic-cxr-2.1.0-test-set-labeled.csv`: manually curated labels used for evaluation of CheXpert and NegBio.
- `IMAGE_FILENAMES`: plain text file with relative paths to all images.
- Linked to all the MIMIC Clinical Databases via `subject_id`.

---

## Folder Structure

Images are stored in patient-specific folders, organized as:

```
files/
├── p100/
│   └── p101/
│       ├── s133/
│       │   ├── 2d58f126-bf35-4fb7-8ae7-7e9d3ea0172e.jpg
│       │   └── 03c72b1f-062f-4ed4-93b1-53968755bee5.jpg
│       ├── s231/
│       ├── s378/
```

Each patient (e.g., `p101`) may have multiple **study folders** (e.g., `s133`, `s231`, etc.), each containing one or more CXR images in `.jpg` format.

---

## Functional Overview

This module supports:
- Mapping `dicom_id` to image paths.
- Loading images as grayscale or RGB using PIL.
---

## Quick Example

### Step 1: Set the data path

Update this in `cxr.py`:

```python
DATA_PATH = r"your_data_path_here"
```

Example:

```python
DATA_PATH = r"D:\Datasets\MIMIC-CXR-JPG"
```

---

### Step 2: Load image paths using metadata

```python
from cxr import get_cxr_paths, load_cxr_image, FILES_PATH

# Automatically discover the metadata CSV
df = get_cxr_paths(FILES_PATH)

# Or specify metadata file directly
# csv_file = r"D:\Datasets\MIMIC-CXR-JPG\mimic-cxr-2.0.0-metadata.csv"
# df = get_cxr_paths(FILES_PATH, csv_file)
```

---

### Step 3: Load and display a single image

```python
if not df.empty:
    img = load_cxr_image(df.iloc[0]["path"])  # Grayscale PIL image
    img.show()  # Optional: preview the image
```

---

## 📂 Module Location

This module is located at:

```
mmai25_hackathon/load_data/cxr.py
```

---