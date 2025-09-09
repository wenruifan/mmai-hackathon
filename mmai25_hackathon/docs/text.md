# Clinical Text Notes I/O Module (`text.py`)

This module provides utilities to load **free-text clinical notes** from the **MIMIC-IV Note (v2.2)** dataset. It supports loading both radiology and discharge notes, with optional metadata and detail tables. For this hackathon, we have used a subset of the dataset while preserving the original folder structure and file organisation.

---

## Data Description

There are four tables in the dataset:

- `radiology.csv`
- `radiology_detail.csv`
- `discharge.csv`
- `discharge_detail.csv`

Each table contains a `note_id` composed of `subject_id`, note domain abbreviation, and a sequential integer.

### Radiology Detail

- `radiology.csv`: contains free-text radiology reports (e.g., X-ray, CT, MRI, ultrasound).
- `radiology_detail.csv`: includes CPT codes, exam names, and metadata (e.g., addendums, links to parent reports).

### Discharge Detail

- `discharge.csv`: long-form discharge summaries including **reason for admission**, **hospital course**, and **discharge plan**.
- `discharge_detail.csv`: contains author placeholders and additional structured metadata.

---

## Directory Structure

Expected note CSVs are found under:

```
mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/
├── discharge.csv
├── discharge_detail.csv
├── radiology.csv
└── radiology_detail.csv
```

---

## Functional Overview

This module supports:
- Loading any of the four note tables into `pandas` DataFrames.
- Subset selection of specific columns (e.g., `note_id`, `text`, `charttime`, etc.).
- Retrieving the full clinical note from the `text` column using a specific note_id.

---

## Quick Example

### Step 1: Set the data path

Update this in `text.py`:

```python
DATA_PATH = r"your_data_path_here"
```

Example:

```python
DATA_PATH = r"D:\Datasets\MIMIC-IV-NOTE"
```

---

### Step 2: Load notes

```python
from text import get_text_notes, load_text_note, NOTE_PATH

# Load radiology notes
radi_df = get_text_notes(NOTE_PATH, subset="radiology", include_detail=False)

# Load discharge notes with detail
disc_df = get_text_notes(NOTE_PATH, subset="discharge", include_detail=True)
```

---

### Step 3: Retrieve a specific note

```python
sample_text = load_text_note(radi_df, note_id=int(radi_df.iloc[0]["note_id"]))
print(sample_text[:300])  # Truncated print
```

You can also retrieve metadata:

```python
text, meta = load_text_note(disc_df, note_id=int(disc_df.iloc[0]["note_id"]), return_meta=True)
print("Note Text:", text[:300])
print("Metadata:", meta)
```

---

## 📂 Module Location

This module is located at:

```
mmai25_hackathon/load_data/text.py
```

---