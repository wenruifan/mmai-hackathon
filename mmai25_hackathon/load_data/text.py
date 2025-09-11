"""
MIMIC-IV clinical notes (free-text) loading utilities.

Functions:
load_mimic_iv_notes(note_path, subset='radiology', include_detail=False, subset_cols=None)
    Loads the selected notes CSV (`radiology.csv` or `discharge.csv`), verifies required ID columns
    (`note_id`, `subject_id`), optionally merges `<subset>_detail.csv` when `include_detail=True`,
    applies optional `subset_cols`, strips/filters empty `text`, and returns a `pd.DataFrame` indexed by
    [`note_id`, `subject_id`].

extract_text_from_note(note, include_metadata=False)
    Extracts the `text` field from a single note `pd.Series`. When `include_metadata=True`, returns
    `(text, metadata_dict)` where `metadata_dict` is the note’s fields excluding `text`.

Preview CLI:
`python -m mmai25_hackathon.load_data.text /path/to/mimic-iv-note-.../note radiology 12345678`
Prints a preview of the loaded notes (columns like `note_id`, `subject_id`, `hadm_id`, `note_type`, `text`) and then
retrieves the note matching the provided `note_id`, printing its full text and selected metadata (e.g., `subject_id`,
`hadm_id`, `note_type`).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from .tabular import merge_multiple_dataframes, read_tabular

REQUIRED_ID_COLS = ["note_id", "subject_id"]


@validate_params(
    {
        "note_path": [str, Path],
        "subset": [StrOptions({"radiology", "discharge"})],
        "include_detail": ["boolean"],
        "subset_cols": [None, list],
    },
    prefer_skip_nested_validation=True,
)
def load_mimic_iv_notes(
    note_path: Union[str, Path],
    subset: Literal["radiology", "discharge"] = "radiology",
    include_detail: bool = False,
    subset_cols: Optional[List[str]] = [
        "hadm_id",
        "note_type",
        "note_seq",
        "charttime",
        "storetime",
        "text",
    ],
) -> pd.DataFrame:
    if isinstance(note_path, str):
        note_path = Path(note_path)

    if not note_path.exists():
        raise FileNotFoundError(f"Notes folder not found: {note_path}")

    subset_path = note_path / f"{subset}.csv"

    if not subset_path.exists():
        raise FileNotFoundError(f"Missing main notes CSV: {subset_path}")

    logger = logging.getLogger(f"{__name__}.load_mimic_iv_notes")
    logger.info("Loading notes from: %s", subset_path)
    df_notes = read_tabular(subset_path, subset_cols=subset_cols, index_cols=REQUIRED_ID_COLS)
    logger.info("Loaded %d notes from: %s", len(df_notes), subset_path)

    id_cols_available = df_notes.columns.intersection(REQUIRED_ID_COLS).to_list()
    if len(id_cols_available) < len(REQUIRED_ID_COLS):
        raise KeyError(f"{subset_path} must contain columns: {REQUIRED_ID_COLS}. Found: {id_cols_available}")

    detail_path = note_path / f"{subset}_detail.csv"
    if include_detail and not detail_path.exists():
        raise FileNotFoundError(f"Missing detail notes CSV: {detail_path}")

    if include_detail:
        logger.info("Including detail from: %s", detail_path)
        df_detail = read_tabular(detail_path, index_cols=REQUIRED_ID_COLS)
        logger.info("Loaded %d detail rows from: %s", len(df_detail), detail_path)
        id_cols_available = df_detail.columns.intersection(REQUIRED_ID_COLS).to_list()
        if len(id_cols_available) < len(REQUIRED_ID_COLS):
            raise KeyError(f"{detail_path} must contain columns: {REQUIRED_ID_COLS}. Found: {id_cols_available}")
        logger.info("Merging detail into main notes on: %s", REQUIRED_ID_COLS)
        df_notes = merge_multiple_dataframes((df_notes, df_detail), ("notes", "detail"), REQUIRED_ID_COLS, "left")
        # Unpack df_notes from list of (paired_keys, df_notes)
        df_notes = df_notes[0]
        _, df_notes = df_notes

    text_included = "text" in df_notes.columns
    if text_included:
        df_notes["text"] = df_notes["text"].astype(str).str.strip()
        available_texts = df_notes["text"] != ""
        df_notes = df_notes[available_texts].copy()
        logger.info("After filtering, %d notes have non-empty text.", len(df_notes))
    else:
        logger.warning("The loaded notes do not include a 'text' column.")

    return df_notes


@validate_params({"note": [pd.Series], "include_metadata": ["boolean"]}, prefer_skip_nested_validation=True)
def extract_text_from_note(note: pd.Series, include_metadata: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Extracts the text from a note Series, optionally returning metadata.

    Args:
        note (pd.Series): A pandas Series representing a note, expected to contain a 'text' column.
        include_metadata (bool): If True, return a tuple of (text, metadata_dict). Default is False.

    Returns:
        Union[str, Tuple[str, Dict[str, Any]]]: The note text; optionally with the note's metadata as a dictionary.

    Raises:
        KeyError: If the 'text' column is not present in the note Series.

    Examples:
        >>> note = pd.Series(
        ...     {"note_id": 1, "subject_id": 101, "text": "Patient is stable.", "note_type": "Discharge summary"}
        ... )
        >>> extract_text_from_note(note)
        'Patient is stable.'
        >>> extract_text_from_note(note, include_metadata=True)
        ('Patient is stable.', {'note_id': 1, 'subject_id': 101, 'note_type': 'Discharge summary'})
    """
    if "text" not in note:
        raise KeyError("The note does not include a 'text' column.")

    logger = logging.getLogger(f"{__name__}.extract_text_from_note")
    logger.info("Extracting text from note with ID: %s", note.get("note_id", "unknown"))

    text = note["text"]
    if not include_metadata:
        return text

    logger.info("Including metadata in the output.")

    metadata = note.drop("text").to_dict()
    return text, metadata


if __name__ == "__main__":
    import argparse

    # Example script:
    # python -m mmai25_hackathon.load_data.text mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note radiology 1
    parser = argparse.ArgumentParser(description="Load MIMIC-IV free-text clinical notes.")
    parser.add_argument("data_path", type=str, help="Path to the MIMIC-IV notes directory (containing CSV files).")
    parser.add_argument(
        "subset",
        type=str,
        choices=["radiology", "discharge"],
        help="Which note subset to load (radiology or discharge).",
    )
    parser.add_argument("note_id", type=int, help="The note_id of the note to retrieve.")
    args = parser.parse_args()

    print(f"Loading {args.subset} notes from: {args.data_path}")
    data = load_mimic_iv_notes(args.data_path, subset=args.subset, include_detail=True)
    print(data.head()[["note_id", "subject_id", "hadm_id", "note_type", "text"]])
    print()

    print(f"Retrieving text for note_id={args.note_id}")
    try:
        text, metadata = extract_text_from_note(
            data.loc[data["note_id"] == args.note_id].squeeze(), include_metadata=True
        )
        print("Note text:")
        print(text)
        print("Metadata:")
        print(metadata)
    except KeyError as e:
        print(e)
