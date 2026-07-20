"""Small shared helpers for classification data loading."""

from __future__ import annotations

import pandas as pd

from models.classification.settings import INPUTS_CSV, SAMPLE_ID_COL


def load_inputs_df() -> pd.DataFrame:
    """Load INPUTS_CSV with a canonical string sample_id for merges."""
    df = pd.read_csv(INPUTS_CSV, dtype={SAMPLE_ID_COL: str})
    df[SAMPLE_ID_COL] = df[SAMPLE_ID_COL].map(lambda x: str(int(x)))
    return df
