"""
Train the main (Siamese text-pair) classifier directly on our own
tournament-judge labels, instead of a PSNR/CLIP-derived proxy score.

classify.py's load_data() derives its training label by taking the argmax
of a psnr/clip-based score per sample. We already have a directly-observed
winning (t_start, t_end) per sample from the 12k SDXL-Turbo tournament run
(tournament_12k_final.csv), so this reuses everything else in classify.py
(OrdinalPairClassifier, split_data, the CE training loop) but swaps in a
load_data() that reads the tournament CSV + mapping_file.json directly,
skipping the metrics-CSV/scoring-function path entirely.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path("/data/home/zarageddes/research/ChordEdit")
sys.path.insert(0, str(REPO_ROOT))

import models.classification.classify as classify

TOURNAMENT_CSV = Path("/shared/ssd_30T/zarageddes/tournament_12k_final.csv")
MAPPING_JSON = Path("/shared/ssd_30T/zarageddes/tournament_12k_sdxlturbo/mapping_file.json")
OUTPUTS_DIR = REPO_ROOT / "models" / "classification" / "outputs" / "sdxlturbo_tournament_12k"


def load_data() -> pd.DataFrame:
    tournament = pd.read_csv(TOURNAMENT_CSV, dtype={"sample_id": str})
    tournament = tournament[tournament["final_t_start"] != "nan"].copy()
    tournament["t_start"] = tournament["final_t_start"].astype(float)
    tournament["t_end"] = tournament["final_t_end"].astype(float)

    mapping = json.loads(MAPPING_JSON.read_text())
    strings = pd.DataFrame([
        {"id": sid, "source_prompt": item["original_prompt"], "target_prompt": item["editing_prompt"]}
        for sid, item in mapping.items()
    ])

    df = pd.merge(
        tournament[["sample_id", "t_start", "t_end"]], strings,
        left_on="sample_id", right_on="id",
    )

    t_start_levels = sorted(df["t_start"].unique())
    t_end_levels = sorted(df["t_end"].unique())
    df["t_start_idx"] = df["t_start"].map({v: i for i, v in enumerate(t_start_levels)})
    df["t_end_idx"] = df["t_end"].map({v: i for i, v in enumerate(t_end_levels)})
    return df


if __name__ == "__main__":
    classify.load_data = load_data
    classify.OUTPUTS_DIR = OUTPUTS_DIR
    classify.train()
