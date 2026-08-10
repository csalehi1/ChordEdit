"""
For each trained classifier, run its predictions on its own test split and
compare the REAL psnr_unedit_part/clip_similarity at the classifier's picked
(t_start, t_end) cell against the fixed ChordEdit-paper default baseline
(0.75, 0.3) -- rounded to the nearest grid point, (0.7, 0.3), since our grid
is discretized in 0.1 steps. Ground truth is the same real metrics CSV for
every classifier (id_to_metrics_sdxlturbo_tournament12k.csv), regardless of
what each classifier was trained to predict, so results are directly
comparable across classifiers.
"""
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path("/data/home/zarageddes/research/ChordEdit")
sys.path.insert(0, str(REPO_ROOT))

from models.classification.model import OrdinalPairClassifier
from models.classification.model_with_image import OrdinalPairClassifierWithImage

METRICS_CSV = REPO_ROOT / "models" / "classification" / "data" / "id_to_metrics_sdxlturbo_tournament12k.csv"
IMAGE_EMB_PATH = REPO_ROOT / "models" / "classification" / "data" / "sdxlturbo_12k_clip_image_embeddings.pt"
BASELINE_T_START, BASELINE_T_END = 0.7, 0.3

RUNS = [
    {
        "label": "Tournament-label (text-only)",
        "run_dir": REPO_ROOT / "models/classification/outputs/sdxlturbo_tournament_12k/20260802_205032",
        "kind": "text",
    },
    {
        "label": "Tournament-label + CLIP image",
        "run_dir": REPO_ROOT / "models/classification/outputs/sdxlturbo_tournament_12k_with_image/20260802_213236",
        "kind": "image",
    },
    {
        "label": "LINEX-score alpha=1 (text-only)",
        "run_dir": REPO_ROOT / "models/classification/outputs/sdxlturbo_tournament12k/20260802_213048",
        "kind": "text",
    },
    {
        "label": "LINEX-score alpha=5 (text-only)",
        "run_dir": REPO_ROOT / "models/classification/outputs/sdxlturbo_tournament12k/20260802_222649",
        "kind": "text",
    },
]


def load_metrics_lookup():
    df = pd.read_csv(METRICS_CSV, dtype={"sample_id": str})
    return {(r.sample_id, round(r.t_start, 1), round(r.t_end, 1)): (r.psnr, r.clip_edited) for r in df.itertuples()}


def predict_text(run_dir, test_df, device):
    ckpt = torch.load(run_dir / "classifier_weights.pt", map_location=device, weights_only=False)
    buckets1 = torch.tensor(ckpt["buckets1"])
    buckets2 = torch.tensor(ckpt["buckets2"])
    model = OrdinalPairClassifier(buckets1=buckets1, buckets2=buckets2, freeze_encoder=True, head_type="CE").to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    srcs = test_df["source_prompt"].tolist()
    tgts = test_df["target_prompt"].tolist()
    t_start, t_end = [], []
    BS = 32
    for i in range(0, len(srcs), BS):
        ts, te = model.predict(srcs[i:i + BS], tgts[i:i + BS])
        t_start.extend(ts.tolist())
        t_end.extend(te.tolist())
    return t_start, t_end


def predict_image(run_dir, test_df, device):
    image_embs = torch.load(IMAGE_EMB_PATH)
    ckpt = torch.load(run_dir / "classifier_weights.pt", map_location=device, weights_only=False)
    buckets1 = torch.tensor(ckpt["buckets1"])
    buckets2 = torch.tensor(ckpt["buckets2"])
    image_dim = next(iter(image_embs.values())).shape[0]
    model = OrdinalPairClassifierWithImage(
        image_dim=image_dim, buckets1=buckets1, buckets2=buckets2, freeze_encoder=True, head_type="CE",
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    srcs = test_df["source_prompt"].tolist()
    tgts = test_df["target_prompt"].tolist()
    imgs = torch.stack([image_embs[sid] for sid in test_df["sample_id"]]).to(device)
    t_start, t_end = [], []
    BS = 32
    with torch.no_grad():
        for i in range(0, len(srcs), BS):
            out1, out2 = model(srcs[i:i + BS], tgts[i:i + BS], imgs[i:i + BS])
            idx1, idx2 = model.decode_bucket_indices(out1, out2, batch_size=len(srcs[i:i + BS]))
            t_start.extend(model.buckets1[idx1].tolist())
            t_end.extend(model.buckets2[idx2].tolist())
    return t_start, t_end


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookup = load_metrics_lookup()
    print(f"loaded {len(lookup)} ground-truth cells", flush=True)

    for run in RUNS:
        test_df = pd.read_parquet(run["run_dir"] / "test.parquet.gz")
        if run["kind"] == "text":
            t_start, t_end = predict_text(run["run_dir"], test_df, device)
        else:
            t_start, t_end = predict_image(run["run_dir"], test_df, device)

        rows = []
        for sid, ts, te in zip(test_df["sample_id"], t_start, t_end):
            ts, te = round(ts, 1), round(te, 1)
            pred_metrics = lookup.get((sid, ts, te))
            base_metrics = lookup.get((sid, BASELINE_T_START, BASELINE_T_END))
            if pred_metrics is None or base_metrics is None:
                continue
            rows.append({
                "sample_id": sid, "pred_t_start": ts, "pred_t_end": te,
                "pred_psnr": pred_metrics[0], "pred_clip": pred_metrics[1],
                "base_psnr": base_metrics[0], "base_clip": base_metrics[1],
            })
        result_df = pd.DataFrame(rows)

        n = len(result_df)
        psnr_win = (result_df["pred_psnr"] > result_df["base_psnr"]).mean()
        clip_win = (result_df["pred_clip"] > result_df["base_clip"]).mean()
        both_win = ((result_df["pred_psnr"] > result_df["base_psnr"]) & (result_df["pred_clip"] > result_df["base_clip"])).mean()
        print(f"\n=== {run['label']} ({n} test samples) ===")
        print(f"  mean PSNR-unedited:  baseline={result_df['base_psnr'].mean():.3f}   classifier-pick={result_df['pred_psnr'].mean():.3f}")
        print(f"  mean CLIP-edited:    baseline={result_df['base_clip'].mean():.3f}   classifier-pick={result_df['pred_clip'].mean():.3f}")
        print(f"  win rate:  PSNR better={psnr_win:.1%}   CLIP better={clip_win:.1%}   both better={both_win:.1%}")

        out_csv = run["run_dir"] / "baseline_vs_pick_comparison.csv"
        result_df.to_csv(out_csv, index=False)
        print(f"  saved -> {out_csv}")


if __name__ == "__main__":
    main()
