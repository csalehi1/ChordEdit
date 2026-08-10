"""
Prototype: same tournament-label training target as train_from_tournament.py,
but the model also sees a frozen, precomputed CLIP embedding of the source
image (see precompute_image_embeddings.py), concatenated into the combiner
vector before the shared MLP body (model_with_image.py).

Reuses classify.py's split_data/_class_weights/_balanced_accuracy/_compute_loss
helpers as-is; only the dataset, model construction, and the forward-pass
call signature differ (image_emb is an extra input).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.classification.model_with_image import OrdinalPairClassifierWithImage
from models.classification.model import mae_buckets
import models.classification.classify as classify
from models.classification.settings import (
    BATCH_SIZE, EPOCHS, ENCODER_LR, FREEZE_ENCODER, MLP_LR,
    USE_CLASS_WEIGHTS, WEIGHT_DECAY, SEED,
)

REPO_ROOT = Path("/data/home/zarageddes/research/ChordEdit")
TOURNAMENT_CSV = Path("/shared/ssd_30T/zarageddes/tournament_12k_final.csv")
MAPPING_JSON = Path("/shared/ssd_30T/zarageddes/tournament_12k_sdxlturbo/mapping_file.json")
IMAGE_EMB_PATH = REPO_ROOT / "models" / "classification" / "data" / "sdxlturbo_12k_clip_image_embeddings.pt"
OUTPUTS_DIR = REPO_ROOT / "models" / "classification" / "outputs" / "sdxlturbo_tournament_12k_with_image"


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


class PairWithImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_embs: dict):
        self.src = df["source_prompt"].tolist()
        self.tgt = df["target_prompt"].tolist()
        self.img = torch.stack([image_embs[sid] for sid in df["sample_id"]])
        self.y1 = torch.tensor(df["t_start_idx"].values, dtype=torch.long)
        self.y2 = torch.tensor(df["t_end_idx"].values, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.img[idx], self.y1[idx], self.y2[idx]


def collate(batch):
    srcs, tgts, imgs, y1s, y2s = zip(*batch)
    return list(srcs), list(tgts), torch.stack(imgs), torch.stack(y1s), torch.stack(y2s)


def eval_loader(model, loader, device, *, class_w_start, class_w_end, n_buckets_start, n_buckets_end):
    model.eval()
    all_p1, all_p2, all_l1, all_l2 = [], [], [], []
    val_loss, n_samples = 0.0, 0
    with torch.no_grad():
        for srcs, tgts, imgs, l1, l2 in loader:
            imgs, l1, l2 = imgs.to(device), l1.to(device), l2.to(device)
            out1, out2 = model(srcs, tgts, imgs)
            loss = classify._compute_loss(
                model, out1, out2, l1, l2,
                class_w_start=class_w_start, class_w_end=class_w_end,
                n_buckets_start=n_buckets_start, n_buckets_end=n_buckets_end,
            )
            val_loss += loss.item() * len(srcs)
            n_samples += len(srcs)
            p1, p2 = model.decode_bucket_indices(out1, out2, batch_size=len(srcs))
            all_p1.append(p1); all_p2.append(p2); all_l1.append(l1); all_l2.append(l2)
    p1, p2 = torch.cat(all_p1), torch.cat(all_p2)
    l1o, l2o = torch.cat(all_l1), torch.cat(all_l2)
    return {
        "loss": val_loss / n_samples,
        "mae_t_start": mae_buckets(p1, l1o).item(),
        "mae_t_end": mae_buckets(p2, l2o).item(),
        "acc_t_start": (p1 == l1o).float().mean().item(),
        "acc_t_end": (p2 == l2o).float().mean().item(),
        "acc_both": ((p1 == l1o) & (p2 == l2o)).float().mean().item(),
        "bal_acc_t_start": classify._balanced_accuracy(p1, l1o, n_buckets_start),
        "bal_acc_t_end": classify._balanced_accuracy(p2, l2o, n_buckets_end),
    }


def train():
    df = load_data()
    image_embs = torch.load(IMAGE_EMB_PATH)
    image_dim = next(iter(image_embs.values())).shape[0]
    print(f"image_dim={image_dim}")

    train_df, val_df, test_df = classify.split_data(df, seed=SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    classify.save_splits(train_df, val_df, test_df, data_dir=run_dir)
    print(f"Dataset: {len(df)} samples  split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}")

    train_loader = DataLoader(PairWithImageDataset(train_df, image_embs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairWithImageDataset(val_df, image_embs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairWithImageDataset(test_df, image_embs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    buckets_start = torch.tensor(sorted(df["t_start"].unique()), dtype=torch.float)
    buckets_end = torch.tensor(sorted(df["t_end"].unique()), dtype=torch.float)
    n_buckets_start, n_buckets_end = len(buckets_start), len(buckets_end)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalPairClassifierWithImage(
        image_dim=image_dim, buckets1=buckets_start, buckets2=buckets_end,
        freeze_encoder=FREEZE_ENCODER, head_type="CE",
    ).to(device)

    if FREEZE_ENCODER:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=MLP_LR, weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": list(model.body.parameters()) + list(model.head1.parameters()) + list(model.head2.parameters()),
             "lr": MLP_LR, "weight_decay": WEIGHT_DECAY},
        ])

    if USE_CLASS_WEIGHTS:
        class_w_start = classify._class_weights(torch.tensor(train_df["t_start_idx"].values), n_buckets_start).to(device)
        class_w_end = classify._class_weights(torch.tensor(train_df["t_end_idx"].values), n_buckets_end).to(device)
    else:
        class_w_start = class_w_end = None

    weights_out = run_dir / "classifier_weights.pt"
    best_val_score = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for srcs, tgts, imgs, y1, y2 in train_loader:
            imgs, y1, y2 = imgs.to(device), y1.to(device), y2.to(device)
            l1, l2 = model(srcs, tgts, imgs)
            loss = classify._compute_loss(
                model, l1, l2, y1, y2,
                class_w_start=class_w_start, class_w_end=class_w_end,
                n_buckets_start=n_buckets_start, n_buckets_end=n_buckets_end,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(srcs)

        train_metrics = eval_loader(model, train_loader, device, class_w_start=class_w_start, class_w_end=class_w_end, n_buckets_start=n_buckets_start, n_buckets_end=n_buckets_end)
        train_metrics["loss"] = epoch_loss / len(train_df)
        val_metrics = eval_loader(model, val_loader, device, class_w_start=class_w_start, class_w_end=class_w_end, n_buckets_start=n_buckets_start, n_buckets_end=n_buckets_end)

        val_score = val_metrics["bal_acc_t_start"]
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            torch.save({"state_dict": model.state_dict(), "buckets1": buckets_start.tolist(), "buckets2": buckets_end.tolist()}, weights_out)
        print(
            f"Epoch {epoch:02d}  train: {classify._fmt_split_metrics(train_metrics)}"
            f"  val: {classify._fmt_split_metrics(val_metrics)}" + ("  *" if improved else "")
        )

    print(f"Saved {weights_out}  (best val bal_acc_t_start={best_val_score:.3f})")
    model.load_state_dict(torch.load(weights_out, map_location=device, weights_only=False)["state_dict"], strict=False)
    test_metrics = eval_loader(model, test_loader, device, class_w_start=class_w_start, class_w_end=class_w_end, n_buckets_start=n_buckets_start, n_buckets_end=n_buckets_end)
    print(f"\nTest: {classify._fmt_split_metrics(test_metrics)}")
    return model


if __name__ == "__main__":
    train()
