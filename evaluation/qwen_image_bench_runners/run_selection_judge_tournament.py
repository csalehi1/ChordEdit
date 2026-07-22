"""
Runs QwenSelectionJudge's 2-round tournament with the Qwen-Image-Bench (27B)
checkpoint over the UltraEdit 100-sample v2 consolidated data-root, splitting
samples across one or more replicas via --shard/--nshards (each replica
should be launched with CUDA_VISIBLE_DEVICES set to its own pair of GPUs --
see run_full_metrics_judge.py for why).

Expects to run on a host with access to /shared/ssd_30T/zarageddes/ (this
was built/tested on the "seribizon" research box).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from evaluation.selection_judge import QwenSelectionJudge

DATA_ROOT = Path("/shared/ssd_30T/zarageddes/ultraedit_100_v2_dataroot")
MAPPING_PATH = DATA_ROOT / "mapping_file.json"

T_VALUES = [round(0.1 * i, 1) for i in range(11)]
BATCH_SIZE = 4  # smaller than the 8B-model's 8 -- 27B sharded across 2 GPUs has less headroom per call

COLUMNS = [
    "sample_id", "editing_instruction",
    "final_t_start", "final_t_end", "final_reasoning",
    "round1_winners_json",
]


def cell_filename(ts, te):
    def fmt(v):
        return f"{v:.1f}".replace(".", "p")
    return f"t_start_{fmt(ts)}__t_end_{fmt(te)}.jpg"


def load_cell(sample_id, ts, te):
    path = DATA_ROOT / sample_id / "cells" / cell_filename(ts, te)
    if not path.exists():
        return None
    with Image.open(path) as img:
        return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen-Image-Bench")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    mapping = json.loads(MAPPING_PATH.read_text())
    all_ids = sorted(mapping.keys())
    if args.max_samples is not None:
        all_ids = all_ids[:args.max_samples]
    shard_selected = [sid for i, sid in enumerate(all_ids) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(shard_selected)} samples", flush=True)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                done.add(row["sample_id"])
        print(f"resuming: {len(done)} samples already done", flush=True)

    write_header = not out_path.exists()
    fout = open(out_path, "a", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(COLUMNS)

    todo = [sid for sid in shard_selected if sid not in done]
    print(f"shard {args.shard}: {len(todo)} samples to run ({len(done)} already done)", flush=True)

    sids, samples = [], []
    for sid in todo:
        item = mapping[sid]
        instruction = item["editing_instruction"]

        groups = []
        for ts in T_VALUES:
            cells = [((ts, te), load_cell(sid, ts, te)) for te in T_VALUES]
            cells = [(key, img) for key, img in cells if img is not None]
            if cells:
                groups.append(cells)
        if not groups:
            continue

        sids.append(sid)
        samples.append({
            "instruction": instruction,
            "src_image": Image.open(DATA_ROOT / item["image_path"]).convert("RGB"),
            "groups": groups,
        })

    def on_progress(round_name, done_count, total):
        print(f"shard {args.shard} [{round_name} {done_count}/{total}]", flush=True)

    judge = QwenSelectionJudge(
        model_id=args.model_id,
        max_new_tokens=4096, repetition_penalty=1.05, enable_thinking=True,
    )
    print(f"Model device map: {judge.model.hf_device_map}", flush=True)
    results = judge.judge_tournament_batch(samples, batch_size=BATCH_SIZE, on_progress=on_progress)

    for sid, sample, result in zip(sids, samples, results):
        if result["best_key"] is None:
            final_ts, final_te, final_reasoning = "nan", "nan", ""
        else:
            final_ts, final_te = result["best_key"]
            final_reasoning = result["reasoning"]
        winners = [(ts, te, reasoning) for (ts, te), reasoning in result["round1_winners"]]
        writer.writerow([
            sid, sample["instruction"],
            final_ts, final_te, final_reasoning,
            json.dumps(winners),
        ])
    fout.flush()
    fout.close()
    print("done ->", out_path, flush=True)


if __name__ == "__main__":
    main()
