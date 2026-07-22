"""
Runs QwenFullMetricsJudge with the Qwen-Image-Bench (27B) checkpoint over
the UltraEdit 100-sample v2 consolidated data-root, splitting samples across
one or more replicas via --shard/--nshards (each replica should be launched
with CUDA_VISIBLE_DEVICES set to its own pair of GPUs -- the model doesn't
fit on a single ~49GB GPU, so device_map="auto" shards it across 2 GPUs per
replica; run multiple replicas in parallel for real throughput).

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
from evaluation.full_metrics_judge import QwenFullMetricsJudge

DATA_ROOT = Path("/shared/ssd_30T/zarageddes/ultraedit_100_v2_dataroot")
MAPPING_PATH = DATA_ROOT / "mapping_file.json"

T_VALUES = [round(0.1 * i, 1) for i in range(11)]
BATCH_SIZE = 16  # ~19.5GB headroom per 2-GPU replica at batch=12 suggested room for ~18-20;
                 # kept some margin since thinking-mode KV cache keeps growing during generation.

FACTORS = QwenFullMetricsJudge.FACTORS
CATEGORIES = list(QwenFullMetricsJudge.CATEGORIES.keys())

COLUMNS = ["sample_id", "t_start", "t_end", "editing_instruction"]
for factor in FACTORS:
    COLUMNS += [f"{factor}_score", f"{factor}_justification"]
COLUMNS += [f"{category}_score" for category in CATEGORIES]
COLUMNS += ["overall_score"]


def cell_filename(ts, te):
    def fmt(v):
        return f"{v:.1f}".replace(".", "p")
    return f"t_start_{fmt(ts)}__t_end_{fmt(te)}.jpg"


def cell_path(sample_id, ts, te):
    return DATA_ROOT / sample_id / "cells" / cell_filename(ts, te)


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
                done.add((row["sample_id"], row["t_start"], row["t_end"]))
        print(f"resuming: {len(done)} cells already scored", flush=True)

    write_header = not out_path.exists()
    fout = open(out_path, "a", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(COLUMNS)

    todo = []
    for sid in shard_selected:
        item = mapping[sid]
        instruction = item["editing_instruction"]
        src_path = DATA_ROOT / item["image_path"]

        for ts in T_VALUES:
            for te in T_VALUES:
                key = (sid, f"{ts:.1f}", f"{te:.1f}")
                if key in done:
                    continue
                tgt_path = cell_path(sid, ts, te)
                if not tgt_path.exists():
                    continue
                todo.append((sid, ts, te, instruction, src_path, tgt_path))

    total = len(todo) + len(done)
    print(f"shard {args.shard}: {len(todo)} cells to score ({len(done)} already done, {total} total)", flush=True)

    judge = QwenFullMetricsJudge(
        model_id=args.model_id,
        max_new_tokens=4096, repetition_penalty=1.05, enable_thinking=True,
    )
    print(f"Model device map: {judge.model.hf_device_map}", flush=True)

    done_count = len(done)
    for b in range(0, len(todo), BATCH_SIZE):
        batch = todo[b: b + BATCH_SIZE]
        instructions = [row[3] for row in batch]
        src_images = [Image.open(row[4]).convert("RGB") for row in batch]
        tgt_images = [Image.open(row[5]).convert("RGB") for row in batch]

        results = judge.judge_batch(instructions, src_images, tgt_images)

        for (sid, ts, te, instruction, _src, _tgt), result in zip(batch, results):
            overall = QwenFullMetricsJudge.overall_score(result)
            category_rollups = QwenFullMetricsJudge.category_scores(result)
            row = [sid, ts, te, instruction]
            for factor in FACTORS:
                row += [result[factor]["score"], result[factor]["justification"]]
            row += [category_rollups[category] for category in CATEGORIES]
            row += [overall]
            writer.writerow(row)
        fout.flush()

        done_count += len(batch)
        last_sid, last_ts, last_te, *_ = batch[-1]
        last_overall = QwenFullMetricsJudge.overall_score(results[-1])
        print(
            f"[{done_count}/{total}] batch of {len(batch)} done, last={last_sid} ts={last_ts} te={last_te} -> {last_overall}",
            flush=True,
        )

    fout.close()
    print("done ->", out_path, flush=True)


if __name__ == "__main__":
    main()
