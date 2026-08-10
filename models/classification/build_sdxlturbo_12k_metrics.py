"""
Build a full-grid PSNR/CLIP metrics CSV for the 12k tournament dataset by
relabeling Salehi's already-computed per-source-dataset metrics files with
our merged tournament_12k sample_id scheme (region_/background_/style_
prefix + orig_sid), matching the exact set of samples in
tournament_12k_sdxlturbo/mapping_file.json.

No new PSNR/CLIP compute needed -- these metrics were already produced when
the SDXL-Turbo cells themselves were generated.

Outputs, into models/classification/data/:
  id_to_metrics_sdxlturbo_tournament12k.csv
    (sample_id, t_start, t_end, t_delta, psnr, clip_edited)
  id_to_string_pair_sdxlturbo_tournament12k.csv
    (id, source_prompt, target_prompt)
"""
import csv
import json
from pathlib import Path

MAPPING_JSON = Path("/shared/ssd_30T/zarageddes/tournament_12k_sdxlturbo/mapping_file.json")

SOURCE_METRICS = {
    "region": "/shared/ssd_30T/salehi/id_to_metrics_ultraeditregion10000_sdxlturbo_clipedit_psnrunedit_original.csv",
    "background": "/shared/ssd_30T/salehi/id_to_metrics_ultraeditbackground1000v2_sdxlturbo_clipedit_psnrunedit_original.csv",
    "style": "/shared/ssd_30T/salehi/id_to_metrics_ultraeditstyle1000v2_sdxlturbo_clipedit_psnrunedit_original.csv",
}

OUT_DIR = Path(__file__).resolve().parent / "data"
METRICS_OUT = OUT_DIR / "id_to_metrics_sdxlturbo_tournament12k.csv"
STRINGS_OUT = OUT_DIR / "id_to_string_pair_sdxlturbo_tournament12k.csv"


def main():
    mapping = json.loads(MAPPING_JSON.read_text())
    print(f"{len(mapping)} samples in tournament_12k mapping_file.json", flush=True)

    # orig_sid -> new_id, per dataset prefix.
    lookup = {}
    for new_id, item in mapping.items():
        lookup.setdefault(item["dataset"], {})[item["orig_sid"]] = new_id

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out = open(METRICS_OUT, "w", newline="")
    writer = csv.writer(metrics_out)
    writer.writerow(["sample_id", "t_start", "t_end", "t_delta", "psnr", "clip_edited"])

    total_written = 0
    for prefix, path in SOURCE_METRICS.items():
        id_map = lookup.get(prefix, {})
        written = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                new_id = id_map.get(row["sample_id"])
                if new_id is None:
                    continue
                psnr = row["psnr_unedit_part"] or "nan"
                clip = row["clip_similarity_target_image_edit_part"] or "nan"
                writer.writerow([new_id, row["t_start"], row["t_end"], "0.0", psnr, clip])
                written += 1
        print(f"{prefix}: wrote {written} rows ({written // 121} samples)", flush=True)
        total_written += written
    metrics_out.close()
    print(f"total: {total_written} rows -> {METRICS_OUT}", flush=True)

    with open(STRINGS_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source_prompt", "target_prompt"])
        for new_id, item in mapping.items():
            writer.writerow([new_id, item["original_prompt"], item["editing_prompt"]])
    print(f"{len(mapping)} rows -> {STRINGS_OUT}", flush=True)


if __name__ == "__main__":
    main()
