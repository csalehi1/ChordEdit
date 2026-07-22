"""Concatenate per-shard CSVs (same header) into one merged output file."""
import argparse
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    header = None
    rows = []
    for shard_path in args.shards:
        with open(shard_path) as f:
            reader = csv.reader(f)
            shard_header = next(reader)
            if header is None:
                header = shard_header
            elif shard_header != header:
                raise ValueError(f"{shard_path} header mismatch: {shard_header} != {header}")
            rows.extend(reader)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"merged {len(args.shards)} shards -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
