"""
Concatenate per-shard CSVs into one merged output file.

Shards may or may not actually have a header row -- a stale/resumed run can
write data rows without ever (re-)writing the header, since the header is
only written when the output file doesn't already exist. Detect this
per-shard by checking whether the first cell of the shard's first line is
literally "sample_id" (the header's column name) rather than an actual
sample_id value, instead of assuming line 1 is always a header -- otherwise
a headerless shard's first data row gets misread as a header and either
falsely flagged as a mismatch against a real header, or silently dropped.
"""
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
            lines = list(csv.reader(f))
        if not lines:
            continue
        first = lines[0]
        if first and first[0] == "sample_id":
            if header is None:
                header = first
            elif first != header:
                raise ValueError(f"{shard_path} header mismatch: {first} != {header}")
            rows.extend(lines[1:])
        else:
            rows.extend(lines)

    if header is None:
        raise ValueError("no shard had a header row -- cannot determine output columns")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"merged {len(args.shards)} shards -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
