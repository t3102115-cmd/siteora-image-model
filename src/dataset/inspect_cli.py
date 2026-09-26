"""CLI used by the dataset notebook: validate a dataset and print stats +
random samples, without modifying anything."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.preprocessing import dataset_statistics
from dataset.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--metadata-file", default="metadata.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    report = validate_dataset(args.dataset_dir, metadata_file=args.metadata_file)
    print(report.summary())
    print()

    if not report.is_valid:
        print("Dataset has issues — see counts above. Fix before training.")
        for issue in (report.missing_files + report.corrupted_files + report.missing_captions)[:20]:
            print(f"  - {issue.file_name}: {issue.reason}")
        sys.exit(1)

    stats = dataset_statistics(args.dataset_dir, metadata_file=args.metadata_file)
    print("Dataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    import json

    with open(Path(args.dataset_dir) / args.metadata_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    sample = random.sample(records, min(args.samples, len(records)))
    print(f"\n{len(sample)} random sample(s):")
    for rec in sample:
        print(f"  {rec['file_name']}: {rec['text']}")


if __name__ == "__main__":
    main()
