"""Build a test.zip that mirrors the train structure, for self-verifying Task1/2/3.

Samples N images stratified by (fruit, style) from the train set, re-indexes them
to 0..N-1, and writes:
    test/images/<id>.jpg
    test/test_labels.csv      (header: file_name,style,fruit)
then zips to test.zip. Upload it to Drive, put its file_id into each notebook's
cell-1 `file_id`, run, and verify the outputs.

Note: with --source all the test images overlap Task1's training data, so Task1
accuracy will look near-perfect (optimistic). Use --source heldout to sample only
from Task1's held-out val split for an honest accuracy estimate.

Usage (run where full train data lives, e.g. RTX):
    python -m src.common.make_test_zip --data data/train --n 1800 --out test.zip
    python -m src.common.make_test_zip --data data/train --source heldout --n 1440
"""

import argparse
import csv
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict

from src.common.dataset import scan
from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--n", type=int, default=1800)
    ap.add_argument("--out", default="test.zip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["all", "heldout"], default="all")
    ap.add_argument("--workdir", default="_test_build")
    args = ap.parse_args()

    samples = [s for s in scan(args.data) if s.fruit is not None and s.style is not None]
    assert samples, f"no labeled samples under {args.data}"
    if args.source == "heldout":
        from src.task1.train import stratified_split
        _, va = stratified_split(samples)            # Task1's unseen 20%
        samples = [samples[i] for i in va]
    print(f"source pool: {len(samples)} images ({args.source})")

    buckets = defaultdict(list)
    for s in samples:
        buckets[(s.fruit, s.style)].append(s)
    per = max(1, args.n // len(buckets))
    rng = random.Random(args.seed)
    chosen = []
    for v in buckets.values():
        rng.shuffle(v)
        chosen += v[:per]
    rng.shuffle(chosen)                              # mix so 0-19.jpg span classes/styles
    chosen = chosen[:args.n]
    print(f"selected: {len(chosen)} images ({per}/cell x {len(buckets)} cells)")

    work = args.workdir
    shutil.rmtree(work, ignore_errors=True)
    img_dir = os.path.join(work, "test", "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for new_id, s in enumerate(chosen):
        shutil.copy(s.path, os.path.join(img_dir, f"{new_id}.jpg"))
        rows.append((f"{new_id}.jpg", s.style, s.fruit))   # CSV order: file_name,style,fruit
    csv_path = os.path.join(work, "test", "test_labels.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "style", "fruit"])
        w.writerows(rows)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(work):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, arcname=os.path.relpath(full, work))  # -> test/...
    shutil.rmtree(work, ignore_errors=True)

    fc = Counter(s.fruit for s in chosen)
    sc = Counter(s.style for s in chosen)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
    print("fruit:", {FRUIT_ID2NAME[k]: v for k, v in sorted(fc.items())})
    print("style:", {STYLE_ID2NAME[k]: v for k, v in sorted(sc.items())})


if __name__ == "__main__":
    main()
