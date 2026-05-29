"""Dataset scanning and loading.

Confirmed dataset layout (Step 0):
    <root>/images/<id>.jpg          # 0.jpg .. 7199.jpg, flat
    <root>/<labels_csv>             # columns: file_name,style,fruit

`img_id` (used in all output files) == the image file name, e.g. "0.jpg".
Test set has the same structure; at eval its label CSV is NOT provided, so the
loaders degrade gracefully (labels become -1 / None) when no CSV is present.

Quick check:  python -m src.common.dataset data/train
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from glob import glob
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class Sample:
    path: str
    img_id: str            # file name WITH extension, e.g. "0.jpg" (the output id)
    fruit: Optional[int]   # None when no label CSV (test time)
    style: Optional[int]


def _find_images_dir(root: str) -> str:
    cand = os.path.join(root, "images")
    return cand if os.path.isdir(cand) else root


def _find_label_csv(root: str) -> Optional[str]:
    hits = glob(os.path.join(root, "*.csv")) + glob(os.path.join(root, "**", "*.csv"), recursive=True)
    return sorted(hits)[0] if hits else None


def _read_labels(csv_path: str) -> dict[str, tuple[int, int]]:
    """file_name -> (fruit, style). Header order in file is file_name,style,fruit."""
    out: dict[str, tuple[int, int]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["file_name"].strip()
            out[fn] = (int(row["fruit"]), int(row["style"]))
    return out


def scan(root: str) -> list[Sample]:
    """Scan `<root>/images` for jpgs; attach labels from the CSV if present."""
    img_dir = _find_images_dir(root)
    files = []
    for ext in IMG_EXTS:
        files += glob(os.path.join(img_dir, f"*{ext}"))
        files += glob(os.path.join(img_dir, f"*{ext.upper()}"))
    files = sorted(set(files), key=lambda p: _id_key(p))

    csv_path = _find_label_csv(root)
    labels = _read_labels(csv_path) if csv_path else {}

    samples = []
    for f in files:
        fn = os.path.basename(f)
        fruit, style = labels.get(fn, (None, None))
        samples.append(Sample(path=f, img_id=fn, fruit=fruit, style=style))
    return samples


def _id_key(path: str):
    """Sort numerically by id when possible (0,1,2,...,10), else lexicographically."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


class FruitStyleDataset(Dataset):
    """Returns (image_tensor, fruit_id, style_id). Unlabeled -> -1."""

    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        f = s.fruit if s.fruit is not None else -1
        st = s.style if s.style is not None else -1
        return img, f, st


if __name__ == "__main__":
    import sys
    from collections import Counter

    root = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    samples = scan(root)
    print(f"root={root}  images={len(samples)}  labeled={sum(s.fruit is not None for s in samples)}")
    fc = Counter(s.fruit for s in samples)
    sc = Counter(s.style for s in samples)
    print("fruit:", {FRUIT_ID2NAME.get(k, k): v for k, v in sorted(fc.items(), key=lambda x: (x[0] is None, x[0]))})
    print("style:", {STYLE_ID2NAME.get(k, k): v for k, v in sorted(sc.items(), key=lambda x: (x[0] is None, x[0]))})
    for s in samples[:3] + samples[-2:]:
        print("  e.g.", s.img_id, "-> fruit", s.fruit, "style", s.style)
