"""Objectively pick the best StarGAN snapshot (run on RTX 4060: needs ckpts + data + GPU).

We previously eyeballed iter-100k under the OLD (running-stats) rendering. Now that
inference uses per-instance stats, re-rank ALL snapshots fairly with the canonical
StarGAN metric: use the (validated) Task1 classifier on generated images to measure

  * style-match rate  : predicted style == target style   (translation success)
  * fruit-keep rate   : predicted fruit == original fruit  (content preservation)

on a FIXED, diverse, stratified batch of real images, with the exact submission-time
inference (instance-stats). Eval-only (no backward) -> safe everywhere.

Usage:
    python -m src.task2.select_ckpt --data data/train --task1 checkpoints/task1.pt \
        --ckpt-dir checkpoints --n 120
Outputs a ranked table and saves grids for the top candidates to report_samples/.
"""

import argparse
import glob
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

from src.common.dataset import scan
from src.common.labels import NUM_STYLE, NUM_FRUIT
from src.task2.models import Generator, label2onehot
from src.task1.model import DualHeadClassifier

_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_generator(path, device):
    ck = torch.load(path, map_location=device)
    G = Generator(c_dim=ck.get("c_dim", NUM_STYLE)).to(device)
    G.load_state_dict(ck["model"] if "model" in ck else ck)
    G.eval()
    # Submission-time inference: per-instance InstanceNorm stats (no running-stats cast).
    for m in G.modules():
        if isinstance(m, nn.InstanceNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return G, ck.get("img_size", 128)


def stratified_sample(samples, n, seed=0):
    buckets = defaultdict(list)
    for s in samples:
        buckets[(s.fruit, s.style)].append(s)
    rng = random.Random(seed)
    per = max(1, n // len(buckets))
    chosen = []
    for v in buckets.values():
        rng.shuffle(v)
        chosen += v[:per]
    return chosen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--task1", default="checkpoints/task1.pt")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--glob", default="stargan_G_*.pt")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    mean, std = _IMNET_MEAN.to(device), _IMNET_STD.to(device)

    # Task1 classifier (judge). EfficientNet/BatchNorm -> standard eval() is correct.
    cck = torch.load(args.task1, map_location=device)
    clf = DualHeadClassifier(pretrained=False).to(device)
    clf.load_state_dict(cck["model"] if isinstance(cck, dict) and "model" in cck else cck)
    clf.eval()
    c_size = cck.get("img_size", 224) if isinstance(cck, dict) else 224

    # Fixed diverse batch of real images with known labels.
    samples = [s for s in scan(args.data) if s.fruit is not None and s.style is not None]
    chosen = stratified_sample(samples, args.n)
    print(f"eval batch: {len(chosen)} images")
    gen_tf = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])

    @torch.no_grad()
    def judge(gen01):
        # gen01: (B,3,128,128) in [0,1] -> classifier input (resize to c_size + imagenet norm)
        x = F.interpolate(gen01, size=c_size, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        fl, sl = clf(x)
        return fl.argmax(1), sl.argmax(1)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, args.glob)))
    assert ckpts, f"no checkpoints matching {args.glob} in {args.ckpt_dir}"

    results = []
    for path in ckpts:
        G, gsize = load_generator(path, device)
        n_transfer = sty_ok = fr_ok = 0
        for i in range(0, len(chosen), args.batch):
            bs = chosen[i:i + args.batch]
            x = torch.stack([gen_tf(Image.open(s.path).convert("RGB")) for s in bs]).to(device)
            forig = torch.tensor([s.fruit for s in bs], device=device)
            sorig = torch.tensor([s.style for s in bs], device=device)
            for tgt in range(NUM_STYLE):
                y = (G(x, label2onehot(torch.full((x.size(0),), tgt)).to(device)) + 1) / 2
                fp, sp = judge(y)
                transfer = sorig != tgt           # only count genuine style changes
                n_transfer += transfer.sum().item()
                sty_ok += ((sp == tgt) & transfer).sum().item()
                fr_ok += ((fp == forig) & transfer).sum().item()
        sty_rate = sty_ok / max(1, n_transfer)
        fr_rate = fr_ok / max(1, n_transfer)
        combined = (sty_rate + fr_rate) / 2
        it = os.path.basename(path).replace("stargan_G_", "").replace(".pt", "")
        results.append((it, sty_rate, fr_rate, combined, path))
        print(f"  {os.path.basename(path):26s} style {sty_rate:.3f}  fruit {fr_rate:.3f}  "
              f"combined {combined:.3f}")

    print("\n=== ranking (by combined: style-match + fruit-keep) ===")
    results.sort(key=lambda r: -r[3])
    for rank, (it, s, f, c, _) in enumerate(results, 1):
        print(f"  {rank:2d}. iter {it:>7}  style {s:.3f}  fruit {f:.3f}  combined {c:.3f}")
    best = results[0]
    print(f"\nRECOMMENDED: stargan_G_{best[0]}.pt  (combined {best[3]:.3f})")

    # Visual grids for the top candidates (fixed 8 images, all 3 styles).
    os.makedirs("report_samples", exist_ok=True)
    vis = stratified_sample(samples, 8, seed=1)
    xv = torch.stack([gen_tf(Image.open(s.path).convert("RGB")) for s in vis]).to(device)
    for it, _, _, _, path in results[:args.top]:
        G, _ = load_generator(path, device)
        with torch.no_grad():
            cols = [xv] + [G(xv, label2onehot(torch.full((xv.size(0),), s)).to(device))
                           for s in range(NUM_STYLE)]
            grid = torch.cat(cols, 0)
        out = f"report_samples/select_{it}.png"
        vutils.save_image((grid + 1) / 2, out, nrow=xv.size(0))
        print(f"  saved {out}")
    print("\nDONE. Pick by combined score; break ties by eye on report_samples/select_*.png.")


if __name__ == "__main__":
    main()
