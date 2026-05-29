"""Task 1 training (run on the RTX 4060 desktop).

Fine-tunes a shared EfficientNet-B0 backbone with two heads (fruit/style) on the
labeled train set. Saves the best checkpoint to checkpoints/task1.pt for upload to
Google Drive (the submission notebook downloads & uses it for test inference).

Usage:
    python -m src.task1.train --data data/train --epochs 15 --batch 64
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.common.dataset import scan, FruitStyleDataset
from src.task1.model import DualHeadClassifier


def stratified_split(samples, val_frac=0.2, seed=42):
    """Split keeping each (fruit,style) cell proportionally represented."""
    import random
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, s in enumerate(samples):
        buckets[(s.fruit, s.style)].append(i)
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for idxs in buckets.values():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def build_transforms(img_size=224):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),       # mild only: strong color aug hurts style
        transforms.RandomRotation(8),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), norm,
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    f_ok = s_ok = n = 0
    for x, fy, sy in loader:
        x, fy, sy = x.to(device), fy.to(device), sy.to(device)
        fp, sp = model.predict(x)
        f_ok += (fp == fy).sum().item()
        s_ok += (sp == sy).sum().item()
        n += x.size(0)
    return f_ok / n, s_ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/task1.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    samples = scan(args.data)
    labeled = [s for s in samples if s.fruit is not None and s.style is not None]
    assert labeled, f"no labeled samples under {args.data} (need a label CSV)"
    print(f"labeled samples: {len(labeled)}")

    train_tf, eval_tf = build_transforms(args.img_size)
    tr_idx, va_idx = stratified_split(labeled)
    train_ds = Subset(FruitStyleDataset(labeled, train_tf), tr_idx)
    val_ds = Subset(FruitStyleDataset(labeled, eval_tf), va_idx)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = DualHeadClassifier(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, fy, sy in train_dl:
            x, fy, sy = x.to(device), fy.to(device), sy.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                fp, sp = model(x)
                loss = ce(fp, fy) + ce(sp, sy)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        f_acc, s_acc = evaluate(model, val_dl, device)
        mean_acc = (f_acc + s_acc) / 2
        print(f"epoch {ep:02d}  loss {running/len(train_ds):.4f}  "
              f"val fruit {f_acc:.4f}  style {s_acc:.4f}  mean {mean_acc:.4f}")
        if mean_acc > best:
            best = mean_acc
            torch.save({"model": model.state_dict(),
                        "img_size": args.img_size,
                        "val_fruit_acc": f_acc, "val_style_acc": s_acc}, args.out)
            print(f"  saved best -> {args.out}")
    print("done. best mean acc:", best)


if __name__ == "__main__":
    main()
