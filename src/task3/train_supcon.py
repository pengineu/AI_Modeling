"""Train the SupCon retrieval embedding (run on the RTX 4060).

Two augmented views per image, supervised-contrastive loss on the combined
(fruit, style) label (18 classes). Augmentation is geometry-mostly (no heavy
color jitter / grayscale) so STYLE cues (pencil/oil/water) are preserved.
Saves checkpoints/supcon.pt for Task3 retrieval.

Usage:
    python -m src.task3.train_supcon --data data/train --epochs 30 --batch 64
"""

import argparse
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.common.dataset import scan
from src.task3.supcon import SupConEmbed, supcon_loss


class TwoViewDataset(Dataset):
    """Returns (view1, view2, combined_label) where label = fruit*3 + style."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        return self.transform(img), self.transform(img), s.fruit * 3 + s.style


def build_aug(img_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # style-preserving: geometric aug + only very mild color (no grayscale)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.0),
        transforms.ToTensor(), norm,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/supcon.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    samples = [s for s in scan(args.data) if s.fruit is not None and s.style is not None]
    assert samples, f"no labeled samples under {args.data}"
    print("train images:", len(samples))
    ds = TwoViewDataset(samples, build_aug(args.img_size))
    loader = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

    model = SupConEmbed(proj_dim=args.proj_dim, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for v1, v2, y in loader:
            x = torch.cat([v1, v2], 0).to(device, non_blocking=True)
            y2 = torch.cat([y, y], 0).to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                z = model(x)
                loss = supcon_loss(z, y2, args.temp)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * v1.size(0)
        sched.step()
        avg = running / len(ds)
        print(f"epoch {ep:02d}  supcon_loss {avg:.4f}", flush=True)
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict(), "img_size": args.img_size,
                        "proj_dim": args.proj_dim}, args.out)
            print(f"  saved -> {args.out}", flush=True)
    print("done. best loss:", best)


if __name__ == "__main__":
    main()
