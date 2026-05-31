"""Task 1 overfitting / leakage diagnostics (run on the RTX 4060: needs task1.pt + full data).

Answers "is the ~1.0 val accuracy real or inflated?" with five eval-only checks
(no backward, so it is safe everywhere):

  1. Train vs Val accuracy gap        -> classic over-fitting (train >> val).
  2. Val confusion matrices + errors  -> which classes/styles, if any, are confused.
  3. Val prediction confidence        -> calibration / memorization smell.
  4. Pixel-level near-duplicate leak  -> the key test: are val images near-identical
                                         to a TRAIN image? (random split would then
                                         inflate val acc and not transfer to test).
  5. Robustness under perturbation    -> how much val acc drops with color/blur/rotation
                                         (generalization margin).

Usage (RTX machine):
    python -m src.task1.verify --data data/train --ckpt checkpoints/task1.pt
"""

import argparse
import os
from collections import Counter

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.common.dataset import scan, FruitStyleDataset
from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME, NUM_FRUIT, NUM_STYLE
from src.task1.model import DualHeadClassifier
from src.task1.train import stratified_split, build_transforms


@torch.no_grad()
def predict_split(model, loader, device):
    """Return (fruit_pred, fruit_true, style_pred, style_true, fruit_conf) tensors."""
    fp, ft, sp, st, conf = [], [], [], [], []
    for x, fy, sy in loader:
        x = x.to(device)
        fl, sl = model(x)
        fp.append(fl.argmax(1).cpu()); ft.append(fy)
        sp.append(sl.argmax(1).cpu()); st.append(sy)
        conf.append(F.softmax(fl, 1).max(1).values.cpu())
    return (torch.cat(fp), torch.cat(ft), torch.cat(sp), torch.cat(st), torch.cat(conf))


def acc(pred, true):
    return (pred == true).float().mean().item()


def confusion(pred, true, n):
    m = torch.zeros(n, n, dtype=torch.int64)
    for p, t in zip(pred.tolist(), true.tolist()):
        m[t, p] += 1
    return m


@torch.no_grad()
def backbone_unused():  # placeholder to keep imports tidy
    pass


def pixel_signatures(paths, size=32):
    """Tiny grayscale pixel signatures (l2-normalized) for near-duplicate detection."""
    sigs = []
    tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((size, size)),
                             transforms.ToTensor()])
    for p in paths:
        v = tf(Image.open(p).convert("RGB")).flatten()
        v = v - v.mean()
        n = v.norm() + 1e-8
        sigs.append(v / n)
    return torch.stack(sigs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--ckpt", default="checkpoints/task1.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dup-thresh", type=float, default=0.95)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location=device)
    img_size = ckpt.get("img_size", 224) if isinstance(ckpt, dict) else 224
    model = DualHeadClassifier(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    samples = scan(args.data)
    labeled = [s for s in samples if s.fruit is not None and s.style is not None]
    # Reproduce the EXACT split used in training (same seed) so val == training's val.
    tr_idx, va_idx = stratified_split(labeled)
    _, eval_tf = build_transforms(img_size)
    tr_ds = Subset(FruitStyleDataset(labeled, eval_tf), tr_idx)
    va_ds = Subset(FruitStyleDataset(labeled, eval_tf), va_idx)
    tr_dl = DataLoader(tr_ds, args.batch, num_workers=args.workers)
    va_dl = DataLoader(va_ds, args.batch, num_workers=args.workers)
    print(f"labeled={len(labeled)}  train={len(tr_idx)}  val={len(va_idx)}")

    # ---- 1. Train vs Val accuracy gap ----
    print("\n=== 1) Train vs Val accuracy (gap = classic over-fitting) ===")
    fp_tr, ft_tr, sp_tr, st_tr, _ = predict_split(model, tr_dl, device)
    fp_va, ft_va, sp_va, st_va, conf_va = predict_split(model, va_dl, device)
    fa_tr, sa_tr = acc(fp_tr, ft_tr), acc(sp_tr, st_tr)
    fa_va, sa_va = acc(fp_va, ft_va), acc(sp_va, st_va)
    print(f"  fruit:  train {fa_tr:.4f}  val {fa_va:.4f}  gap {fa_tr - fa_va:+.4f}")
    print(f"  style:  train {sa_tr:.4f}  val {sa_va:.4f}  gap {sa_tr - sa_va:+.4f}")
    print("  (gap ~0 => not classic over-fit; large positive gap => memorizing train)")

    # ---- 2. Val confusion + misclassified ids ----
    print("\n=== 2) Val confusion matrices ===")
    print("  fruit (rows=true, cols=pred):", FRUIT_ID2NAME)
    print(confusion(fp_va, ft_va, NUM_FRUIT).tolist())
    print("  style (rows=true, cols=pred):", STYLE_ID2NAME)
    print(confusion(sp_va, st_va, NUM_STYLE).tolist())
    val_samples = [labeled[i] for i in va_idx]
    errs = [(val_samples[i].img_id, ft_va[i].item(), fp_va[i].item())
            for i in range(len(val_samples)) if fp_va[i] != ft_va[i]]
    print(f"  fruit misclassified ({len(errs)}): {errs[:20]}")

    # ---- 3. Confidence ----
    print("\n=== 3) Val fruit-prediction confidence (max softmax) ===")
    print(f"  mean {conf_va.mean():.4f}  min {conf_va.min():.4f}  "
          f"<0.6: {(conf_va < 0.6).sum().item()} samples")

    # ---- 4. Pixel-level near-duplicate leakage (train <-> val) ----
    print("\n=== 4) Near-duplicate leakage (val image vs nearest TRAIN image) ===")
    tr_paths = [labeled[i].path for i in tr_idx]
    va_paths = [s.path for s in val_samples]
    tr_sig = pixel_signatures(tr_paths).to(device)
    va_sig = pixel_signatures(va_paths).to(device)
    max_sim = torch.empty(len(va_paths))
    nn_idx = torch.empty(len(va_paths), dtype=torch.long)
    for i in range(0, len(va_paths), 256):
        sims = va_sig[i:i + 256] @ tr_sig.t()
        m, j = sims.max(dim=1)
        max_sim[i:i + 256] = m.cpu(); nn_idx[i:i + 256] = j.cpu()
    for thr in (0.90, 0.95, 0.98, 0.99):
        cnt = (max_sim >= thr).sum().item()
        print(f"  val imgs with a train near-dup >= {thr}:  {cnt}/{len(va_paths)} "
              f"({100*cnt/len(va_paths):.1f}%)")
    p95k = max(1, min(len(max_sim), int(0.95 * len(max_sim))))
    print(f"  max-sim distribution: mean {max_sim.mean():.3f}  median "
          f"{max_sim.median():.3f}  p95 {max_sim.kthvalue(p95k)[0]:.3f}")
    leak = (max_sim >= args.dup_thresh)
    if leak.any():
        # Of the leaked val images, how many share the train neighbour's fruit label?
        share = sum(val_samples[i].fruit == labeled[tr_idx[nn_idx[i]]].fruit
                    for i in range(len(va_paths)) if leak[i])
        print(f"  among {leak.sum().item()} leaked (>= {args.dup_thresh}), "
              f"{share} share the neighbour's fruit label "
              f"=> these val correct-preds may be inflated.")
        print("  INTERPRET: high % here means val acc is optimistic vs a truly novel test set.")
    else:
        print("  -> no strong pixel near-duplicates across the split: val acc is trustworthy.")

    # ---- 5. Robustness under perturbation ----
    print("\n=== 5) Robustness: val acc under perturbation (generalization margin) ===")
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    hard_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.GaussianBlur(3),
        transforms.RandomRotation(15),
        transforms.ToTensor(), norm,
    ])
    hard_dl = DataLoader(Subset(FruitStyleDataset(labeled, hard_tf), va_idx),
                         args.batch, num_workers=args.workers)
    fp_h, ft_h, sp_h, st_h, _ = predict_split(model, hard_dl, device)
    print(f"  fruit: clean {fa_va:.4f} -> perturbed {acc(fp_h, ft_h):.4f}")
    print(f"  style: clean {sa_va:.4f} -> perturbed {acc(sp_h, st_h):.4f}")
    print("  (small drop => robust/genuine features; large drop => brittle shortcuts)")

    print("\nDONE. Summary heuristic:")
    print("  - gap ~0 + low near-dup% + small perturbation drop  => acc is REAL, ship it.")
    print("  - high near-dup% (>~20-30%)                          => val inflated; report back.")


if __name__ == "__main__":
    main()
