"""Task 2 StarGAN training (run on the RTX 4060 desktop).

Multi-domain (3 styles) image-to-image translation. A single generator learns to
restyle a fruit image into any target style while preserving content (via cycle
reconstruction). Saves checkpoints/stargan_G.pt for the submission notebook.

Stability (after the iter-14k mode collapse, see TASK2_COLLAPSE_REPORT.md):
  - Discriminator uses spectral normalization (Lipschitz constraint) -> D no longer
    overpowers G. This is the primary fix.
  - n_critic default 1 (balanced D:G updates).
  - Generator EMA -> smoother, more stable samples; EMA weights are what we save.
  - Gradient clipping on G/D -> tames the adv-loss explosion spikes.
  - Checkpoints are also saved iter-tagged (stargan_G_<iter>.pt) so a later collapse
    never destroys earlier good weights.
None of this changes the Generator architecture / state_dict keys, so the submission
notebook's inlined Generator and load_state_dict stay compatible.

Usage:
    python -m src.task2.train --data data/train --iters 200000 --batch 16
Smoke / early check (sample often!):
    python -m src.task2.train --iters 2000 --sample-every 200
"""

import argparse
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from src.common.dataset import scan, FruitStyleDataset
from src.common.labels import NUM_STYLE
from src.task2.models import Generator, Discriminator, label2onehot


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for pe, pm in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-critic", type=int, default=1)      # was 5; SN-D + 1:1 is stable
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-rec", type=float, default=10.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=5.0)  # 0 to disable
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--out", default="checkpoints/stargan_G.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("samples_task2", exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1] for Tanh
    ])
    samples = [s for s in scan(args.data) if s.style is not None]
    assert samples, f"no styled samples under {args.data}"
    print("train images:", len(samples))
    ds = FruitStyleDataset(samples, tf)
    loader = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    data_iter = infinite(loader)

    G = Generator(c_dim=NUM_STYLE).to(device)
    D = Discriminator(image_size=args.img_size, c_dim=NUM_STYLE).to(device)
    G_ema = copy.deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    g_opt = torch.optim.Adam(G.parameters(), args.lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), args.lr, betas=(0.5, 0.999))

    def save_ckpt(path, it):
        torch.save({"model": G_ema.state_dict(), "img_size": args.img_size,
                    "c_dim": NUM_STYLE, "iter": it}, path)

    # Fixed batch for visual progress sampling
    fixed_x, _, _ = next(data_iter)
    fixed_x = fixed_x.to(device)

    g_adv = g_cls = g_rec = torch.tensor(float("nan"))
    for it in range(1, args.iters + 1):
        x_real, _, style_org = next(data_iter)
        x_real = x_real.to(device)
        style_org = style_org.to(device)
        style_trg = style_org[torch.randperm(style_org.size(0))]
        c_org = label2onehot(style_org).to(device)
        c_trg = label2onehot(style_trg).to(device)

        # ===== Train D (hinge adversarial loss; D is spectral-normalized) =====
        src_real, cls_real = D(x_real)
        d_real = F.relu(1.0 - src_real).mean()
        d_cls = F.cross_entropy(cls_real, style_org)
        x_fake = G(x_real, c_trg)
        src_fake, _ = D(x_fake.detach())
        d_fake = F.relu(1.0 + src_fake).mean()
        d_loss = d_real + d_fake + args.lambda_cls * d_cls
        d_opt.zero_grad(); d_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(D.parameters(), args.grad_clip)
        d_opt.step()

        # ===== Train G (every n_critic) =====
        if it % args.n_critic == 0:
            x_fake = G(x_real, c_trg)
            src_fake, cls_fake = D(x_fake)
            g_adv = -src_fake.mean()
            g_cls = F.cross_entropy(cls_fake, style_trg)
            x_rec = G(x_fake, c_org)
            g_rec = (x_real - x_rec).abs().mean()
            g_loss = g_adv + args.lambda_cls * g_cls + args.lambda_rec * g_rec
            g_opt.zero_grad(); g_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), args.grad_clip)
            g_opt.step()
            update_ema(G_ema, G, args.ema_decay)

        if it % args.log_every == 0:
            print(f"iter {it:6d}/{args.iters}  D {d_loss.item():.3f}  "
                  f"G_adv {g_adv.item():.3f}  G_cls {g_cls.item():.3f}  "
                  f"G_rec {g_rec.item():.3f}", flush=True)

        if it % args.sample_every == 0:
            with torch.no_grad():
                cols = [fixed_x]
                for s in range(NUM_STYLE):
                    c = label2onehot(torch.full((fixed_x.size(0),), s)).to(device)
                    cols.append(G_ema(fixed_x, c))
                grid = torch.cat(cols, dim=0)
                save_image((grid + 1) / 2, f"samples_task2/iter_{it:06d}.png",
                           nrow=fixed_x.size(0))

        if it % args.save_every == 0 or it == args.iters:
            save_ckpt(args.out, it)                                          # latest (EMA)
            tagged = os.path.join(os.path.dirname(args.out) or ".",
                                  f"stargan_G_{it:06d}.pt")
            save_ckpt(tagged, it)                                            # history snapshot
            print(f"  saved EMA G -> {args.out} and {tagged} (iter {it})", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
