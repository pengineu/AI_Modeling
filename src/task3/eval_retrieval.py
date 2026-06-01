"""Task 3 retrieval-quality evaluation (run on the RTX 4060: CLIP + data + GPU).

Quantifies whether zero-shot CLIP retrieval is good enough, so we can decide
*with evidence* whether any fine-tuning is warranted. Uses the labeled train set:
embed every image with CLIP, and for each image (query) take its Top-K nearest
neighbours by cosine similarity (self excluded), then report label-based
precision@K:

  - fruit P@K : fraction of the K neighbours sharing the query's fruit
  - style P@K : fraction sharing the query's style
  - both  P@K : fraction sharing BOTH fruit and style

Random baselines: fruit 1/6=0.167, style 1/3=0.333, both 1/18=0.056.
High P@K (well above baseline) => zero-shot CLIP retrieval is strong, no fine-tune
needed. Mirrors the submission notebook's retrieval exactly (CLIP get_image_features,
L2-normalized, cosine). Eval-only (no backward).

Usage:
    python -m src.task3.eval_retrieval --data data/train --ks 1,5,10
    python -m src.task3.eval_retrieval --max-n 1800        # quick stratified subsample
"""

import argparse
import random
from collections import defaultdict

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.common.dataset import scan
from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME


def retrieval_precision(embs, fruit, style, ks, batch=256):
    """embs: (N,D) L2-normalized. fruit,style: (N,) long. Returns metrics dict."""
    N = embs.size(0)
    maxk = max(ks)
    dev = embs.device
    fruit, style = fruit.to(dev), style.to(dev)
    tot = {k: [0, 0, 0] for k in ks}            # [fruit_hits, style_hits, both_hits]
    # per-class accumulators at the largest reported K
    rep = max(ks)
    pf = defaultdict(lambda: [0, 0]); ps = defaultdict(lambda: [0, 0])  # [hits, count]
    for i in range(0, N, batch):
        q = embs[i:i + batch]
        sims = q @ embs.t()                     # (b, N)
        idx = torch.arange(q.size(0), device=dev)
        sims[idx, i + idx] = -2.0               # exclude self
        nn = sims.topk(maxk, dim=1).indices     # (b, maxk)
        nf, ns = fruit[nn], style[nn]           # (b, maxk)
        qf = fruit[i:i + batch].unsqueeze(1)
        qs = style[i:i + batch].unsqueeze(1)
        fm = (nf == qf); sm = (ns == qs); bm = fm & sm
        for k in ks:
            tot[k][0] += fm[:, :k].sum().item()
            tot[k][1] += sm[:, :k].sum().item()
            tot[k][2] += bm[:, :k].sum().item()
        # per-class at rep
        for r in range(q.size(0)):
            f = qf[r, 0].item(); s = qs[r, 0].item()
            pf[f][0] += fm[r, :rep].sum().item(); pf[f][1] += rep
            ps[s][0] += sm[r, :rep].sum().item(); ps[s][1] += rep
    out = {k: tuple(v / (N * k) for v in tot[k]) for k in ks}
    return out, rep, pf, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--max-n", type=int, default=0, help="0=all; else stratified subsample")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    samples = [s for s in scan(args.data) if s.fruit is not None and s.style is not None]
    if args.max_n and args.max_n < len(samples):
        buckets = defaultdict(list)
        for s in samples:
            buckets[(s.fruit, s.style)].append(s)
        rng = random.Random(0); per = max(1, args.max_n // len(buckets))
        sub = []
        for v in buckets.values():
            rng.shuffle(v); sub += v[:per]
        samples = sub
    print(f"images: {len(samples)}")

    # use_safetensors=True avoids the .bin torch.load path that newer transformers
    # block on torch<2.6 (CVE-2025-32434). Same weights as the .bin, so the
    # measured embeddings match the submission notebook exactly.
    clip = CLIPModel.from_pretrained(args.clip, use_safetensors=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    @torch.no_grad()
    def embed(paths):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        b = proc(images=imgs, return_tensors="pt").to(device)
        return torch.nn.functional.normalize(clip.get_image_features(**b), dim=-1)

    paths = [s.path for s in samples]
    embs = []
    for i in range(0, len(paths), args.batch):
        embs.append(embed(paths[i:i + args.batch]))
    embs = torch.cat(embs)
    fruit = torch.tensor([s.fruit for s in samples])
    style = torch.tensor([s.style for s in samples])
    print("embeddings:", tuple(embs.shape))

    metrics, rep, pf, ps = retrieval_precision(embs, fruit, style, ks, batch=256)

    print("\n=== retrieval precision@K (zero-shot CLIP) ===")
    print(f"{'K':>4} | {'fruit P@K':>10} | {'style P@K':>10} | {'both P@K':>9}")
    print(f"{'rand':>4} | {1/6:>10.3f} | {1/3:>10.3f} | {1/18:>9.3f}")
    for k in ks:
        f, s, b = metrics[k]
        print(f"{k:>4} | {f:>10.3f} | {s:>10.3f} | {b:>9.3f}")

    print(f"\n=== per-class P@{rep} ===")
    print("fruit:")
    for fid in sorted(pf):
        h, c = pf[fid]; print(f"  {FRUIT_ID2NAME[fid]:>10}: {h/c:.3f}")
    print("style:")
    for sid in sorted(ps):
        h, c = ps[sid]; print(f"  {STYLE_ID2NAME[sid]:>13}: {h/c:.3f}")

    print("\nVERDICT GUIDE:")
    print("  fruit/style P@K 가 baseline 대비 크게 높으면(예: fruit>0.8, style>0.8)")
    print("  -> zero-shot CLIP retrieval 충분, fine-tuning 불필요.")
    print("  baseline 근처로 낮은 클래스가 있으면 -> 그 축에 한해 fine-tuning 검토.")


if __name__ == "__main__":
    main()
