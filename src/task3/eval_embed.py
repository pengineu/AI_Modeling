"""Compare Task3 retrieval embeddings by label precision@K (run on the RTX 4060).

Measures fruit/style/both precision@K (self-excluded Top-K) for a chosen image
embedding on the labeled train set, so we can pick the best GENUINE embedding
(label-blind retrieval) for the label-match grading.

  --method clip    : CLIP ViT-B/32 (public)            [needs transformers]
  --method task1   : Task1 backbone features            [--ckpt checkpoints/task1.pt]
  --method supcon  : SupCon projection embedding        [--ckpt checkpoints/supcon.pt]

Usage:
    python -m src.task3.eval_embed --method task1  --ckpt checkpoints/task1.pt
    python -m src.task3.eval_embed --method supcon --ckpt checkpoints/supcon.pt
    python -m src.task3.eval_embed --method clip
"""

import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.common.dataset import scan
from src.common.labels import FRUIT_ID2NAME, STYLE_ID2NAME
from src.task3.eval_retrieval import retrieval_precision

_IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(*_IMNET)])


def build_embedder(method, ckpt, device):
    if method == "clip":
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-base-patch32"
        try:
            clip = CLIPModel.from_pretrained(name, use_safetensors=True)
        except Exception:
            clip = CLIPModel.from_pretrained(name)
        clip = clip.to(device).eval()
        proc = CLIPProcessor.from_pretrained(name)

        @torch.no_grad()
        def embed(paths):
            imgs = [Image.open(p).convert("RGB") for p in paths]
            b = proc(images=imgs, return_tensors="pt").to(device)
            feat = clip.get_image_features(**b)
            if not torch.is_tensor(feat):
                feat = feat.pooler_output
            return torch.nn.functional.normalize(feat, dim=-1)
        return embed

    if method == "task1":
        from src.task1.model import DualHeadClassifier
        ck = torch.load(ckpt, map_location=device)
        m = DualHeadClassifier(pretrained=False).to(device)
        m.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
        m.eval()
        tf = _tf(ck.get("img_size", 224) if isinstance(ck, dict) else 224)

        @torch.no_grad()
        def embed(paths):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)
            return torch.nn.functional.normalize(m.backbone(x), dim=-1)
        return embed

    if method == "supcon":
        from src.task3.supcon import SupConEmbed
        ck = torch.load(ckpt, map_location=device)
        m = SupConEmbed(proj_dim=ck.get("proj_dim", 128), pretrained=False).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        tf = _tf(ck.get("img_size", 224))

        @torch.no_grad()
        def embed(paths):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(device)
            return m(x)
        return embed

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--method", choices=["clip", "task1", "supcon"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ks", default="1,5,10")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.ks.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| method:", args.method)
    samples = [s for s in scan(args.data) if s.fruit is not None and s.style is not None]
    print("images:", len(samples))

    embed = build_embedder(args.method, args.ckpt, device)
    paths = [s.path for s in samples]
    embs = torch.cat([embed(paths[i:i + args.batch]) for i in range(0, len(paths), args.batch)])
    fruit = torch.tensor([s.fruit for s in samples])
    style = torch.tensor([s.style for s in samples])
    print("embeddings:", tuple(embs.shape))

    metrics, rep, pf, ps = retrieval_precision(embs, fruit, style, ks, batch=256)
    print(f"\n=== precision@K ({args.method}) ===")
    print(f"{'K':>4} | {'fruit':>7} | {'style':>7} | {'both':>7}")
    print(f"{'rand':>4} | {1/6:>7.3f} | {1/3:>7.3f} | {1/18:>7.3f}")
    for k in ks:
        f, s, b = metrics[k]
        print(f"{k:>4} | {f:>7.3f} | {s:>7.3f} | {b:>7.3f}")
    print(f"\nper-class P@{rep}  fruit:", {FRUIT_ID2NAME[i]: round(v[0]/v[1], 3) for i, v in sorted(pf.items())})
    print(f"per-class P@{rep}  style:", {STYLE_ID2NAME[i]: round(v[0]/v[1], 3) for i, v in sorted(ps.items())})


if __name__ == "__main__":
    main()
