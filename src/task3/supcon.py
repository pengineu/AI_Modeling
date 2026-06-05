"""Supervised Contrastive (SupCon) embedding for Task3 retrieval.

Trains an EfficientNet-B0 backbone + projection head so that images sharing the
combined (fruit, style) label are close in the normalized embedding space, and
different ones are far. At inference the embedding is used for cosine Top-K
retrieval (image-only, label-blind -> genuine).

Reference: Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SupConEmbed(nn.Module):
    def __init__(self, proj_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        b = models.efficientnet_b0(weights=weights)
        in_feats = b.classifier[1].in_features  # 1280
        b.classifier = nn.Identity()
        self.backbone = b
        self.proj = nn.Sequential(
            nn.Linear(in_feats, in_feats), nn.ReLU(inplace=True),
            nn.Linear(in_feats, proj_dim),
        )

    def forward(self, x):
        """Return L2-normalized projection embedding (the retrieval space)."""
        return F.normalize(self.proj(self.backbone(x)), dim=-1)


def supcon_loss(z, labels, temperature: float = 0.07):
    """SupCon loss. z: (B, D) L2-normalized. labels: (B,) combined labels."""
    device = z.device
    B = z.size(0)
    sim = (z @ z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()       # stability
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    exp = torch.exp(sim).masked_fill(self_mask, 0.0)
    labels = labels.view(-1, 1)
    pos = (labels == labels.t()) & (~self_mask)                    # same label, not self
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    pos_cnt = pos.sum(1)
    mean_lp = (pos * log_prob).sum(1) / pos_cnt.clamp(min=1)
    has_pos = pos_cnt > 0
    if not has_pos.any():
        return z.sum() * 0.0
    return -(mean_lp[has_pos]).mean()
