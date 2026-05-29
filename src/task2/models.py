"""Task 2: StarGAN v1 generator and discriminator (multi-domain style transfer).

A single generator transfers an image into any of the 3 style domains, conditioned
on a target-style one-hot map. The discriminator gives a real/fake patch score plus
a style classification logit. Designed for 128x128 to fit an 8GB RTX 4060.

Reference: Choi et al., "StarGAN" (CVPR 2018).
"""

import torch
import torch.nn as nn

from src.common.labels import NUM_STYLE


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """G(image, target_style_onehot) -> image in the target style.

    The style label is broadcast as extra channels concatenated to the input.
    """

    def __init__(self, conv_dim=64, c_dim=NUM_STYLE, n_res=6):
        super().__init__()
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]
        # Down-sampling x2
        cur = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(cur, cur * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(cur * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            cur *= 2
        # Bottleneck
        for _ in range(n_res):
            layers.append(ResidualBlock(cur))
        # Up-sampling x2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(cur, cur // 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(cur // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            cur //= 2
        layers += [nn.Conv2d(cur, 3, 7, 1, 3, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # c: (B, c_dim) one-hot -> (B, c_dim, H, W) spatial replication
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    """PatchGAN: outputs (src_score, style_logits)."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=NUM_STYLE, n_layers=6):
        super().__init__()
        layers = [nn.Conv2d(3, conv_dim, 4, 2, 1), nn.LeakyReLU(0.01)]
        cur = conv_dim
        for _ in range(1, n_layers):
            layers += [nn.Conv2d(cur, cur * 2, 4, 2, 1), nn.LeakyReLU(0.01)]
            cur *= 2
        self.main = nn.Sequential(*layers)
        ksize = image_size // (2 ** n_layers)
        self.src = nn.Conv2d(cur, 1, 3, 1, 1, bias=False)        # patch real/fake
        self.cls = nn.Conv2d(cur, c_dim, ksize, bias=False)      # style logits

    def forward(self, x):
        h = self.main(x)
        return self.src(h), self.cls(h).view(x.size(0), -1)


def label2onehot(labels: torch.Tensor, dim: int = NUM_STYLE) -> torch.Tensor:
    out = torch.zeros(labels.size(0), dim, device=labels.device)
    out[torch.arange(labels.size(0)), labels.long()] = 1.0
    return out
