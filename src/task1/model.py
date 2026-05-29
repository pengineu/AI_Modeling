"""Task 1 model: shared EfficientNet-B0 backbone with two classification heads.

One backbone, two linear heads:
  - fruit head : 6-way (apple ... pineapple)
  - style head : 3-way (pencil / oil / water)

Uses torchvision so the submission notebook needs no extra dependency beyond torch.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.common.labels import NUM_FRUIT, NUM_STYLE


class DualHeadClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()            # keep pooled features
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fruit_head = nn.Linear(in_feats, NUM_FRUIT)
        self.style_head = nn.Linear(in_feats, NUM_STYLE)

    def forward(self, x):
        feats = self.dropout(self.backbone(x))
        return self.fruit_head(feats), self.style_head(feats)

    @torch.no_grad()
    def predict(self, x):
        """Return (fruit_label, style_label) tensors of predicted ids."""
        fruit_logits, style_logits = self.forward(x)
        return fruit_logits.argmax(1), style_logits.argmax(1)


# Inference-time preprocessing must match training. Exposed so the notebook reuses it.
def build_transform(img_size: int = 224):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
