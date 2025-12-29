import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0Video(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)         # (B*T, D)
        feats = feats.view(B, T, -1)     # (B, T, D)

        feats = feats.mean(dim=1)        # temporal average pooling
        logits = self.classifier(feats)  # (B, 1)

        return logits.squeeze(1)
