import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetB4Video(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # Remove classifier
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # EfficientNet-B4 feature dim = 1792
        self.classifier = nn.Linear(1792, 1)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)

        feats = feats.view(B, T, -1).mean(dim=1)
        logits = self.classifier(feats).squeeze(1)

        return logits
