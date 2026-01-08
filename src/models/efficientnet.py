import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0Video(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feat_dim = 1280

        # 🔥 Frame Attention (KEY ADDITION)
        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Linear(self.feat_dim, 1)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  # (B*T, D)

        feats = feats.view(B, T, self.feat_dim)           # (B, T, D)

        # 🔥 Learnable attention weights over frames
        attn_scores = self.attention(feats)               # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        video_feat = (feats * attn_weights).sum(dim=1)    # (B, D)

        logits = self.classifier(video_feat).squeeze(1)
        return logits
    
    def extract_features(self, x):
        """
        Input: x (B, T, C, H, W)
        Output: (B, 1280) - temporally averaged global pooled features
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # Pass through backbone (which is already .features)
        feats = self.backbone(x)                    # shape: (B*T, 1280, 7, 7)

        # Global average pooling → (B*T, 1280)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.view(B * T, -1)               # (B*T, 1280)

        # Reshape back and average over time dimension
        feats = feats.view(B, T, -1)                # (B, T, 1280)
        pooled = feats.mean(dim=1)                  # (B, 1280)

        return pooled
