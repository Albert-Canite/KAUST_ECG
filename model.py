import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentAwareCNN(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.conv_segments = nn.ModuleList([
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(1, 4, kernel_size=4, stride=1, padding=0),
        ])
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 360)
        segments = [
            x[:, :, 0:120],
            x[:, :, 120:240],
            x[:, :, 240:360],
            x,
        ]
        pooled = []
        for seg, conv in zip(segments, self.conv_segments):
            out = conv(seg)
            out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
            pooled.append(out)
        feats = torch.cat(pooled, dim=1)
        feats = F.relu(self.fc1(feats))
        feats = self.dropout(feats)
        logits = self.fc2(feats)
        return logits
