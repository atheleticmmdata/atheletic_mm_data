from __future__ import annotations

import torch
import torch.nn as nn


class FallbackTemporalBlock(nn.Module):
    """Lightweight temporal block used when mamba_ssm is unavailable."""

    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, 2 * d_model)
        self.dw_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size - 1,
        )
        self.proj_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        a, b = self.proj_in(x).chunk(2, dim=-1)
        x = a * torch.sigmoid(b)

        x = x.transpose(1, 2)
        x = self.dw_conv(x)
        x = x[:, :, : residual.size(1)]
        x = x.transpose(1, 2)

        x = self.proj_out(x)
        x = self.dropout(x)
        return x + residual


class MambaTemporalBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        try:
            from mamba_ssm import Mamba

            self.block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_mamba = True
        except Exception:
            self.block = FallbackTemporalBlock(d_model=d_model)
            self.use_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.block(x)
        return x + residual


class StreamEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, depth: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([MambaTemporalBlock(d_model=d_model) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x)


class MambaSportsNet(nn.Module):
    def __init__(
        self,
        kinematic_dim: int = 18,
        physiological_dim: int = 4,
        d_model: int = 128,
        depth: int = 4,
        fatigue_classes: int = 3,
        technique_classes: int = 3,
    ) -> None:
        super().__init__()
        self.kin_encoder = StreamEncoder(kinematic_dim, d_model=d_model, depth=depth)
        self.phy_encoder = StreamEncoder(physiological_dim, d_model=d_model, depth=depth)

        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.grf_head = nn.Linear(d_model, 1)
        self.fatigue_head = nn.Linear(d_model, fatigue_classes)
        self.technique_head = nn.Linear(d_model, technique_classes)

    def forward(self, kinematic: torch.Tensor, physiological: torch.Tensor):
        h_kin = self.kin_encoder(kinematic)
        h_phy = self.phy_encoder(physiological)

        g = self.gate(torch.cat([h_kin, h_phy], dim=-1))
        z = g * h_kin + (1.0 - g) * h_phy

        grf_pred = self.grf_head(z).squeeze(-1)
        pooled = z.mean(dim=1)
        fatigue_logits = self.fatigue_head(pooled)
        technique_logits = self.technique_head(pooled)

        return {
            "grf_pred": grf_pred,
            "fatigue_logits": fatigue_logits,
            "technique_logits": technique_logits,
        }
