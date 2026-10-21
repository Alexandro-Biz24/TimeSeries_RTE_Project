import torch
from torch import nn

from MTST.Models.full_model import MTST_full


class MTSTDayAhead(nn.Module):
    """
    Adapt original MTST_full to forecast 24 French clock-hour load slots.

    Inputs:
        x_history: (batch, input_len, history_dim)
        x_future:  (batch, 24, future_dim), known at forecast origin

    Output:
        (batch, 24), one load prediction per French local clock hour.

    The dataset's validity mask handles nonexistent/missing DST slots.
    """

    def __init__(
        self,
        history_dim,
        future_dim,
        embed_dim=64,
        heads=16,
        dropout=0.1,
        n_layers=2,
        max_len=5000,
        high_res=1,
        mid_res=6,
        low_res=24,
    ):
        super().__init__()

        if history_dim < 1 or future_dim < 1:
            raise ValueError("history_dim and future_dim must be positive")

        if min(high_res, mid_res, low_res) < 1:
            raise ValueError("Resolution factors must be positive integers")

        self.high_res = high_res
        self.mid_res = mid_res
        self.low_res = low_res

        # Retain the original multi-resolution Transformer
        self.backbone = MTST_full(
            input_dim=history_dim,
            embed_dim=embed_dim,
            heads=heads,
            dropout=dropout,
            n_layers=n_layers,
            output_len=24,
            max_len=max_len,
        )

        # Encode known future covariates
        self.future_encoder = nn.Sequential(
            nn.Linear(future_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        # Predict electricity demand for each target hour
        self.load_head = nn.Sequential(
            nn.Linear(history_dim + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x_history, x_future):

        if x_history.ndim != 3 or x_future.ndim != 3:
            raise ValueError(
                "Inputs must be 3D: (batch, time, features)"
            )

        if x_future.shape[1] != 24:
            raise ValueError(
                "x_future must have exactly 24 clock-hour slots"
            )

        if x_history.shape[0] != x_future.shape[0]:
            raise ValueError("Historical and future batch sizes differ")

        if x_history.shape[2] != self.backbone.input_dim:
            raise ValueError("Incorrect historical feature count")

        if x_future.shape[2] != self.future_encoder[0].in_features:
            raise ValueError("Incorrect future feature count")

        # Original MTST multi-resolution processing
        historical_forecast = self.backbone(
            x_history,
            self.high_res,
            self.mid_res,
            self.low_res,
        )

        # Known calendar / RTE features for D+1
        known_future = self.future_encoder(x_future)

        # Combine historical and future representations
        combined = torch.cat(
            [historical_forecast, known_future],
            dim=-1
        )

        # Final consumption forecast: (batch, 24)
        return self.load_head(combined).squeeze(-1)