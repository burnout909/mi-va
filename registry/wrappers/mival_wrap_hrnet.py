"""HRNetV2 delineation (MedicalAI-DP Hugging Face Space), single lead, 500 Hz.

The Space feeds each lead z-scored and then multiplied by 0.1 (its demo arrays
have per-lead SD 0.1) and thresholds three independent logit channels (P, QRS,
T) with fixed cut-offs. This module applies the x0.1 after the contract's
z-score and turns the three thresholded channels into four class scores
(none, P, QRS, T) so the shared mask decoder can read them; where channels
overlap, QRS wins over P and T, then P over T.
"""

import sys

import torch
from torch import nn

SPACE_CODE = "/opt/ECG_Delineation"
CUTOFFS = (0.001163482666015625, 0.15087890625, -0.587890625)


class _Config:
    pass


class HRNetV2Delineation(nn.Module):
    def __init__(self, data_len=5000):
        super().__init__()
        if SPACE_CODE not in sys.path:
            sys.path.insert(0, SPACE_CODE)
        from res.impl.HRNetV2 import HRNetV2

        config = _Config()
        config.data_len = data_len
        config.kernel_size = 5
        config.dilation = 1
        config.num_stages = 3
        config.num_blocks = 6
        config.num_modules = [1, 1, 1, 4, 3]
        config.use_bottleneck = [1, 0, 0, 0, 0]
        config.stage1_channels = 128
        config.num_channels_init = 48
        config.interpolate_mode = "linear"
        config.output_size = 3
        self.model = HRNetV2(config)
        self.register_buffer("cutoffs", torch.tensor(CUTOFFS).reshape(1, 3, 1), persistent=False)

    def forward(self, x):
        logits = self.model(x * 0.1)
        on = (logits >= self.cutoffs).float()
        p, qrs, t = on[:, 0], on[:, 1], on[:, 2]
        scores = torch.stack([0.5 * torch.ones_like(p), 2.0 * p, 3.0 * qrs, 1.5 * t], dim=1)
        return scores
