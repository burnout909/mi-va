"""OpenECG codec_v6 frame head, single lead at 500 Hz, 10 s.

The codec takes (signal, lead_id) with the signal rank-normalized into [-1, 1]
per window and returns (frame, beat, rhythm) logits of shape (B, T, C). This
module rank-normalizes with the package's own function, passes lead II's id,
and returns the frame logits (B, T, 4: other, P, QRS, T); the card reads them
with class_axis 2. ``convert`` writes the codec's weights as a plain
state_dict of this module so the framework's strict load applies.
"""

import sys

import numpy as np
import torch
from torch import nn

OPENECG_CODE = "/opt/openecg"
LEAD_II = 1  # openecg LUDB lead order i, ii, iii, avr, avl, avf, v1..v6


def _openecg():
    if OPENECG_CODE not in sys.path:
        sys.path.insert(0, OPENECG_CODE)


class OpenEcgFrame(nn.Module):
    def __init__(self, ckpt="/data/mi-val/models/openecg-codec-v6/codec_v6.pt", lead_id=LEAD_II):
        super().__init__()
        _openecg()
        from openecg.stage2.model import load_model_from_ckpt

        self.codec, _ = load_model_from_ckpt(ckpt, device="cpu")
        self.lead_id = int(lead_id)

    def forward(self, x):
        from openecg.dsp import rank_normalize

        signal = x[:, 0, :].detach().cpu().numpy()
        normed = np.stack([np.asarray(rank_normalize(row), dtype=np.float32) for row in signal])
        tensor = torch.from_numpy(normed).to(x.device)
        lead = torch.full((x.shape[0],), self.lead_id, dtype=torch.long, device=x.device)
        out = self.codec(tensor, lead)
        return out[0]


def convert(target):
    module = OpenEcgFrame()
    torch.save(module.state_dict(), target)
