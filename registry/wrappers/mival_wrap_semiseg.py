"""SemiSegECG encoder-decoder (VUNO), single lead, 4 classes per sample.

The published ``EncoderDecoder.forward`` returns a dict; this module returns
the ``seg_logits`` tensor (B, 4, T) interpolated back to the input length, as
the published forward does. Built from the checkpoint's own config.
"""

import sys

import torch.nn.functional as F
from torch import nn

SEMISEG_CODE = "/opt/semi-seg-ecg/src"


class SemiSegEncoderDecoder(nn.Module):
    def __init__(self, backbone, decode_head):
        super().__init__()
        if SEMISEG_CODE not in sys.path:
            sys.path.insert(0, SEMISEG_CODE)
        # algorithms.base.init_model_from_cfg does the same two lookups but
        # imports the training stack (tensorboard) at module level.
        from models import backbones, decode_heads

        (backbone_name, backbone_kwargs), = backbone.items()
        (head_name, head_kwargs), = decode_head.items()
        self.backbone = backbones.__dict__[backbone_name](**backbone_kwargs)
        self.decode_head = decode_heads.__dict__[head_name](**head_kwargs)

    def forward(self, x):
        logits = self.decode_head(self.backbone(x))
        return F.interpolate(logits, size=x.size(2), mode="linear",
                             align_corners=self.decode_head.align_corners)
