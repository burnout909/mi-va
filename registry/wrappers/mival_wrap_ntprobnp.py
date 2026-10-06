"""AI-NT-proBNP (Hamburg City Health Study), 12 leads at 250 Hz, first 2048 samples.

After the contract's 0.5 Hz Butterworth high-pass and crop, the authors apply
neurokit2's 50 Hz powerline filter (a 5-sample moving average run forwards
and backwards at 250 Hz), z-score each lead and subtract its median.
"""

import sys

import numpy as np
import torch
from torch import nn

NTPROBNP_CODE = "/opt/AI-NT-proBNP"


def _load_file(name, path):
    """Import one author file under a unique module name (several repos call their package ``src``)."""
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class AiNtProBnp(nn.Module):
    def __init__(self, fs=250, powerline=50):
        super().__init__()
        ResNet1d = _load_file("mival_ext_ntprobnp_model", NTPROBNP_CODE + "/src/model.py").ResNet1d

        self.net = ResNet1d(input_dim=(12, 2048),
                            blocks_dim=list(zip([64, 128, 196, 256, 320], [2048, 1024, 256, 64, 16])),
                            n_classes=1, kernel_size=17, dropout_rate=0.2)
        self.width = int(fs / powerline)

    def forward(self, x):
        from scipy import signal as sgn

        raw = x.detach().cpu().numpy().astype(np.float64)
        b = np.ones(self.width)
        clean = sgn.filtfilt(b, [len(b)], raw, axis=-1)
        mean = clean.mean(axis=-1, keepdims=True)
        std = clean.std(axis=-1, keepdims=True)
        std[std == 0] = 1.0
        scaled = (clean - mean) / std
        scaled = scaled - np.median(scaled, axis=-1, keepdims=True)
        return self.net(torch.from_numpy(scaled.astype(np.float32)).to(x.device))
