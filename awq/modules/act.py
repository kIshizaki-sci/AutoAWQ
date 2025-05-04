import torch.nn as nn


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        if len(x.shape)==3:
            return self.act(x) / self.scales.view(1, 1, -1).to(x.device)
        elif len(x.shape)==2:
            return self.act(x) / self.scales.view(1, -1).to(x.device)
