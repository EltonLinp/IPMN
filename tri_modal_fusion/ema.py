from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """
    Lightweight EMA tracker storing weights on CPU.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._update(model, initialize=True)

    def _update(self, model: nn.Module, initialize: bool = False) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if not param.dtype.is_floating_point:
                    continue
                if initialize or name not in self.shadow:
                    self.shadow[name] = param.detach().cpu().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.detach().cpu(), alpha=1.0 - self.decay)

    def update(self, model: nn.Module) -> None:
        self._update(model, initialize=False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.shadow)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    param.copy_(self.shadow[name].to(param.device, dtype=param.dtype))
