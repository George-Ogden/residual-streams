import numpy as np
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("distribution")
class Distribution(Metric):
    """Mean normalised activation."""
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.value += np.array([
            self.process(activations.cpu()).mean(axis=-2).sum(-2)
            for activations in model_output.layer_activations
        ])
        super().update(batch, model_output)
    
    @classmethod
    def process(cls, activations: torch.Tensor) -> torch.Tensor:
        sorted_values = torch.sort(activations, dim=-1).values
        normalised_values = sorted_values / torch.abs(sorted_values).sum(dim=-1, keepdim=True)
        return normalised_values