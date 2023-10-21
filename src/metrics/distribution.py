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
        # assert torch.all(sorted_values >= 0), (sorted_values.min(), sorted_values.max())
        if not torch.all(sorted_values > 0):
            sorted_values = torch.where(sorted_values <= 0, torch.zeros_like(sorted_values), sorted_values)
        normalised_values = sorted_values / sorted_values.sum(dim=-1, keepdim=True)
        return normalised_values