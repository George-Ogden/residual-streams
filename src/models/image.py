from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models import ResNet
from torchvision import transforms

from .base import VariableLengthClassifierOutput, VariableLengthModelForClassification

@VariableLengthModelForClassification.register("resnet")
class VariableLengthResNetForImageClassification(VariableLengthModelForClassification):
    # models and weights
    MODELS = {
        "resnet18": (resnet18, ResNet18_Weights),
        "resnet34": (resnet34, ResNet34_Weights),
        "resnet50": (resnet50, ResNet50_Weights),
        "resnet101": (resnet101, ResNet101_Weights),
        "resnet152": (resnet152, ResNet152_Weights),
    }
    def __init__(self, model: ResNet, transform: transforms.Compose):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.torso = nn.ModuleList([
            getattr(model, name)
            for name in model._modules.keys()
            if name.startswith("layer")
        ])
        self.head = nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc,
        )
        self.transform = transform

    def forward(self, pixel_values: torch.Tensor) -> VariableLengthClassifierOutput:
        x = pixel_values

        layer_outputs = []
        layer_predictions = []
        x = self.feature_extractor(x)

        for layer in self.torso:
            if layer[0].downsample:
                # downsample the predictions at each layer
                layer_predictions = [
                    layer[0].downsample(x)
                    for x in layer_predictions
                ]
            for block in layer:
                x = block(x)
                # convert to [B, N, H]
                layer_outputs.append(x.mean(dim=(-1, -2)).unsqueeze(-2))
                layer_predictions.append(x)
        layer_predictions = [
            # make predictions for each layer
            self.head(x)
            for x in layer_predictions
        ]

        return VariableLengthClassifierOutput(
            layer_activations=layer_outputs,
            layer_predictions=layer_predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, j) for i, layer in enumerate(self.torso) for j in range(len(layer))
        ]
    
    @classmethod
    def _from_pretrained(cls, model_name: str) -> VariableLengthResNetForImageClassification:
        model, weights = VariableLengthResNetForImageClassification.MODELS[model_name]
        return cls(model(weights="DEFAULT"), weights.DEFAULT.transforms())
    
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # transform the images
        batch["pixel_values"] = [
            self.transform(image.convert("RGB")) for image in batch["image"]
        ]
        return batch