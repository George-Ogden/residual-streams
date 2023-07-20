from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, BertForSequenceClassification, BertTokenizer, PreTrainedTokenizer, RobertaForSequenceClassification, RobertaTokenizer, GPT2ForSequenceClassification, GPT2Tokenizer
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from torchvision.models import ResNet

@dataclass
class VariableLengthClassifierOutput(ModelOutput):
    layer_activations: Optional[List[torch.FloatTensor]] = None
    layer_predictions: Optional[torch.FloatTensor] = None

class VariableLengthModelForClassification(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        ...
    
    @abc.abstractproperty
    def layers(self) -> List[Tuple[int, int]]:
        """Returns a list of tuples of layer indices, where each tuple represents a layer group."""

    @abc.abstractstaticmethod
    def from_pretrained(model_name: str) -> VariableLengthModelForClassification:
        ...
    
class ReducedLengthModelForSequenceClassification(VariableLengthModelForClassification):
    @abc.abstractproperty
    def model(self) -> PreTrainedModel:
        ...
    
    @abc.abstractproperty
    def tokenizer(self) -> PreTrainedTokenizer:
        ...

    @abc.abstractproperty
    def torso(self) -> nn.Module:
        ...
    
    @abc.abstractproperty
    def head(self) -> Optional[nn.Module]:
        ...
    
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        kwargs |= {
            "output_hidden_states": True,
        }
        outputs: SequenceClassifierOutput = self.model(*args, **kwargs)
        predictions = None
        if self.head is not None:
            predictions = [self.head(hidden_state) for hidden_state in outputs.hidden_states]
        return VariableLengthClassifierOutput(
            layer_activations=outputs.hidden_states,
            layer_predictions=predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]

class ReducedLengthModelForImageClassification(VariableLengthModelForClassification):
    def __init__(self, model: ResNet):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.torso: List[nn.Module] = [
            getattr(model, name)
            for name in model._modules.keys()
            if name.startswith("layer")
        ]
        self.head = nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc,
        )

    def forward(self, x: torch.Tensor) -> VariableLengthClassifierOutput:
        layer_outputs = []
        layer_predictions = []
        x = self.feature_extractor(x)

        for layer in self.torso:
            if layer[0].downsample:
                layer_predictions = [
                    layer[0].downsample(x)
                    for x in layer_predictions
                ]
            for sublayer in layer:
                x = sublayer(x)
                layer_outputs.append(x)
                layer_predictions.append(x)
        layer_predictions = [
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
    
    @staticmethod
    def from_pretrained(model_name: str) -> ReducedLengthModelForImageClassification:
        ...

class ReducedLengthBertForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: BertForSequenceClassification, tokenizer: BertTokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> BertTokenizer:
        return self._tokenizer

    @property
    def model(self) -> BertForSequenceClassification:
        return self._model
    
    @property
    def torso(self) -> nn.Module:
        return self.model.bert.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier
    
    @staticmethod
    def from_pretrained(model_name: str) -> ReducedLengthBertForSequenceClassification:
        return ReducedLengthBertForSequenceClassification(BertForSequenceClassification.from_pretrained(model_name), BertTokenizer.from_pretrained(model_name))

class ReducedLengthRobertaForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: RobertaForSequenceClassification, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> RobertaTokenizer:
        return self._tokenizer

    @property
    def model(self) -> RobertaForSequenceClassification:
        return self._model
    
    @property
    def torso(self) -> nn.Module:
        return self.model.roberta.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier
    
    @staticmethod
    def from_pretrained(model_name: str) -> ReducedLengthRobertaForSequenceClassification:
        return ReducedLengthRobertaForSequenceClassification(RobertaForSequenceClassification.from_pretrained(model_name), RobertaTokenizer.from_pretrained(model_name))

class ReducedLengthGPT2ForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: GPT2ForSequenceClassification, tokenizer: GPT2Tokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> GPT2Tokenizer:
        return self._tokenizer

    @property
    def model(self) -> GPT2ForSequenceClassification:
        return self._model

    @property
    def torso(self) -> nn.Module:
        return self.model.transformer.h
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.score
    
    @staticmethod
    def from_pretrained(model_name: str) -> ReducedLengthGPT2ForSequenceClassification:
        return ReducedLengthGPT2ForSequenceClassification(GPT2ForSequenceClassification.from_pretrained(model_name), GPT2Tokenizer.from_pretrained(model_name))