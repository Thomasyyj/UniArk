import torch
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import (
    RobertaOutput,
    RobertaSelfOutput,
)

from transformers.models.bert.modeling_bert import (
    BertOutput,
    BertSelfOutput,
)


def create_new_forward(module):
    def new_forward(
        hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = module.dense(hidden_states)
        hidden_states = module.adapter(hidden_states)
        hidden_states = module.dropout(hidden_states)
        hidden_states = module.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    return new_forward


class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0, std=0.01)
                nn.init.constant(m.bias, 0)


    def forward(self, inputs):
        return self.fc(inputs) + inputs


def adapter(model, adapter_dim, device):
    model_modules = list(model.named_modules())

    with torch.no_grad():
        for name, layer in model_modules:
            if isinstance(
                layer,
                (RobertaSelfOutput, RobertaOutput),
            ):
                adapter = Adapter(layer.dense.weight.size(0), adapter_dim).to(device)
                layer.register_module("adapter", adapter)
                layer.forward = create_new_forward(layer)
            elif isinstance(
                layer,
                (BertSelfOutput, BertOutput),
            ):
                adapter = Adapter(layer.dense.weight.size(0), adapter_dim).to(device)
                layer.register_module("adapter", adapter)
                layer.forward = create_new_forward(layer)
