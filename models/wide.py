import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Wide(nn.Module):
    """non-linearities for sparse feature.
    input_dim: int
        if the wide model receives 2 features with 5 individual values each,
        `input_dim = 10`
    output_dim: int, default = 1
        size of the ouput tensor connection with final output.
    """

    def __init__(self, input_dim=130, output_dim=64):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.wide_linear = nn.Embedding(input_dim, output_dim, padding_idx=0)
        # (Sum(Embedding) + bias) is equivalent to (OneHotVector + Linear)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        """initialize Embedding and bias like nn.Linear. See [original
        implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear).
        """
        nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        out = self.wide_linear(X.long()).sum(dim=1) + self.bias
        return out
