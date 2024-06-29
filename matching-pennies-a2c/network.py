import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.logits = nn.Parameter(torch.ones(output_size) / output_size)
        self.value = nn.Parameter(torch.tensor([0.0]))

    def forward(self):
        return self.value, nn.Softmax(dim = -1)(self.logits)
