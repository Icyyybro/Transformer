import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        # vocab_size表示词表大小
        super(Generator, self).__init__()
        # 将batch_size * len * d_model 变成 batch_size * len * vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)