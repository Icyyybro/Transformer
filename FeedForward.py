import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


"""前馈全连接层"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        # 一共有两个线性层，d_ff为第一个线性层输出维度
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.w1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x
