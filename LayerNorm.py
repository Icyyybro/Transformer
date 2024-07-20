import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features为特征维度，Transformer中指d_model
        # eps为了防止分母为0
        super(LayerNorm, self).__init__()

        # 设置两个可训练的变量a2和b2，其实就是归一化后需要缩放和偏移的参数
        # 这两个参数是可训练的
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 这里的乘法是两个相同形状的tensor的对应位置的数做乘法，和matmul矩阵乘法有区别
        return self.a2 * (x - mean) / (std + self.eps) + self.b2