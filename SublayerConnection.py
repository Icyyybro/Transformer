from torch import nn
from LayerNorm import LayerNorm


"""残差链接"""
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size为词嵌入维度
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # sublayer为跳跃的层
        return x + self.dropout(sublayer(self.norm(x)))
