from torch import nn
from LayerNorm import LayerNorm
from SublayerConnection import SublayerConnection
from attention import clones


"""EncoderLayer"""
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size: 词嵌入维度
        # self_attn: 自注意力机制类的实例化
        # feed_forward: 前馈全连接层类的实例化
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # lambda x: 表示一个需要输入x的函数，这个函数作为参数传入sublayer中，等到sublayer调用forward函数并传入参数时会计算出结果。
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


"""Encoder"""
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)