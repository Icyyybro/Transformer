from torch import nn
from LayerNorm import LayerNorm
from SublayerConnection import SublayerConnection
from attention import clones, attention

"""解码器层"""
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.attn = self_attn
        # 注意这里的src_attn不是自注意力了，Key和Value向量是编码器传过来的
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # memory为编码器的输出，大小为batch_size * len * d_model
        m = memory
        # 第一层的注意力机制为自注意力机制，target_mask作用是遮掩后面的信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二层注意力使用的是常规注意力机制，K和V是编码器输出的memory，Q是上一层注意力的传来的，source_mask是将不需要用到的信息遮掩到
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)


"""解码器"""
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)