import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


# 文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model: 词嵌入维度
        # vocab:词表大小
        super(Embeddings, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: 输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)





"""位置编码"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置矩阵 max_len表示一个句子最多的单词数，d_model表示每个单词的词嵌入维度
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，词汇的绝对位置就是用它的索引去表示
        # 变成max_len X 1的矩阵, torch.arange的作用是
        position = torch.arange(0, max_len).unsqueeze(1)
        # 将位置信息加入矩阵中，这就要求位置矩阵pe和绝对位置矩阵大小都为max_len X d_model，
        # 要达到这样的变换，就需要1Xd_model的变换矩阵div_term，并将这个矩阵的数字缩放足够小，便于梯度下降
        # 初始化两次，依次初始化奇数位置和偶数位置
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 在0维度扩展，变成batch_size * 句子长度 * d_model
        pe = pe.unsqueeze(0)
        pe = pe.requires_grad_(False)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 文本序列词嵌入表示
        # pe的编码太长了， 将第二个维度，也就是maxlen的维度，缩小成句子长度
        pos = self.pe[:, :x.size(-2)]
        x = x + pos
        return x


# 掩码张量函数
def subsequent_mask(size):
    # size: 代表掩码张量后两个维度，形成一个方阵
    attn_shape = (1, size, size)
    # 使用np.ones()先构建一个全1的张量，然后利用np.triu()形成上三角函数
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 反转上三角矩阵，并将numpy转换为tensor
    return torch.from_numpy(1 - subsequent_mask)



