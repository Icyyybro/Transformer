import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


def attention(query, key, value, mask=None, dropout=None):
    """注意力机制"""
    # query, key, value 形状为batch_size * len * d_model
    # 取得d_model，就是query最后一个维度
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))

    # 判断是否使用张量
    if mask is not None:
        # 使用masked_fill填充，将掩码张量和scores张量的每个位置比较，如果掩码张量的位置为0，则用一个非常小的数字替代
        # 这样分数会很小，无法选中
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores最后一维做softmax操作，即按列进行softmax，变成概率
    # 因为查询向量query每行代表一个字，去和key.transpose(-2, -1)得到一行，表示这个字与句子中所有字的关联大小
    # 那么按列进行softmax即：将每一行进行softmax，表示这个字和所有字的关联系数
    p_attn = F.softmax(scores, dim=-1)

    # 使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


"""多头注意力机制"""
# 本部分需要用到深度拷贝包copy，之前的复制A操作仅仅是新建了一个指针B指向了A，在内存中还是只有一份，而深度拷贝可以复制为两份互不干扰的A和B

# 首先定义克隆函数clones，其中有两个参数。
# module表明需要拷贝的神经层，N表示拷贝的份数
def clones(module, N):
    """将module深度拷贝N份放入ModuleList中"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 因为要均匀分配，所以先做一个诊断
        assert embedding_dim % head == 0
        # 得到每个头的词特征维度的数量
        self.d_k = embedding_dim // head
        self.head = head
        # 因为多头注意力一共有4个不相关的线形层，分别为K,Q,V的线性层和拼接之后的线性层，所以复制5次。
        # 因为线性层之后不改变tensor形状，所以输入和输出都是词嵌入维度
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        # 如果存在mask，需要将mask扩展维度，代表多头的第n个头，需要在batch_size后面添加维度
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 首先利用zip将输入的QKV和linear组合到一起，在分别取出
        # 使用view方法将tensor的d_model拆分为头数head和每个头分到的特征维度d_k，-1代表每个句子的单词数量
        # 因为attention使用的是倒数第1个维度和倒数第2个维度，且倒数第一个维度需要表示d_model,倒数第2个维度需要表示单词数量
        # 所以将第1维度和第2维度调换位置，目的是使单词数和每个头的特征维度更靠近，这样调用attention时才会找到单词之间的关系
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 因为此时得到的x是4维的，所以需要合并head维度和d_k维度
        # 因为这里需要先进行transpose方法，再进行view方法，所以中间需要插入contiguous，否则transpose后无法进行view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 拼接后再进行一次线性层
        return self.linears[-1](x)
