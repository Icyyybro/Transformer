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
        self.dropout = dropout
    def forward(self):
