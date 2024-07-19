import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

from attention import attention
from input import Embeddings, subsequent_mask

"""test embedding"""
d_model = 512
vocab = 1000

# 创建一个测试文本，此文本共有2句话，每句话4个词，下面x即为词文本的数字化表示（即一个词对应一个数字）
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
emb_result = emb(x)

"""test attention"""
query = key = value = emb_result
print(x.size())
mask = subsequent_mask(x.size(1))
# mask = torch.zeros(2, 4, 4)
att_result = attention(query, key, value, mask)
print(att_result)