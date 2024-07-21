import torch
import torch.nn.functional as F
from torch import nn
import math
import copy

from Decoder import Decoder, DecoderLayer
from Encoder import Encoder, EncoderLayer
from EncoderDecoder import EncoderDecoder
from FeedForward import PositionwiseFeedForward
from Generator import Generator
from attention import MultiHeadedAttention
from input import PositionalEncoding, Embeddings


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # c(attn)表示深度拷贝attn得到的相同的attn。相当于x = c(attn), 并将x作为参数传入
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
                           nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
                           Generator(d_model, target_vocab))

    # 模型构建完成后，需要初始化模型的参数
    # 这里使用的方法为：如果模型参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model