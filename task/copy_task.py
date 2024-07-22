# 针对transformer模型的优化器
import torch
from pyitcast.transformer_utils import get_std_opt, run_epoch, Batch, greedy_decode
# 导入标签平滑工具包
from pyitcast.transformer_utils import LabelSmoothing
# 导入损失计算工具包
from pyitcast.transformer_utils import SimpleLossCompute
from input import subsequent_mask
from make_model import make_model


"""构建数据生成器"""
def data_generator(V, batch, num_batch):
    # V: 随机生成数字最大值
    # batch: batch_size
    # num_batch: 一轮训练有多少个batch
    for i in range(num_batch):
        data = torch.randint(1, V, size=(batch, 10))
        # 使数据第一列数字为1，这就成了起始标志
        # 当解码器进行第一次解码的时候，会使用起始标志作为输入
        data[:, 0] = 1
        # 源数据和目标数据是一样的
        source = target = data
        source.requires_grad_(False)
        target.requires_grad_(False)
        yield Batch(source, target)



V = 11
batch_size = 8
num_batch = 30
max_len = 10
start_symbol = 1
epochs = 10


model = make_model(V, V, N=2)
# 优化器
model_optimizer = get_std_opt(model)
# 使用LabelSmoothing获得标签平滑对象
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# 损失函数：使用标签平滑结果
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)



def run(model, loss, epochs):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, batch_size, num_batch), model, loss)
        model.eval()
        run_epoch(data_generator(V, batch_size, num_batch), model, loss)

    model.eval()
    source = torch.LongTensor([1, 3, 2, 5, 4, 6, 7, 8, 9, 10])
    source_mask = torch.ones(1, 1, max_len)

    # 使用贪婪编码，找到每个单词最可能的结果
    result = greedy_decode(model, source, source_mask, max_len=max_len, start_symbol=start_symbol)
    print(result)


if __name__ == '__main__':
    run(model, loss, epochs)