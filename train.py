# 针对transformer模型的优化器
from pyitcast.transformer_utils import get_std_opt
# 导入标签平滑工具包
from pyitcast.transformer_utils import LabelSmoothing
# 导入损失计算工具包
from pyitcast.transformer_utils import SimpleLossCompute

from make_model import make_model

V = 11
batch_size = 20
num_batch = 30

model = make_model(V, V, N=2)
# 优化器
model_optimizer = get_std_opt(model)
# 使用LabelSmoothing获得标签平滑对象
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
# 损失函数：使用标签平滑结果
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)