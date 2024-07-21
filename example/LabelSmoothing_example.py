import torch
from matplotlib import pyplot as plt
from pyitcast.transformer_utils import LabelSmoothing

# size: 目标数据的词汇总数，也是模型最后一层得到张量最后一维大小
# padding_idx: 表示要将哪些tensor中的数字替换成0
# smoothing: 表示标签平滑程度。如果原来的标签表示值为l, 则平滑后的值域变为 [1 - smoothing, 1 + smoothing]
crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

# 假定一个模型最后输出的预测结果为
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0]])
# 标签的表示值为0, 1, 2
target = torch.LongTensor([2, 1, 0])
# 使用标签平滑
result = crit(predict, target)
print(result)
# 绘制标签平滑图
plt.imshow(crit.true_dist)
plt.show()