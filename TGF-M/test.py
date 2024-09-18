import torch
import torch.nn as nn
from torch.jit import script
from torch.utils.tensorboard import SummaryWriter
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.utils.features import get_bond_feature_dims
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_acc = EventAccumulator("/media/aita130/AIDD/hw/CoAtGIN-main/model2/log401")
event_acc.Reload()

# 获取所有的标量
scalars = event_acc.Tags()['scalars']

# 读取特定的标量数据，例如 'valid/mae' 和 'train/mae'
valid_mae = event_acc.Scalars('valid/mae')
train_mae = event_acc.Scalars('train/mae')

# # 打印数据
# for scalar in valid_mae:
#     print(f"valid_maeStep: {scalar.step}, Value: {scalar.value}")

# for scalar in train_mae:
#     print(f"train_maeStep: {scalar.step}, Value: {scalar.value}")
# 打印数据
for scalar in valid_mae:
    print(f"{scalar.value}")

# for scalar in train_mae:
#     print(f"{scalar.value}")
# dataset = PygPCQM4Mv2Dataset(root = '/media/aita130/AIDD/hw/CoAtGIN-main/model/dataset')
# split_idx = dataset.get_idx_split()
# split_idx['train'] = split_idx['train'][:33786]
# print(split_idx)

# data, slices = torch.load('/media/aita130/AIDD/hw/CoAtGIN-main/model/dataset/pcqm4m-v2/processed/geometric_data_processed.pt')

# sum_distance = data.sum_distance
# edge_distance = data.edge_distance
# print(sum_distance,sum_distance.shape)
# mean_value = edge_distance.mean()
# print(mean_value)
# print(edge_distance,edge_distance.shape)
# max_value = sum_distance.max()
# non_zero_min = sum_distance[sum_distance > 0].min()
# print(max_value,non_zero_min)
# # 定义数据区间
# bins = [0,120,240]
# non_zero_min = sum_distance[sum_distance == 120]
# print(non_zero_min)
# # 统计各个区间的数据数量
# counts = torch.histc(sum_distance, bins=len(bins)-1, min=0, max=240)
#
# # 将结果转换为 Python 列表
# distribution = counts.tolist()
#
# print(distribution)

# # 假设的 edge_attr 张量
# edge_attr = torch.tensor([
#     [3, 0, 1], [3, 1, 0],[1,0,3]
# ])
#
# # 创建类型字典
# type_dict = {}
# type_id = 0
#
# # 遍历 edge_attr 中的每条边
# for edge in edge_attr:
#     # 将特征转换为字符串标识符
#     type_str = '-'.join(map(str, edge.tolist()))
#
#     # 如果这个类型是新的，则添加到字典中
#     if type_str not in type_dict:
#         type_dict[type_str] = type_id
#         type_id += 1
#
# # 为每条边分配类型编号
# edge_type_ids = torch.tensor([type_dict['-'.join(map(str, edge.tolist()))] for edge in edge_attr])
#
# print(edge_type_ids)
# #
# # # data = torch.load(pt_path)
# # # (Data(edge_index=[2, 109093666], edge_attr=[109093666, 3], x=[52970672, 9], y=[3746620], pos=[52970672, 3]),
# # #  defaultdict(<class 'dict'>, {'edge_index': tensor([        0,        40,        74,  ..., 109093636, 109093644, 109093666]),
# # #                             'edge_attr': tensor([        0,        40,        74,  ..., 109093636, 109093644,        109093666]),
# # #                             'x': tensor([       0,       18,       35,  ..., 52970656, 52970661, 52970672]),
# # #                             'y': tensor([      0,       1,       2,  ..., 3746618, 3746619, 3746620]),
# # #                             'pos': tensor([       0,       18,       35,  ..., 52970656, 52970661, 52970672])}))
# # # Number of molecules: 3378606/3746620
# # # Total number of atoms: 47711654/52970672
# #
# # data.edge_types = edge_type_ids
# # # 更新slices对象
# # slices['edge_types'] = slices['edge_index']
# # print("save...")
# # # 写入.pt文件
# # torch.save((data, slices), '/media/aita130/AIDD/hw/CoAtGIN-main/model/dataset/pcqm4m-v2/processed/geometric_data_processed.pt')
#
# print("data:",data)
# print("slices:",slices)
# edge_types = data.edge_types
# print(edge_types)
# print(edge_types.max())
