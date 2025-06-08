# # client_fedssa.py
# import copy
# from collections import defaultdict
#
# import numpy as np
# import torch
# from tqdm import tqdm
# from src.client_base import ClientBase
# from torch import nn
#
# class ClientFedPSYSSALong(ClientBase):
#     def __init__(self, args, client_id, trainset, testset, statistic):
#         super().__init__(args, client_id, trainset, testset, statistic)
#
#         self.private_keys, self.shared_keys = self.get_parameter_keys()
#
#         # 初始化优化器
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=self.learning_rate,
#             momentum=0.9
#             # weight_decay=args.inner_wd
#         )
#
#         # 客户端特有属性
#         self.owned_classes = self.get_owned_classes()
#         # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         # self.alpha.data.fill_(0.5)
#         self.globalHead =copy.deepcopy(args.head).to(self.device)
#         self.decay_rounds=2
#         self.miu_0 = 0.5 # 0.1,0.3,0.5,0.7,0.9,1.0
#
#         # self.local_protos = {}  # dict
#         # self.global_protos = None  # list
#
#         # self.proto_loss = nn.MSELoss()
#         # self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=0.9)
#         self.accuracyClass = [0] * self.num_classes
#         self.cdist = self.dict_to_vector()
#
#         # self.class_weights = self._calculate_weights_from_statistic()
#         # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
#         self.loss_fn = nn.CrossEntropyLoss(weight=self.cdist)
#
#
#     def _calculate_weights_from_statistic(self):
#         """从 self.statistic 计算类别权重"""
#         # 将字典转换为按类别顺序排列的张量
#         class_counts = torch.zeros(self.num_classes, device=self.device)
#         for class_id, count in self.statistic.items():
#             class_counts[int(class_id)] = count
#
#         # 计算权重（样本数越少，权重越大）
#         weights = 1.0 / (class_counts + 1e-8)  # 加1e-8防止除零
#         # 归一化权重（保持均值不变）
#         weights = weights / weights.mean()
#         return weights.to(self.device)
#
#     def get_parameter_keys(self):
#         """分离共享参数和私有参数"""
#         # all_keys = list(self.model.state_dict().keys())
#         # private_keys = all_keys[:-2]  # 假设最后两层是分类头
#         # shared_keys = all_keys[-2:]
#         # all_keys = list(self.model.state_dict().keys())
#         private_keys = list(self.model.base.state_dict().keys()) # 假设最后两层是分类头
#         shared_keys = list(self.model.head.state_dict().keys())
#         return private_keys, shared_keys
#
#     def get_owned_classes(self):
#         """获取客户端拥有的类别"""
#         owned = set()
#         for _, y in self.get_train_loader():
#             owned.update(y.unique().tolist())
#         return list(owned)
#
#     # def aggregate(self, raw_feats):
#     #     for label in raw_feats.keys():
#     #         feats = torch.stack(raw_feats[label], dim=0)
#     #         self.local_protos[label] = torch.mean(feats, dim=0)
#
#     # def collect_feats(self):
#     #     train_loader = self.get_train_loader()
#     #     self.model.eval()
#     #
#     #     raw_feats = defaultdict(list)
#     #     with torch.inference_mode():
#     #         for x, y in train_loader:
#     #             x, y = x.to(self.device), y.to(self.device)
#     #             feature = self.model.base(x)
#     #
#     #             for label in torch.unique(y):
#     #                 protos = feature[label == y].detach()
#     #                 raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
#     #
#     #     return raw_feats
#
#     def dict_to_vector(self):
#         # 初始化全零向量
#         vector = [0] * self.num_classes
#
#         # 遍历字典，填充样本数
#         for key, count in self.statistic.items():
#             idx = int(key)  # 将字符串键转为整数索引
#             if idx < self.num_classes:
#                 vector[idx] = count
#             else:
#                 # 可选：处理超出范围的类别（如忽略或警告）
#                 pass
#         dist = torch.tensor(vector, device=self.device)
#
#         nonzero_mask = dist > 0
#         nonzero_dist = dist[nonzero_mask]
#         # rs = max(nonzero_dist)/min(nonzero_dist)  #计算类不平衡程度的指数
#         dist = dist/dist.sum()
#         nonzero_dist = nonzero_dist/nonzero_dist.sum()
#
#         entropy = -torch.sum(nonzero_dist * torch.log(nonzero_dist + 1e-9))
#         rs_alpha = torch.sigmoid(entropy)
#         alpha = torch.exp(-0.5 * entropy)
#         print(entropy,rs_alpha)
#         print("反：",entropy,alpha)
#
#         max_entropy = torch.log(torch.tensor(self.num_classes, device=self.device))
#         normalized_entropy = entropy / max_entropy  # 完全平衡时为1，最不平衡时为0
#         print("normalized_entropy",normalized_entropy)
#         # print("如果是除max，",normalized_entropy,torch.sigmoid(normalized_entropy))
#         # print("max-min",max_entropy- entropy)
#
#
#
#         # rs_alpha = 0.8
#         # rs_alpha = normalized_entropy
#         # print("max_entropy",max_entropy,"entropy",entropy)
#         cdist = dist / dist.max()
#         cdist = cdist.to(self.device)
#         # cdist = cdist * (1.0 - rs_alpha) + rs_alpha
#         cdist = cdist * (1.0 - alpha) + alpha
#         # cdist = cdist * (max_entropy- entropy) + max_entropy
#         # cdist = cdist.reshape((1, -1))
#         return cdist
#
#     def train(self):
#         """本地训练过程"""
#         self.model.train()
#         train_loader = self.get_train_loader()
#
#         # self.fixBase()
#
#         # 动态调整alpha
#         # self.adjust_alpha()
#
#         self.alignModels(self.globalHead)
#         # self.fuse_global_paramsWithClassImbalanced()
#
#         # 融合全局参数
#         # self.fuse_global_params()
#
#
#         # for epoch in range(self.local_epochs):
#         #     local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
#         #     for x, y in local_iter:
#         #         x, y = x.to(self.device), y.to(self.device)
#         #         self.optimizer.zero_grad()
#         #         # feature = self.model.base(x)
#         #         # output = self.model.head(feature)
#         #         # loss = self.loss(output, y)
#         #
#         #         loss = self.loss(self.model(x), y)
#         #         # if self.global_protos is not None:
#         #         #     global_protos = torch.stack(self.global_protos, dim=0)
#         #         #     loss += self.proto_loss(feature, global_protos[y])
#         #         loss.backward()
#         #         self.optimizer.step()
#         #
#         #         local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
#         #
#         #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
#         #         self.optimizer.step()
#
#         # 初始化类别的正确预测数量和总数量
#         correct_per_class = [0] * self.num_classes
#         total_per_class = [0] * self.num_classes
#
#         for epoch in range(self.local_epochs):
#             local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
#             for x, y in local_iter:
#                 x, y = x.to(self.device), y.to(self.device)
#                 self.optimizer.zero_grad()
#
#                 # hs = self.model.base(x)
#                 # ws = self.model.head.weight
#                 # hs, ws = hs.to(self.device), ws.to(self.device)
#                 #
#                 #
#                 # logits = self.cdist * hs.mm(ws.transpose(0, 1))
#                 # loss= self.loss(logits, y)
#
#                 # 计算模型输出和损失
#                 output = self.model(x)
#                 # loss1 = self.loss(output, y)
#                 loss = self.loss_fn(output, y)
#
#                 loss.backward()
#                 self.optimizer.step()
#
#                 # 更新类别正确预测数量
#                 _, predicted = torch.max(output, 1)
#                 for label, pred in zip(y, predicted):
#                     total_per_class[label.item()] += 1
#                     if label.item() == pred.item():
#                         correct_per_class[label.item()] += 1
#
#                 # 显示训练进度
#                 local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
#
#             # 每个 epoch 结束后打印每个类别的准确率
#             for class_id in range(self.num_classes):
#                 accuracy = correct_per_class[class_id] / total_per_class[class_id] if total_per_class[class_id] > 0 else 0
#                 self.accuracyClass[class_id] =accuracy
#             # print(self.accuracyClass)
#
#         # raw_feats = self.collect_feats()
#         # self.aggregate(raw_feats)
#
#     def adjust_alpha(self):
#         """动态调整参数融合系数"""
#         if self.args.current_round <= self.decay_rounds:
#             decay = torch.cos(torch.tensor(self.args.current_round * torch.pi /
#                                            (self.decay_rounds * 2)))
#             self.alpha.data.fill_(self.miu_0 * decay.item())
#         else:
#             self.alpha.data.fill_(0.0)
#
#     def fuse_global_params(self):
#         """融合全局共享参数"""
#         global_params = self.globalHead.state_dict()
#         # local_params = self.model.head.state_dict()
#         # global_params = self.server.shared_params
#         local_params = self.model.head.state_dict()
#         self.alpha = self.alpha.to(self.device)
#         with torch.no_grad():
#             # for key, param in self.globalHead.named_parameters():
#             for key in self.shared_keys:
#                 for cls in self.owned_classes:
#                     # 参数融合公式
#                     local_params[key][cls] = (
#                             local_params[key][cls] * self.alpha +
#                             global_params[key][cls] * (1 - self.alpha)
#                     )
#         self.model.head.load_state_dict(local_params)
#
#     def fuse_global_paramsWithClassImbalanced(self):
#         """融合全局共享参数"""
#         global_params = self.globalHead.state_dict()
#         # local_params = self.model.head.state_dict()
#         # global_params = self.server.shared_params
#         local_params = self.model.head.state_dict()
#         # self.alpha = self.alpha.to(self.device)
#         with torch.no_grad():
#             # for key, param in self.globalHead.named_parameters():
#             for key in self.shared_keys:
#                 for cls in self.owned_classes:
#                     alpha =self.statistic[str(cls)]/self.sample_size
#
#                     # 参数融合公式
#                     local_params[key][cls] = (
#                             local_params[key][cls] * alpha +
#                             global_params[key][cls] * (1 - alpha)
#                     )
#         self.model.head.load_state_dict(local_params)
#
#
#     # def fixBase(self):
#     #     for param in self.model.base.parameters():
#     #         param.requires_grad = False
#     #     for param in self.model.head.parameters():
#     #         param.requires_grad = True
#     #     train_loader = self.get_train_loader()
#     #     for epoch in range(2):
#     #         p_local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
#     #         for x, y in p_local_iter:
#     #             # for i, (x, y) in enumerate(train_loader):
#     #             if type(x) == type([]):
#     #                 x[0] = x[0].to(self.device)
#     #             else:
#     #                 x = x.to(self.device)
#     #             y = y.to(self.device)
#     #             output = self.model(x)
#     #             loss = self.loss(output, y)
#     #             self.optimizer_per.zero_grad()
#     #             loss.backward()
#     #             self.optimizer_per.step()
#     #             p_local_iter.set_description(f"client {self.client_id} P_local_epoch: {epoch} loss: {loss.item():.4f}")
#     #
#     #     max_local_epochs = self.local_epochs
#     #     for param in self.model.base.parameters():
#     #         param.requires_grad = True
#
#     # plan A
#     def alignModels(self,global_head):
#         # Get class-specific prototypes from the local model
#         local_prototypes = [[] for _ in range(self.num_classes)]
#         train_loader = self.get_train_loader()
#
#         # print(f'client{id}')
#         for x_batch, y_batch in train_loader:
#             x_batch = x_batch.to(self.device)
#             y_batch = y_batch.to(self.device)
#
#             with torch.no_grad():
#                 proto_batch = self.model.base(x_batch)
#                 # proto_batch = self.model.projection(proto_batch)
#
#             # Scatter the prototypes based on their labels
#             for proto, y in zip(proto_batch, y_batch):
#                 local_prototypes[y.item()].append(proto)
#
#         mean_prototypes = []
#
#         # print(f'client{self.id}')
#         for class_prototypes in local_prototypes:
#
#             if not class_prototypes == []:
#                 # Stack the tensors for the current class
#                 stacked_protos = torch.stack(class_prototypes)
#
#                 # Compute the mean tensor for the current class
#                 mean_proto = torch.mean(stacked_protos, dim=0)
#                 mean_prototypes.append(mean_proto)
#             else:
#                 mean_prototypes.append(None)
#
#         mean_prototypes_tensor = torch.stack([proto for proto in mean_prototypes if proto is not None])
#         # mean_prototypes_tensor = torch.stack(mean_prototypes)
#         # mean prototypes经过本地head，得到的
#         alignment_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
#         # alignment_loss_fn = torch.nn.MSELoss()
#
#         alignment_optimizer = torch.optim.SGD(global_head.parameters(),
#                                               lr=0.01)  # Adjust learning rate and optimizer as needed
#         # alignment_loss_fn = torch.nn.MSELoss()
#         # protoLocalLogits = self.model.head(mean_prototypes_tensor)
#         # proroGlobalLogits = global_head(mean_prototypes_tensor)
#         #
#         # loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
#         #                          torch.nn.functional.softmax(proroGlobalLogits, dim=1))
#         #
#         # alignment_optimizer.zero_grad()
#         # loss.backward()
#         # alignment_optimizer.step()
#         for _ in range(1):  # Iterate for 1 epochs; adjust as needed
#             protoLocalLogits = self.model.head(mean_prototypes_tensor)
#             proroGlobalLogits = global_head(mean_prototypes_tensor)
#
#             loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
#                                      torch.nn.functional.softmax(proroGlobalLogits, dim=1))
#
#
#             alignment_optimizer.zero_grad()
#             loss.backward()
#             alignment_optimizer.step()
#
#         # Substitute the parameters of the base, enabling personalization
#         for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
#             old_param.data = new_param.data.clone()
#         # self.globalHead = global_head
#     # plan B（样本）
#     # def alignModels(self, global_head):
#     #     local_prototypes = [[] for _ in range(self.num_classes)]
#     #     train_loader = self.get_train_loader()
#     #
#     #     # print(f'client{id}')
#     #     for x_batch, y_batch in train_loader:
#     #         x_batch = x_batch.to(self.device)
#     #         y_batch = y_batch.to(self.device)
#     #
#     #         with torch.no_grad():
#     #             proto_batch = self.model.base(x_batch)
#     #
#     #         # Scatter the prototypes based on their labels
#     #         for proto, y in zip(proto_batch, y_batch):
#     #             local_prototypes[y.item()].append(proto)
#     #
#     #     mean_prototypes = []
#     #
#     #     # print(f'client{self.id}')
#     #     for class_prototypes in local_prototypes:
#     #
#     #         if not class_prototypes == []:
#     #             # Stack the tensors for the current class
#     #             stacked_protos = torch.stack(class_prototypes)
#     #
#     #             # Compute the mean tensor for the current class
#     #             mean_proto = torch.mean(stacked_protos, dim=0)
#     #             mean_prototypes.append(mean_proto)
#     #         else:
#     #             mean_prototypes.append(None)
#     #
#     #     mean_prototypes_tensor = torch.stack([proto for proto in mean_prototypes if proto is not None])
#     #
#     #     # local_logits = self.model.head(mean_prototypes_tensor)
#     #     # global_logits = global_head(mean_prototypes_tensor)
#     #     # local_probs = torch.softmax(local_logits, dim=1)
#     #     # global_probs = torch.softmax(global_logits, dim=1)
#     #
#     #
#     #     global_params = global_head.state_dict()
#     #     local_params = self.model.head.state_dict()
#     #     # localWeight = local_probs/(local_probs+global_probs)
#     #     # globalWeight = global_probs/(local_probs+global_probs)
#     #
#     #     with torch.no_grad():
#     #         # 针对每个类别分别进行处理
#     #         for cls in self.owned_classes:
#     #             # 当前类别的 mean prototype
#     #             mean_proto = mean_prototypes[cls]
#     #
#     #             # 如果当前类别有原型，则继续计算
#     #             if mean_proto is not None:
#     #                 # 获取该类别在局部和全局模型中的 logits
#     #                 local_logits = self.model.head(mean_proto.unsqueeze(0))  # 维度：1xfeature_dim
#     #                 global_logits = global_head(mean_proto.unsqueeze(0))
#     #
#     #                 # Softmax 计算该类别的概率
#     #                 local_probs = torch.softmax(local_logits, dim=1)
#     #                 global_probs = torch.softmax(global_logits, dim=1)
#     #
#     #                 # 计算局部和全局模型的权重
#     #                 local_weight = local_probs[:, cls] / (local_probs[:, cls] + global_probs[:, cls])
#     #                 global_weight = global_probs[:, cls] / (local_probs[:, cls] + global_probs[:, cls])
#     #
#     #                 # 更新该类别的参数
#     #                 for key in self.shared_keys:
#     #                     local_params[key][cls] = (
#     #                             local_params[key][cls] * local_weight +
#     #                             global_params[key][cls] * global_weight
#     #                     )
#     #
#     #         # 更新模型的 head 参数
#     #         self.model.head.load_state_dict(local_params)
#     #     # with torch.no_grad():
#     #     #     # for key, param in self.globalHead.named_parameters():
#     #     #     for key in self.shared_keys:
#     #     #         for cls in self.owned_classes:
#     #     #             # # 参数融合公式
#     #     #             # local_params[key][cls] = (
#     #     #             #         local_params[key][cls] * localWeight[cls] +
#     #     #             #         global_params[key][cls] * globalWeight[cls]
#     #     #             # )
#     #     #             local_param = local_params[key]
#     #     #             global_param = global_params[key]
#     #     #
#     #     #             local_weight_for_cls = localWeight[:, cls]  # 每个 batch 对应该类的局部权重
#     #     #             global_weight_for_cls = globalWeight[:, cls]  # 每个 batch 对应该类的全局权重
#     #     #
#     #     #             # 按 batch 和类加权
#     #     #             weighted_local_param = local_param * local_weight_for_cls.sum()
#     #     #             weighted_global_param = global_param * global_weight_for_cls.sum()
#     #     #
#     #     #             # 融合局部和全局参数
#     #     #             local_params[key] = weighted_local_param + weighted_global_param
#     #     # self.model.head.load_state_dict(local_params)
#
#     # plan D 按类别
#     # # 类别权重加权
#     # def alignModels(self,global_head):
#     #     # Get class-specific prototypes from the local model
#     #     local_prototypes = [[] for _ in range(self.num_classes)]
#     #     train_loader = self.get_train_loader()
#     #
#     #     # print(f'client{id}')
#     #     for x_batch, y_batch in train_loader:
#     #         x_batch = x_batch.to(self.device)
#     #         y_batch = y_batch.to(self.device)
#     #
#     #         with torch.no_grad():
#     #             proto_batch = self.model.base(x_batch)
#     #
#     #         # Scatter the prototypes based on their labels
#     #         for proto, y in zip(proto_batch, y_batch):
#     #             local_prototypes[y.item()].append(proto)
#     #
#     #     mean_prototypes = []
#     #
#     #     # print(f'client{self.id}')
#     #     for class_prototypes in local_prototypes:
#     #
#     #         if not class_prototypes == []:
#     #             # Stack the tensors for the current class
#     #             stacked_protos = torch.stack(class_prototypes)
#     #
#     #             # Compute the mean tensor for the current class
#     #             mean_proto = torch.mean(stacked_protos, dim=0)
#     #             mean_prototypes.append(mean_proto)
#     #         else:
#     #             mean_prototypes.append(None)
#     #
#     #     alignment_optimizer = torch.optim.SGD(global_head.parameters(),
#     #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
#     #     alignment_loss_fn = torch.nn.MSELoss()
#     #
#     #     # 初始化每个类别的准确率统计
#     #     num_classes = len(mean_prototypes)  # 假设 mean_prototypes 是一个列表，长度为类别数量
#     #     class_accuracies = [1.0] * num_classes  # 初始化每个类别的准确率为 1.0
#     #     class_counts = [0] * num_classes  # 记录每个类别的样本数
#     #
#     #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
#     #         for x_batch, y_batch in train_loader:
#     #             x_batch = x_batch.to(self.device)
#     #             y_batch = y_batch.to(self.device)
#     #             global_proto_batch = self.model.base(x_batch)
#     #
#     #             # 计算每个类别的预测结果
#     #             outputs = global_head(global_proto_batch)  # 假设 global_proto_batch 是模型的输出 logits
#     #             preds = torch.argmax(outputs, dim=1)
#     #             correct = (preds == y_batch).float()  # 正确预测的样本
#     #
#     #             # 更新每个类别的准确率
#     #             for label in y_batch.unique():
#     #                 label = label.item()
#     #                 mask = (y_batch == label)
#     #                 class_counts[label] += mask.sum().item()
#     #                 class_accuracies[label] = correct[mask].sum().item() / mask.sum().item()  # 更新准确率
#     #
#     #             # print("class_accuracies",class_accuracies)
#     #             # 计算加权损失
#     #             loss = 0
#     #             for label in y_batch.unique():
#     #                 label = label.item()
#     #                 if mean_prototypes[label] is not None:
#     #                     # 使用准确率作为权重（准确率越低，权重越大）
#     #                     # weight =math.exp(1.0-class_accuracies[label])
#     #                     weight =1.0-class_accuracies[label]
#     #                     # print("weight:",weight)# 准确率越低，权重越大
#     #                     loss += weight * alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label])
#     #
#     #             # 反向传播和优化
#     #             alignment_optimizer.zero_grad()
#     #             loss.backward()
#     #             alignment_optimizer.step()
#     #
#     #     # Substitute the parameters of the base, enabling personalization
#     #     for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
#     #         old_param.data = new_param.data.clone()
# client_fedssa.py
import copy
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.client_base import ClientBase
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import rcParams


class ClientFedPSYSSALong(ClientBase):
    def __init__(self, args, client_id, trainset, testset, statistic):
        super().__init__(args, client_id, trainset, testset, statistic)

        # self.loss = None
        self.upload_data = None
        self.class_weights = None
        self.private_keys, self.shared_keys = self.get_parameter_keys()

        # 初始化优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
            # weight_decay=args.inner_wd
        )

        # 客户端特有属性
        self.owned_classes = self.get_owned_classes()
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.alpha.data.fill_(0.5)
        self.globalHead = copy.deepcopy(args.head).to(self.device)
        self.globalEpochRound = 0
        self.proto_weight = {}  # 存储每个类别的置信度分数
        # self.decay_rounds=2
        # self.miu_0 = 0.5 # 0.1,0.3,0.5,0.7,0.9,1.0

        self.local_protos = {}  # dict
        # self.global_protos = None  # list

        # self.proto_loss = nn.MSELoss()
        # self.optimizer_per = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.accuracyClass = [0] * self.num_classes
        # self.cdist = self.dict_to_vector()

        # self.class_weights = self._calculate_weights_from_statistic()
        # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # self.loss_fn = nn.CrossEntropyLoss(weight=self.cdist)

    def _calculate_weights_from_statistic(self):
        """从 self.statistic 计算类别权重，用于处理类别不均衡问题"""
        # 将字典转换为按类别顺序排列的张量
        class_counts = torch.zeros(self.num_classes, device=self.device)
        for class_id, count in self.statistic.items():
            class_counts[int(class_id)] = count

        # 计算权重（样本数越少，权重越大）
        weights = 1.0 / (class_counts + 1e-8)  # 加1e-8防止除零
        # 归一化权重（保持均值不变）
        weights = weights / weights.mean()
        return weights.to(self.device)

    def get_parameter_keys(self):
        """分离共享参数和私有参数"""
        # all_keys = list(self.model.state_dict().keys())
        # private_keys = all_keys[:-2]  # 假设最后两层是分类头
        # shared_keys = all_keys[-2:]
        # all_keys = list(self.model.state_dict().keys())
        private_keys = list(self.model.base.state_dict().keys())  # 假设最后两层是分类头
        shared_keys = list(self.model.head.state_dict().keys())
        return private_keys, shared_keys

    def get_owned_classes(self):
        """获取客户端拥有的类别"""
        owned = set()
        for _, y in self.get_train_loader():
            owned.update(y.unique().tolist())
        return list(owned)

    def get_shared_params(self):
        """获取共享参数"""
        return {k: v for k, v in self.model.state_dict().items() if k.startswith('base.')}

    def get_private_params(self):
        """获取私有参数"""
        return {k: v for k, v in self.model.state_dict().items() if not k.startswith('base.')}

    def set_params(self, params):
        """加载服务器共享参数（忽略私有参数不匹配）"""
        model_state = self.model.state_dict()

        # 1. 分离共享参数（假设共享参数以`base.`为前缀）
        shared_params = {k: v for k, v in params.items() if k.startswith('base.')}
        model_shared_state = {k: v for k, v in model_state.items() if k.startswith('base.')}

        # 2. 检查参数匹配性（仅警告，不中断）
        missing_shared = set(model_shared_state.keys()) - set(shared_params.keys())
        unexpected_shared = set(shared_params.keys()) - set(model_shared_state.keys())
        if missing_shared:
            print(f"客户端 {self.client_id} 缺失共享参数: {missing_shared}")
        if unexpected_shared:
            print(f"客户端 {self.client_id} 意外共享参数: {unexpected_shared}")

        # 3. 加载共享参数（strict=False允许部分加载）
        model_shared_state.update(shared_params)
        self.model.load_state_dict(model_shared_state, strict=False)

    def aggregate(self, raw_feats, raw_scores):
        """
        计算置信度
        :param raw_feats: 原始特征集合，用于构建本地原型
        :param raw_scores: 原始置信度分数集合，用于评估原型的可靠性
        """
        for label, feats in raw_feats.items():
            feats = F.normalize(torch.stack(feats, dim=0))
            score = torch.tensor(raw_scores[label])
            self.local_protos[label] = torch.mean(feats, dim=0)
            # 计算原型的置信度权重
            score_proto = F.softmax(self.model.head(self.local_protos[label].unsqueeze(0)), dim=1)
            # 取原型在对应类别上的预测概率作为置信度
            self.proto_weight[label] = score_proto[0, label].cpu().detach()

    def collect_feats(self):
        train_loader = self.get_train_loader()
        self.model.eval()

        raw_feats = defaultdict(list)
        raw_scores = defaultdict(list)
        with torch.inference_mode():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                feature = self.model.base(x)
                # 模型输出的类别概率
                score = F.softmax(self.model.head(feature), dim=1)

                for label in torch.unique(y):
                    protos = feature[label == y].detach()
                    raw_feats[label.item()].extend(list(torch.unbind(protos, dim=0)))
                    # 收集每个样本在真实类别上的预测概率作为置信度分数
                    raw_scores[label.item()].extend(score[label == y][:, label].detach().cpu().detach())

        return raw_feats, raw_scores
        # for label in raw_feats.keys():
        #     if len(raw_feats[label]) > 0:
        #         proto = torch.mean(torch.stack(raw_feats[label]), dim=0)
        #         score_proto = F.softmax(self.model.head(proto.unsqueeze(0)), dim=1)
        #         self.proto_weights[label] = score_proto[0, label].item()
        # return raw_feats, self.proto_weights

    def dict_to_vector(self):
        # 初始化全零向量
        vector = [0] * self.num_classes

        # 遍历字典，填充样本数
        for key, count in self.statistic.items():
            idx = int(key)  # 将字符串键转为整数索引
            if idx < self.num_classes:
                vector[idx] = count
            else:
                # 可选：处理超出范围的类别（如忽略或警告）
                pass
        dist = torch.tensor(vector, device=self.device)

        nonzero_mask = dist > 0
        nonzero_dist = dist[nonzero_mask]
        # rs = max(nonzero_dist)/min(nonzero_dist)  #计算类不平衡程度的指数
        dist = dist / dist.sum()
        nonzero_dist = nonzero_dist / nonzero_dist.sum()

        dist = dist / dist.sum()

        # entropy = -torch.sum(nonzero_dist * torch.log(nonzero_dist + 1e-9))
        # max_entropy = torch.log(torch.tensor(nonzero_dist.size(0), device=self.device))
        # # print("max_entropy - entropy",max_entropy - entropy)
        # rs_alpha = torch.sigmoid(entropy/max_entropy)+0.25
        # # rs_alpha = torch.exp(-0.5 * entropy)
        # # print(entropy,rs_alpha)
        # print("entropy：",entropy,max_entropy,rs_alpha,rs_alpha+0.25)

        rs_alpha = 0.9
        cdist = dist / dist.max()
        cdist = cdist.to(self.device)
        # cdist = cdist * (max_entropy - entropy) + max_entropy
        cdist = cdist * (1.0 - rs_alpha) + rs_alpha
        # self.cdist = cdist
        cdist = cdist.reshape((1, -1))
        return cdist

    def resample_global_params(self, gama=0.9, noise_std=0.01):
        """
        对全局模型参数添加扰动：w=gama*w+(1-gama)*w'
        :param gama: 融合因子，接近1代表小扰动
        :param noise_std: 噪声标准差
        """
        with torch.no_grad():
            for name, param in self.globalHead.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * noise_std
                    param.data = gama * param.data + (1 - gama) * noise

    def resample_with_class_adaptive_noise(self, min_std=0.005, max_std=0.05):
        """
        对每个类别的参数添加类别自适应扰动，稀有类别扰动更强
        :param min_std: 用于常见类
        :param max_std: 用于稀有类
        """
        global_params = self.globalHead.state_dict()

        # 样本总数和按类别权重归一化
        class_counts = torch.zeros(self.num_classes)
        for cls, count in self.statistic.items():
            class_counts[int(cls)] = count
        norm_counts = class_counts / (class_counts.sum() + 1e-6)
        adaptive_weights = 1.0 - norm_counts  # 稀有类接近1，常见类接近0
        adaptive_weights /= adaptive_weights.max()  # 归一化

        with torch.no_grad():
            for key in self.shared_keys:
                param_tensor = global_params[key]
                for cls in self.owned_classes:
                    cls_weight = adaptive_weights[cls].item()
                    # 自适应噪声插值
                    noise_std = min_std + (max_std - min_std) * cls_weight
                    noise = torch.randn_like(param_tensor[cls]) * noise_std
                    param_tensor[cls] = 0.9 * param_tensor[cls] + 0.1 * noise

        self.globalHead.load_state_dict(global_params)

    def resample_with_entropy_adaptive_noise(self, base_std=0.01, min_std=0.005, max_std=0.05):
        """
        使用预测熵控制扰动强度，熵越高扰动越大
        """
        import torch.nn.functional as F

        global_params = self.globalHead.state_dict()
        logits_dict = {cls: [] for cls in self.owned_classes}

        # 收集每类预测的logits
        self.model.eval()
        loader = self.get_train_loader()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                features = self.model.base(x_batch)
                logits = self.globalHead(features)
                for logit, label in zip(logits, y_batch):
                    logits_dict[label.item()].append(logit)

        # 计算每类平均预测熵
        entropy_dict = {}
        for cls, logit_list in logits_dict.items():
            if not logit_list:
                entropy_dict[cls] = 0.0  # 无数据视为确定
                continue
            stacked_logits = torch.stack(logit_list)
            probs = F.softmax(stacked_logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()
            entropy_dict[cls] = entropy.item()

        # 归一化熵
        entropy_tensor = torch.tensor([entropy_dict.get(cls, 0.0) for cls in self.owned_classes])
        entropy_norm = (entropy_tensor - entropy_tensor.min()) / (entropy_tensor.max() - entropy_tensor.min() + 1e-6)
        entropy_dict_norm = dict(zip(self.owned_classes, entropy_norm.tolist()))

        # 使用归一化熵加权扰动
        with torch.no_grad():
            for key in self.shared_keys:
                param_tensor = global_params[key]
                for cls in self.owned_classes:
                    cls_entropy_weight = entropy_dict_norm[cls]
                    noise_std = min_std + (max_std - min_std) * cls_entropy_weight
                    noise = torch.randn_like(param_tensor[cls]) * noise_std
                    param_tensor[cls] = 0.9 * param_tensor[cls] + 0.1 * noise

        self.globalHead.load_state_dict(global_params)

    def train(self):
        """本地训练过程"""
        self.globalEpochRound += 1
        self.model.train()
        train_loader = self.get_train_loader()

        # 计算类别权重
        # self.class_weights = self._calculate_weights_from_statistic()
        # self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # 模型重采样
        # self.resample_global_params(gama=0.9, noise_std=0.02)
        # self.resample_with_class_adaptive_noise()
        # self.resample_with_entropy_adaptive_noise()

        # self.fixBase()

        # 动态调整alpha
        # self.adjust_alpha()
        # self.alignModels(self.globalHead)
        # self.fuse_global_paramsWithClassImbalanced()

        # 融合全局参数
        # self.fuse_global_params()

        # for epoch in range(self.local_epochs):
        #     local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
        #     for x, y in local_iter:
        #         x, y = x.to(self.device), y.to(self.device)
        #         self.optimizer.zero_grad()
        #         # feature = self.model.base(x)
        #         # output = self.model.head(feature)
        #         # loss = self.loss(output, y)
        #
        #         loss = self.loss(self.model(x), y)
        #         # if self.global_protos is not None:
        #         #     global_protos = torch.stack(self.global_protos, dim=0)
        #         #     loss += self.proto_loss(feature, global_protos[y])
        #         loss.backward()
        #         self.optimizer.step()
        #
        #         local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")
        #
        #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
        #         self.optimizer.step()

        # 初始化类别的正确预测数量和总数量
        # correct_per_class = [0] * self.num_classes
        # total_per_class = [0] * self.num_classes
        # print("before__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")
        for epoch in range(self.local_epochs):
            local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
            for x, y in local_iter:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                #
                # hs = self.model.base(x)
                # ws = self.model.head.weight
                # hs, ws = hs.to(self.device), ws.to(self.device)
                #
                #
                # logits = self.cdist * hs.mm(ws.transpose(0, 1))
                # loss= self.loss(logits, y)
                #
                # # 计算模型输出和损失

                output = self.model(x)
                # loss = self.loss_fn(output, y)
                loss = self.loss(output, y)  # ！！！改进点①！！！

                loss.backward()
                self.optimizer.step()

                # 更新类别正确预测数量
                # _, predicted = torch.max(output, 1)
                # for label, pred in zip(y, predicted):
                #     total_per_class[label.item()] += 1
                #     if label.item() == pred.item():
                #         correct_per_class[label.item()] += 1

                # 显示训练进度
                local_iter.set_description(f"client {self.client_id} epoch: {epoch} loss: {loss.item():.4f}")

                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)

            # 每个 epoch 结束后打印每个类别的准确率
            # for class_id in range(self.num_classes):
            #     accuracy = correct_per_class[class_id] / total_per_class[class_id] if total_per_class[class_id] > 0 else 0
            #     self.accuracyClass[class_id] =accuracy
            # print(self.accuracyClass)

        # raw_feats = self.collect_feats()
        # self.aggregate(raw_feats)
        
        # _, proto_weights = self.collect_feats()
        # self.upload_data = {
        #     "model_state": self.model.state_dict(),
        #     "proto_weights": proto_weights
        # }
        raw_feats, raw_scores = self.collect_feats()
        self.aggregate(raw_feats, raw_scores)
        torch.cuda.empty_cache()
        # print("after__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")

    def adjust_alpha(self):
        """动态调整参数融合系数"""
        if self.args.current_round <= self.decay_rounds:
            decay = torch.cos(torch.tensor(self.args.current_round * torch.pi /
                                           (self.decay_rounds * 2)))
            self.alpha.data.fill_(self.miu_0 * decay.item())
        else:
            self.alpha.data.fill_(0.0)

    def fuse_global_params(self):
        """融合全局共享参数"""
        global_params = self.globalHead.state_dict()
        # local_params = self.model.head.state_dict()
        # global_params = self.server.shared_params
        local_params = self.model.head.state_dict()
        self.alpha = self.alpha.to(self.device)
        with torch.no_grad():
            # for key, param in self.globalHead.named_parameters():
            for key in self.shared_keys:
                for cls in self.owned_classes:
                    # 参数融合公式
                    local_params[key][cls] = (
                            local_params[key][cls] * self.alpha +
                            global_params[key][cls] * (1 - self.alpha)
                    )
        self.model.head.load_state_dict(local_params)

    def fuse_global_paramsWithClassImbalanced(self):
        """融合全局共享参数"""
        global_params = self.globalHead.state_dict()
        # local_params = self.model.head.state_dict()
        # global_params = self.server.shared_params
        local_params = self.model.head.state_dict()
        # self.alpha = self.alpha.to(self.device)
        with torch.no_grad():
            # for key, param in self.globalHead.named_parameters():
            for key in self.shared_keys:
                for cls in self.owned_classes:
                    alpha = self.statistic[str(cls)] / self.sample_size

                    # 参数融合公式
                    local_params[key][cls] = (
                            local_params[key][cls] * alpha +
                            global_params[key][cls] * (1 - alpha)
                    )
        self.model.head.load_state_dict(local_params)

    # def fixBase(self):
    #     for param in self.model.base.parameters():
    #         param.requires_grad = False
    #     for param in self.model.head.parameters():
    #         param.requires_grad = True
    #     train_loader = self.get_train_loader()
    #     for epoch in range(2):
    #         p_local_iter = tqdm(train_loader, desc=f"client {self.client_id} epoch: {epoch}")
    #         for x, y in p_local_iter:
    #             # for i, (x, y) in enumerate(train_loader):
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)
    #             self.optimizer_per.zero_grad()
    #             loss.backward()
    #             self.optimizer_per.step()
    #             p_local_iter.set_description(f"client {self.client_id} P_local_epoch: {epoch} loss: {loss.item():.4f}")
    #
    #     max_local_epochs = self.local_epochs
    #     for param in self.model.base.parameters():
    #         param.requires_grad = True

    # plan A
    def alignModels(self, global_head):
        # self.plot_confusion_matrix(global_head,"Before Align Confusion Matrix",self.globalEpochRound)
        # Get class-specific prototypes from the local model
        # 构造每个类别的局部特征原型
        local_prototypes = [[] for _ in range(self.num_classes)]  # 每个类对应一个列表
        train_loader = self.get_train_loader()

        # print(f'client{id}')
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                proto_batch = self.model.base(x_batch)  # 提取特征，不是最终分类结果
                # proto_batch = self.model.projection(proto_batch)

            # Scatter the prototypes based on their labels
            # 将每个样本的特征，按标签分类，存入 local_prototypes
            for proto, y in zip(proto_batch, y_batch):
                local_prototypes[y.item()].append(proto)

        mean_prototypes = []

        # print(f'client{self.id}')
        for class_prototypes in local_prototypes:

            if not class_prototypes == []:
                # Stack the tensors for the current class，获取当前类的所有特征
                # 对序列数据内部的张量进行扩维拼接
                stacked_protos = torch.stack(class_prototypes)

                # Compute the mean tensor for the current class，求特征原型
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes.append(mean_proto)
            else:
                mean_prototypes.append(None)

        mean_prototypes_tensor = torch.stack([proto for proto in mean_prototypes if proto is not None])
        # mean_prototypes_tensor = torch.stack(mean_prototypes)
        # mean prototypes经过本地head，得到的
        alignment_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # 以KL散度作为损失函数
        # alignment_loss_fn = torch.nn.MSELoss()

        alignment_optimizer = torch.optim.SGD(global_head.parameters(),
                                              lr=0.01)  # Adjust learning rate and optimizer as needed
        # alignment_loss_fn = torch.nn.MSELoss()
        # protoLocalLogits = self.model.head(mean_prototypes_tensor)
        # proroGlobalLogits = global_head(mean_prototypes_tensor)
        #
        # loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
        #                          torch.nn.functional.softmax(proroGlobalLogits, dim=1))
        #
        # alignment_optimizer.zero_grad()
        # loss.backward()
        # alignment_optimizer.step()
        for _ in range(1):  # Iterate for 1 epochs; adjust as needed
            # 将特征原型分别送入本地分类头和全局分类头
            protoLocalLogits = self.model.head(mean_prototypes_tensor)
            proroGlobalLogits = global_head(mean_prototypes_tensor)
            # 计算KL散度
            loss = alignment_loss_fn(torch.nn.functional.softmax(protoLocalLogits, dim=1),
                                     torch.nn.functional.softmax(proroGlobalLogits, dim=1))

            alignment_optimizer.zero_grad()
            loss.backward()  # 用KL loss进行反向传播
            alignment_optimizer.step()  # 更新全局分类头

        # self.plot_confusion_matrix(global_head,"After Align Confusion Matrix",self.globalEpochRound)
        # Substitute the parameters of the base, enabling personalization
        for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
        torch.cuda.empty_cache()
        # print("after__________>",torch.cuda.memory_allocated() / 1024 ** 2, "MB used")

        # self.globalHead = global_head

    # 画混淆矩阵
    # autodl-tmp/FedPSY312/results/kronoDroid/dirichlet
    def plot_confusion_matrix(self, globalHead, title, epoch):
        # rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        output_dir = f"/root/autodl-tmp/FedPSY312/results/kronoDroid/hunxiaoKronoDroid/{epoch}"
        output_path = os.path.join(output_dir, f"{self.client_id}-{title}.png")

        # 检查并创建文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        test_loader = self.get_test_loader()

        true_labels = []
        predicted_labels = []

        globalHead.eval()  # 设置为评估模式
        self.model.eval()  # 如果需要用到特征提取器

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # 获取特征
                features = self.model.base(x_batch)
                logits = globalHead(features)
                preds = logits.argmax(dim=1)  # 获取预测类别

                true_labels.extend(y_batch.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        # 绘制混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.xlabel("Predict Label")
        plt.ylabel("True Label")
        plt.savefig(output_path)  # 保存图片
        # plt.show()

    # plan B（样本）
    # def alignModels(self, global_head):
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             proto_batch = self.model.base(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #     mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             mean_prototypes.append(mean_proto)
    #         else:
    #             mean_prototypes.append(None)
    #
    #     mean_prototypes_tensor = torch.stack([proto for proto in mean_prototypes if proto is not None])
    #
    #     # local_logits = self.model.head(mean_prototypes_tensor)
    #     # global_logits = global_head(mean_prototypes_tensor)
    #     # local_probs = torch.softmax(local_logits, dim=1)
    #     # global_probs = torch.softmax(global_logits, dim=1)
    #
    #
    #     global_params = global_head.state_dict()
    #     local_params = self.model.head.state_dict()
    #     # localWeight = local_probs/(local_probs+global_probs)
    #     # globalWeight = global_probs/(local_probs+global_probs)
    #
    #     with torch.no_grad():
    #         # 针对每个类别分别进行处理
    #         for cls in self.owned_classes:
    #             # 当前类别的 mean prototype
    #             mean_proto = mean_prototypes[cls]
    #
    #             # 如果当前类别有原型，则继续计算
    #             if mean_proto is not None:
    #                 # 获取该类别在局部和全局模型中的 logits
    #                 local_logits = self.model.head(mean_proto.unsqueeze(0))  # 维度：1xfeature_dim
    #                 global_logits = global_head(mean_proto.unsqueeze(0))
    #
    #                 # Softmax 计算该类别的概率
    #                 local_probs = torch.softmax(local_logits, dim=1)
    #                 global_probs = torch.softmax(global_logits, dim=1)
    #
    #                 # 计算局部和全局模型的权重
    #                 local_weight = local_probs[:, cls] / (local_probs[:, cls] + global_probs[:, cls])
    #                 global_weight = global_probs[:, cls] / (local_probs[:, cls] + global_probs[:, cls])
    #
    #                 # 更新该类别的参数
    #                 for key in self.shared_keys:
    #                     local_params[key][cls] = (
    #                             local_params[key][cls] * local_weight +
    #                             global_params[key][cls] * global_weight
    #                     )
    #
    #         # 更新模型的 head 参数
    #         self.model.head.load_state_dict(local_params)
    #     # with torch.no_grad():
    #     #     # for key, param in self.globalHead.named_parameters():
    #     #     for key in self.shared_keys:
    #     #         for cls in self.owned_classes:
    #     #             # # 参数融合公式
    #     #             # local_params[key][cls] = (
    #     #             #         local_params[key][cls] * localWeight[cls] +
    #     #             #         global_params[key][cls] * globalWeight[cls]
    #     #             # )
    #     #             local_param = local_params[key]
    #     #             global_param = global_params[key]
    #     #
    #     #             local_weight_for_cls = localWeight[:, cls]  # 每个 batch 对应该类的局部权重
    #     #             global_weight_for_cls = globalWeight[:, cls]  # 每个 batch 对应该类的全局权重
    #     #
    #     #             # 按 batch 和类加权
    #     #             weighted_local_param = local_param * local_weight_for_cls.sum()
    #     #             weighted_global_param = global_param * global_weight_for_cls.sum()
    #     #
    #     #             # 融合局部和全局参数
    #     #             local_params[key] = weighted_local_param + weighted_global_param
    #     # self.model.head.load_state_dict(local_params)

    # plan D 按类别
    # # 类别权重加权
    # def alignModels(self,global_head):
    #     # Get class-specific prototypes from the local model
    #     local_prototypes = [[] for _ in range(self.num_classes)]
    #     train_loader = self.get_train_loader()
    #
    #     # print(f'client{id}')
    #     for x_batch, y_batch in train_loader:
    #         x_batch = x_batch.to(self.device)
    #         y_batch = y_batch.to(self.device)
    #
    #         with torch.no_grad():
    #             proto_batch = self.model.base(x_batch)
    #
    #         # Scatter the prototypes based on their labels
    #         for proto, y in zip(proto_batch, y_batch):
    #             local_prototypes[y.item()].append(proto)
    #
    #     mean_prototypes = []
    #
    #     # print(f'client{self.id}')
    #     for class_prototypes in local_prototypes:
    #
    #         if not class_prototypes == []:
    #             # Stack the tensors for the current class
    #             stacked_protos = torch.stack(class_prototypes)
    #
    #             # Compute the mean tensor for the current class
    #             mean_proto = torch.mean(stacked_protos, dim=0)
    #             mean_prototypes.append(mean_proto)
    #         else:
    #             mean_prototypes.append(None)
    #
    #     alignment_optimizer = torch.optim.SGD(global_head.parameters(),
    #                                           lr=0.01)  # Adjust learning rate and optimizer as needed
    #     alignment_loss_fn = torch.nn.MSELoss()
    #
    #     # 初始化每个类别的准确率统计
    #     num_classes = len(mean_prototypes)  # 假设 mean_prototypes 是一个列表，长度为类别数量
    #     class_accuracies = [1.0] * num_classes  # 初始化每个类别的准确率为 1.0
    #     class_counts = [0] * num_classes  # 记录每个类别的样本数
    #
    #     for _ in range(1):  # Iterate for 1 epochs; adjust as needed
    #         for x_batch, y_batch in train_loader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #             global_proto_batch = self.model.base(x_batch)
    #
    #             # 计算每个类别的预测结果
    #             outputs = global_head(global_proto_batch)  # 假设 global_proto_batch 是模型的输出 logits
    #             preds = torch.argmax(outputs, dim=1)
    #             correct = (preds == y_batch).float()  # 正确预测的样本
    #
    #             # 更新每个类别的准确率
    #             for label in y_batch.unique():
    #                 label = label.item()
    #                 mask = (y_batch == label)
    #                 class_counts[label] += mask.sum().item()
    #                 class_accuracies[label] = correct[mask].sum().item() / mask.sum().item()  # 更新准确率
    #
    #             # print("class_accuracies",class_accuracies)
    #             # 计算加权损失
    #             loss = 0
    #             for label in y_batch.unique():
    #                 label = label.item()
    #                 if mean_prototypes[label] is not None:
    #                     # 使用准确率作为权重（准确率越低，权重越大）
    #                     # weight =math.exp(1.0-class_accuracies[label])
    #                     weight =1.0-class_accuracies[label]
    #                     # print("weight:",weight)# 准确率越低，权重越大
    #                     loss += weight * alignment_loss_fn(global_proto_batch[y_batch == label], mean_prototypes[label])
    #
    #             # 反向传播和优化
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             alignment_optimizer.step()
    #
    #     # Substitute the parameters of the base, enabling personalization
    #     for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
    #         old_param.data = new_param.data.clone()
