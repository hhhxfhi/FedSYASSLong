# server_fedssa.py
import copy

import torch
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.server_base import ServerBase
from mem_utils import MemReporter
from src.FedPSYSSALong.client_FedPSYSSALong import ClientFedPSYSSALong
from src.FedTGP.Trainable_prototypes import Trainable_prototypes
from utils import calculate_personalization_weight
# from src.FedPSYSSALong.FedMHA import FedMHAggregator


class ServerFedPSYSSALong(ServerBase):
    def __init__(self, args):
        super().__init__(args)

        # 初始化全局共享参数
        self.received_confidences = defaultdict(dict)
        self.shared_params = copy.deepcopy(args.head.state_dict())
        self.aggregatedHead = copy.deepcopy(args.head.state_dict())  # 所有参与客户端整体 head 的加权平均版本（按样本数）
        self.initialize_clients(ClientFedPSYSSALong)
        self.class_distribution = defaultdict(list)  # 记录每个类别在哪些客户端中出现了
        # self.class_accuracy = defaultdict(list)
        # self.class_num = defaultdict(list)
        self.sample_weights = []  # 记录每个客户端的样本数量，用于加权聚合

        # 保存与训练的全局参数
        self.pretrained_global_params = None
        # self.aggregatedProjection = self.args.projection.state_dict()

        # 设置超参数
        # self.decay_rounds = 2
        # self.miu_0 = 0.5
        self.feature_dim = args.head.weight.shape[1]    # 特征维度
        self.num_classes = args.head.weight.shape[0]    # 类别数

        self.pretrained_global_params = None  # 保存预训练全局参数
        self.pretrain_rounds = args.server["pretrain_rounds"]  # 与训练轮数
        self.pretrain_sample_ration = args.server["pretrain_sample_ration"]  # 预训练客户端样本比例

        # 初始化注意力聚合器
        # self.attn_aggregator = FedMHAggregator(
        #     num_classes=self.num_classes, feature_dim=self.feature_dim, n_head=4
        # ).to(self.device)

        self.collect_class_distribution()

    def receive(self):
        self.received_protos = []
        self.received_protos_class = []
        sample_sizes = []
        proto_weights_sizes = []

        for client in self.join_clients:
            protos = []
            protosClass = []
            sizes = []
            weights = []
            for key in client.local_protos.keys():
                protos.append(client.local_protos[key])
                self.reporter.track_upload(client.local_protos[key])
                protosClass.append(key)
                sizes.append(client.statistic[str(key)])
                self.reporter.track_upload(client.statistic[str(key)])
                weights.append(client.proto_weight[key])
                self.reporter.track_upload(client.proto_weight[key])

            self.received_protos.extend(protos)
            self.received_protos_class.extend(protosClass)
            sample_sizes.extend(sizes)
            proto_weights_sizes.extend(weights)

        self.sample_weights = np.array(sample_sizes) / np.sum(sample_sizes)
        self.proto_weights = np.array(proto_weights_sizes) / np.sum(proto_weights_sizes)

        self.proto_weights += self.sample_weights
        self.proto_weights *= len(self.received_protos)
        self.sample_weights *= len(self.received_protos)

    def collect_class_distribution(self):
        """收集各客户端的类别分布"""
        for client in self.clients:
            self.sample_weights.append(client.sample_size)  # 获取样本量
            for cls in client.owned_classes:
                self.class_distribution[cls].append(client.client_id)
        self.reporter.track_upload(self.sample_weights)  # 记录上传通信大小

    def collect_class_num(self):
        """收集每个客户端拥有的类别数量"""
        for client in self.clients:
            self.class_num[client.client_id] = client.statistic

    def collect_class_accuracy(self):
        """收集各客户端的类别准确率"""
        for client in self.clients:
            self.class_accuracy[client.client_id] = client.accuracyClass
        self.reporter.track_upload(self.class_accuracy)

    # def collect_class_distribution(self):
    #     """收集每个客户端上每个类的数量"""
    #     for client in self.clients:
    #         self.sample_weights.append(client.sample_size)
    #         # 遍历每个客户端拥有的类别
    #         for cls in client.owned_classes:
    #             # 如果客户端 ID 没有对应条目，初始化为一个空字典
    #             if client.client_id not in self.class_distribution:
    #                 self.class_distribution[client.client_id] = {}
    #             # 将该类在客户端上的数量记录下来
    #             self.class_distribution[client.client_id][cls] = self.class_distribution[client.client_id].get(cls, 0) + \
    #                                                              client.class_counts[cls]

    # # 自动合法维度函数
    # def get_legal_hidden_dim(self, param_dim, num_heads):
    #     return param_dim-(param_dim%num_heads)
    def pretrain_global_model(self):
        """预训练全局模型"""
        if self.pretrained_global_params is not None:
            print("already had w0\n")
            return
        print(f"Start pretraining global model({self.pretrain_rounds} round)...")
        # 按照比例选择客户端进行预训练
        num_pretrain_clients = max(1, int(len(self.clients) * self.pretrain_sample_ration))
        pretrain_indices = torch.randperm(len(self.clients))[:num_pretrain_clients]
        pretrain_clients = [self.clients[i] for i in pretrain_indices]
        print(f"use {num_pretrain_clients}/{len(self.clients)} clients to pretrain...")

        # 从第一个客户端获取参数结构
        first_client = pretrain_clients[0]
        global_shared_params = first_client.get_shared_params()

        # 开始预训练
        for epoch in range(self.pretrain_rounds):
            # 客户端训练
            local_params_list = []
            for client in pretrain_clients:
                client.set_params(global_shared_params)
                client.train()
                local_params = client.get_shared_params()
                local_params_list.append(local_params)

            # FedAVG聚合
            global_shared_params = self._fedavg_aggregate(local_params_list)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                avg_acc = self._evaluate_globally(global_params)
                print(f"epoch: {eopch + 1}/{self.pretrain_rounds}, global acc: {avg_acc:.4f}")

        # 保存预训练参数
        self.pretrained_global_params = global_shared_params
        print(f"pretrain global model done.")

    def _fedavg_aggregate(self, local_params_list):
        """FedAVG"""
        if not local_params_list:
            return None

        # 初始化聚合参数
        aggregate_params = {}
        first_params = local_params_list[0]
        for name, param in first_params.items():
            aggregate_params[name] = torch.zeros_like(param)

        # 平均聚合
        num_clients = len(local_params_list)
        for params in local_params_list:
            for name, param in params.items():
                aggregate_params[name] += param / num_clients

        return aggregate_params

    def _evaluate_globally(self, params):
        """在客户端评估全局参数"""
        total_acc = 0.0
        count = 0
        for client in self.clients:
            if hasattr(client, 'evaluate'):
                acc = client.evaluate(params)
                total_acc += acc
                count += 1

        return total_acc / count if count > 0 else 0.0

    def _calculate_lambda_values(self, client_params_list):
        """计算每个客户端的模型差异权重λ"""
        lambda_values = []

        for params in client_params_list:
            # 计算与预训练参数ω0的差异
            diff_score = self._calculate_model_difference(params)

            # 应用分段函数转换为权重
            if diff_score >= 10 or diff_score <= -1:
                f_x = 1.0 / abs(diff_score) if diff_score != 0 else 0
            else:
                f_x = abs(diff_score)

            lambda_values.append(f_x)

        # 归一化权重
        if sum(lambda_values) > 0:
            lambda_values = [v / sum(lambda_values) for v in lambda_values]
        else:
            lambda_values = [1.0 / len(lambda_values)] * len(lambda_values)

        return lambda_values

    def _calculate_model_difference(self, params):
        """计算参数与预训练参数ω0的差异分数"""
        if self.pretrained_global_params is None:
            return 0.0

        diff_norm = 0.0
        total_elements = 0

        for k in params.keys():
            if k in self.pretrained_global_params and params[k].requires_grad:
                diff = params[k] - self.pretrained_global_params[k]
                diff_norm += torch.sum(torch.abs(diff)).item()
                total_elements += diff.numel()

        return diff_norm / total_elements if total_elements > 0 else 0.0

    def aggregate_shared_params(self):
        """聚合共享参数"""
        aggregated = defaultdict(lambda: defaultdict(list))
        client_models = []  # 客户端模型参数
        client_sample_sizes = []  # 客户端样本数量
        # client_params_list = []     # 客户端参数列表
        # # 提取客户端head参数并构建聚合向量
        # client_vectors = []
        # client_heads = []
        # confidence_weights = defaultdict(list)      # 存储置信度分数
        # 收集所有客户端的共享分类头参数
        for client in self.join_clients:
            client_params = client.model.head.state_dict()  # 客户端分类头参数
            # client_params_list.append(client_params)    # 收集客户端的分类头参数
            # client_confidence = client.upload_data["proto_weights"]   # 获取置信度分数
            # # 提取客户端head参数并构建聚合向量
            # client_heads.append(client_params)
            # flat = torch.cat([param.flatten()
            #                   for k, param in client_params.items()
            #                   if k in client.shared_keys    # 只聚合共享参数
            #                   ])
            # client_vectors.append(flat)

            client_models.append(client.model.head.state_dict())
            client_sample_sizes.append(client.sample_size)
            self.reporter.track_upload(client_params)
            for key in client.shared_keys:
                for cls in client.owned_classes:
                    aggregated[key][cls].append(client_params[key][cls])
                    # confidence_weights[cls].append(client_confidence.get(cls, 0.5))

        """适应多头注意力机制的聚合方式PlanB（在客户端模型中加入多头注意力机制，同时预训练和聚合共享参数）"""
        # 确保已经完成预训练
        if self.pretrained_global_params is None:
            self.pretrain_global_model()

        # 初始化聚合参数
        global_shared_params = {k: v for k, v in client_params_list[0].items() if k.startswith('base.')}
        aggregated_params = {k: torch.zeros_like(v) for k, v in global_shared_params.items()}
        # 计算每个客户端的模型差异权重
        if self.pretrained_global_params is not None:
            lambda_values = self._calculate_lamda_values(client_models)
        else:
            lambda_values = [1.0 for _ in client_models]

        # 将样本量和λ值转换为张量
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        lambda_tensor = torch.tensor(lambda_values, dtype=torch.float).to(self.device)
        # 计算最终权重
        client_weight = size_tensor / torch.sum(lambda_tensor)

        # 加权聚合
        for key in self.aggregatedHead.keys():
            param_list = [client_model[key] for client_model in client_models]
            param_tensor = torch.stack(param_list, dim=0)
            weight_shape = [-1]+[1]*(len(param_tensor.shape)-1)
            weight_tensor = client_weight.reshape(weight_shape)
            self.aggregatedHead[key] = torch.sum(param_tensor*weight_tensor, dim=0)

        # # 普通全局平均加权策略：按模型聚合
        # size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        # client_weight = size_tensor / torch.sum(size_tensor)
        # for key in self.aggregatedHead.keys():
        #     param = [client_model[key] for client_model in client_models]
        #     param = torch.stack(param, dim=0)
        #     shape = [-1] + [1] * (len(param.shape) - 1)
        #     self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

        # 多头注意力机制聚合
        # client_vectors = torch.stack(client_vectors).to(self.device)  # (num_clients, input_dim)
        #
        # # 调用mha计算聚合权重
        # if not hasattr(self, "mha_aggregator"):
        #     param_dim = client_vectors.shape[1]  # ← param_dim = 1105
        #     # num_heads = 8
        #     # hidden_dim = self.get_legal_hidden_dim(param_dim,num_heads) # ← 1104
        #     self.mha_aggregator = FedMHAggregator(param_dim=param_dim,num_heads=8).to(self.device)
        # output, agg_weights, attn_matrix = self.mha_aggregator(client_vectors.to(self.device))
        #
        # # 加权聚合分类头
        # for key in self.aggregatedHead.keys():
        #     stacked = torch.stack([head[key] for head in client_heads], dim=0)
        #     shape = [len(self.join_clients)]+[1]*(stacked.ndim-1)
        #     weighted = stacked*agg_weights.view(shape)  # 注意力加权
        #     self.aggregatedHead[key] = weighted.sum(dim=0)

        # # ------------------------- 多头注意力机制PlanA -------------------------
        # # 收集所有客户端的分类头参数
        # all_weights = [client.model.head.weight.data for client in
        #                self.join_clients]  # 获取分类头权重
        # all_biases = [client.model.head.bias.data for client in self.join_clients]  # 获取偏置项
        #
        # # 使用注意力聚合权重
        # aggregated_weight = self.attn_aggregator(all_weights)
        #
        # # 平均偏置项
        # aggregated_bias = torch.mean(torch.stack(all_biases), dim=0)
        #
        # # 聚合分类头
        # for cls in range(self.num_classes):
        #     self.aggregatedHead['weight'][cls] = aggregated_weight[cls]
        #     self.aggregatedHead['bias'][cls] = aggregated_bias[cls]

        join_ids = [c.client_id for c in self.join_clients]  # 当前参与客户端的ID列表
        # 按类聚合
        for key in self.shared_params.keys():
            for cls in aggregated[key]:
                # 只获取当前参与客户端中属于该类的client_id
                valid_clients = [
                    cid for cid in self.class_distribution[cls]
                    if cid in join_ids  # 关键过滤步骤
                ]

                # 获取对应客户端的权重（需确保self.sample_weights是client_id到权重的映射），改进点
                weights_sample = torch.tensor([
                    self.sample_weights[cid]
                    for cid in valid_clients
                ]).to(self.device)  # 获取参与该类聚合的客户端样本数
                weights_sample = weights_sample.float()
                weights_sample /= weights_sample.sum()
                weights = weights_sample

                # 获取对应客户端的权重（效果不佳）
                # weights_sample = torch.tensor([
                #     self.clients[cid].statistic.get(str(cls), 0)
                #     for cid in valid_clients
                # ], dtype=torch.float).to(self.device)

                # # 使用置信度分数作为权重
                # # 获取样本量和置信度权重
                # sample_weights = torch.tensor([
                #     self.sample_weights[cid]
                #     for cid in valid_clients]).to(self.device)
                # # conf_weights = torch.tensor([self.received_confidences[cls].get(cid, 0.0)
                # #                 for cid in valid_clients], dtype=torch.float).to(self.device)
                # conf_weights = torch.tensor([
                #     self.clients[cid].proto_weight.get(str(cls), 0)
                #     for cid in valid_clients
                # ], dtype=torch.float).to(self.device)

                if weights_sample.sum() == 0:
                    raise ValueError(f"Class {cls} has no valid clients in this round")
                weights = weights_sample/weights_sample.sum()    # 此时weights.sum()等于1

                # if conf_weights.sum() == 0:
                #     weights = conf_weights
                #     # raise ValueError(f"Class {cls} has no valid clients in this round")
                # else:
                #     weights = conf_weights / conf_weights.sum()  # 此时weights.sum()等于1

                # # 效果不佳
                # weights = torch.tensor([self.proto_weights[cls]], dtype=torch.float).to(self.device)
                # weights /= weights.sum()
                # print("proto_weights[cls]:" + str(self.proto_weights[cls]) + ",weights:"+str(weights)+"\n")
                # weights = sample_weights * conf_weights     # 乘积融合
                # weights /= weights.sum()

                # 原注释
                # weights = weights_sample+weights_class
                # weights /= weights.sum()
                #
                # weights = weights.float()
                # if weights.sum() == 0:
                #     raise ValueError(f"Class {cls} has no valid clients in this round")
                # weights /= weights.sum()

                # 加权聚合
                params = torch.stack(aggregated[key][cls])
                shape = [-1] + [1] * (len(params.shape) - 1)
                self.shared_params[key][cls] = torch.sum(
                    params * weights.view(*shape),
                    dim=0
                )

                # 直接平均
                # params = torch.stack(aggregated[key][cls])
                # # 对每个参数进行求平均
                # self.shared_params[key][cls] = torch.mean(params, dim=0)

    def aggregateHeadSample(self):
        client_models = []
        client_sample_sizes = []
        for client in self.join_clients:
            client_models.append(client.model.head.state_dict())
            client_sample_sizes.append(client.sample_size)
            # self.reporter.track_upload(client.model.state_dict())
        size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
        # self.reporter.track_upload(size_tensor)
        client_weight = size_tensor / torch.sum(size_tensor)

        for key in self.aggregatedHead.keys():
            param = [client_model[key] for client_model in client_models]
            param = torch.stack(param, dim=0)
            shape = [-1] + [1] * (len(param.shape) - 1)
            self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

    # def aggregateHeadAvg(self):
    #     client_models = []
    #     client_sample_sizes = []
    #     for client in self.join_clients:
    #         client_models.append(client.model.head.state_dict())
    #         client_sample_sizes.append(client.sample_size)
    #         # self.reporter.track_upload(client.model.state_dict())
    #     size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
    #     # self.reporter.track_upload(size_tensor)
    #     client_weight = size_tensor / torch.sum(size_tensor)
    #
    #     for key in self.aggregatedHead.keys():
    #         param = [client_model[key] for client_model in client_models]
    #         param = torch.stack(param, dim=0)
    #         shape = [-1] + [1] * (len(param.shape) - 1)
    #         self.aggregatedHead[key] = torch.sum(param * client_weight.reshape(shape), dim=0)

    def dispatch_parameters(self):
        """分发全局共享参数"""
        for client in self.join_clients:
            client_params = client.model.head.state_dict()
            # for key in self.shared_params.keys():
            #     for cls in client.owned_classes:
            #         client_params[key][cls] = self.shared_params[key][cls]
            for key in self.shared_params.keys():
                # 改进点
                client_params[key] = 0.8 * self.shared_params[key] + 0.2 * self.aggregatedHead[key].to('cpu')
            client.globalHead.load_state_dict(client_params)
            self.reporter.track_download(client_params)
            # client.model.projection.load_state_dict(self.aggregatedProjection)
            # client.globalHead.load_state_dict(self.aggregatedHead)

    def train(self):
        self.reporter = MemReporter()
        acc_record = {}

        for i in range(self.global_rounds):
            self.args.current_round = i
            self.join_clients = self.select_join_clients()
            print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")

            for client in self.join_clients:
                client.train()
                # self.reporter.track_upload(client.upload_data)

            self.receive()

            # 参数聚合
            self.aggregate_shared_params()
            # self.aggregateHeadSample()
            # self.aggregateHeadAvg()
            # if (i%5==0):
            #     self.aggregateProjection()

            self.dispatch_parameters()
            # self.receive_and_aggregate()
            # self.dispatch()

            print("================== evaluate =================")
            acc = self.evaluate(i)
            acc_record[i] = acc
            print(f"================= global round: {i}, accuracy: {acc} =================")

        uploadCom, downloadCom, usedMemory = self.getReport()
        return acc_record, uploadCom, downloadCom, usedMemory

# # server_fedssa.py
# import copy
#
# import torch
# import numpy as np
# from collections import defaultdict
# from src.server_base import ServerBase
# from mem_utils import MemReporter
# from src.FedSSA.client_FedSSA import ClientFedSSA
#
# class ServerFedSSA(ServerBase):
#     def __init__(self, args):
#         super().__init__(args)
#
#         # 初始化全局共享参数
#         # self.shared_params = copy.deepcopy(args.head.state_dict())
#         # self.aggregatedHead = self.args.head.state_dict()
#         self.initialize_clients(ClientFedSSA)
#         self.class_distribution = defaultdict(list)
#         self.sample_weights = []
#
#         # 设置超参数
#         self.decay_rounds = 2
#         self.miu_0 = 0.5
#
#         self.collect_class_distribution()
#         self.initializeGlobalHead()
#
#     def initializeGlobalHead(self):
#         Global_class_headers = defaultdict()
#         for key, paras in self.args.head.named_parameters():
#             Global_class_headers[key] = defaultdict()
#             for s in range(self.args.num_classes):
#                 Global_class_headers[key][s] = self.args.head[key][s]
#         self.Global_class_headers=Global_class_headers
#
#     def collect_class_distribution(self):
#         """收集各客户端的类别分布"""
#         for client in self.clients:
#             self.sample_weights.append(client.sample_size)
#             for cls in client.owned_classes:
#                 self.class_distribution[cls].append(client.client_id)
#
#     def aggregate_shared_params(self):
#         # """聚合共享参数"""
#         # aggregated = self.args.head.state_dict()
#         # # 收集所有客户端的共享参数
#         # for client in self.join_clients:
#         #     client_params = client.head.state_dict()
#         #     for key in client.shared_keys:
#         #         for cls in client.owned_classes:
#         #             aggregated[key][cls].append(client_params[key][cls])
#         #
#         # # 加权平均聚合
#         # for key in self.aggregatedHead.keys():
#         #     for cls in aggregated[key]:
#         #         params = torch.stack(aggregated[key][cls])
#         #         weights = torch.tensor([
#         #             self.sample_weights[cid]
#         #             for cid in self.class_distribution[cls]
#         #         ]).to(self.device)
#         #         weights /= weights.sum()
#         #
#         #         # 扩展维度进行加权求和
#         #         shape = [-1] + [1] * (len(params.shape) - 1)
#         #         self.aggregatedHead[key][cls] = torch.sum(
#         #             params * weights.view(*shape),
#         #             dim=0
#         #         )
#         """聚合共享参数"""
#         aggregated = defaultdict(lambda: defaultdict(list))
#
#         # 收集所有客户端的共享参数
#         for client in self.join_clients:
#             client_params = client.model.head.state_dict()
#             for key in self.Global_class_headers.keys():
#                 for cls in client.owned_classes:
#                     aggregated[key][cls].append(client_params[key][cls])
#
#         # 计算权重
#         # client_models = []
#         # client_sample_sizes = []
#         # for client in self.join_clients:
#         #     client_models.append(client.model.head.state_dict())
#         #     client_sample_sizes.append(client.sample_size)
#         #     self.reporter.track_upload(client.model.head.state_dict())
#         # size_tensor = torch.tensor(client_sample_sizes, dtype=torch.float).to(self.device)
#         # self.reporter.track_upload(size_tensor)
#         # client_weight = size_tensor / torch.sum(size_tensor)
#         #
#         # for key in self.Global_class_headers.keys():
#         #     param = [client_model[key] for client_model in client_models]
#         #     param = torch.stack(param, dim=0)
#         #     shape = [-1] + [1] * (len(param.shape) - 1)
#         #     self.aggregated[key] = torch.sum(param * client_weight.reshape(shape), dim=0)
#
#         # 加权平均聚合
#         for key in self.Global_class_headers.keys():
#             for cls in aggregated[key]:
#                 params = torch.stack(aggregated[key][cls])
#                 weights = torch.tensor([
#                     self.sample_weights[cid]
#                     for cid in self.class_distribution[cls]
#                 ]).to(self.device)
#                 # weights /= weights.sum()
#                 weights = weights.float()  # Convert weights to float
#                 weights /= weights.sum()  # Perform the division
#
#                 # 扩展维度进行加权求和
#                 shape = [-1] + [1] * (len(params.shape) - 1)
#                 self.Global_class_headers[key][cls] = torch.sum(
#                     params * weights.view(*shape),
#                     dim=0
#                 )
#
#     def dispatch_parameters(self):
#         """分发全局共享参数"""
#         for client in self.join_clients:
#             client_params = client.model.head.state_dict()
#             # for key in self.aggregatedHead.keys():
#             for key, paras in self.Global_class_headers.items():
#                 for cls in client.owned_classes:
#                     client_params[key][cls] = self.Global_class_headers[key][cls]
#             client.globalHead.load_state_dict(client_params)
#
#
#     def train(self):
#         self.reporter = MemReporter()
#         acc_record = {}
#
#         for i in range(self.global_rounds):
#             self.args.current_round = i
#             self.join_clients = self.select_join_clients()
#             print(f"============== global round: {i}, number of join clients: {len(self.join_clients)} ==============")
#
#             for client in self.join_clients:
#                 client.train()
#
#             # 参数聚合
#             self.aggregate_shared_params()
#
#             self.dispatch_parameters()
#
#             print("================== evaluate =================")
#             acc = self.evaluate()
#             acc_record[i] = acc
#             print(f"================= global round: {i}, accuracy: {acc} =================")
#
#         uploadCom, downloadCom = self.reporter.print_communication_stats()
#         return acc_record, uploadCom, downloadCom
