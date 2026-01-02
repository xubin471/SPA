import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
写一个图注意力原型提取器
"""
class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim//2)
        self.key = nn.Linear(dim, dim//2)
        self.value = nn.Linear(dim, dim)
        self.mapping = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)

    def forward(self, q_init, k_init):
        """
        Inputs:
            q_init: [1 dim]
            k_init: [N dim]
        Outputs:
            refined q_init [1 dim]
        """
        x1 = q_init
        x2 = k_init
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.query.out_features) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # print(attn)
        # print(f"CA_gamma:{self.gamma}")
        return q_init + self.gamma*self.mapping(torch.matmul(attn, v))


class GPG(nn.Module):
    def __init__(self, num_gcn_layers=2, feature_dim=512, num_nodes=50):
        super().__init__()
        self.num_gcn_layers = num_gcn_layers
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes

        self.gcn_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(nn.Linear(self.feature_dim, self.feature_dim))

        # 修正：AdaptiveAvgPool1d 的参数是输出的长度
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_nodes)
        self.alpha = nn.Parameter(torch.tensor(-4.0),requires_grad=True)
        self.CA = CrossAttention()

    def forward(self, fts, msk):
        """
        fts: [1, 512, 256, 256]
        msk: [1, 256, 256] (应为单通道掩码)
        """
        fts = F.interpolate(fts, size=msk.shape[-2:], mode='bilinear', align_corners=True)

        # 1. 计算传统原型 (作为图的全局节点)
        # 确保 msk 是 [1, 1, H, W] 以便广播
        msk_expanded = msk.unsqueeze(1) # [1, 1, 256, 256]
        glob_proto = torch.sum(fts * msk_expanded, dim=(-2, -1)) / (torch.sum(msk) + 1e-8) # Shape: [1, 512]

        # 2. 提取所有前景特征作为局部节点池
        # 修正: msk应为bool或float类型, 并且只有1个通道
        fg_fts_pixels = fts.masked_select(msk_expanded.bool()).view(1, self.feature_dim, -1) # Shape: [1, 512, N]

        # 如果没有前景像素，直接返回全局原型
        if fg_fts_pixels.shape[-1] == 0:
            return glob_proto # 返回 [1 512]

        # 3. 从局部节点池中采样得到固定数量的局部节点
        # 修正：正确使用 AdaptiveAvgPool1d
        local_protos = self.adaptive_pooling(fg_fts_pixels) # Shape: [1, 512, num_nodes]
        local_protos = local_protos.permute(0, 2, 1).squeeze(0) # Shape: [num_nodes, 512]

        # 4. 组合成图的所有节点
        # 修正: glob_proto也需要调整形状
        proto = torch.cat((glob_proto, local_protos), dim=0) # Shape: [num_nodes + 1, 512]



        H = proto # Initial node features
        # 动态邻接矩阵
        for gcn_layer in self.gcn_layers:
            # 5. 构建图并执行GCN
            proto_norm = F.normalize(H, dim=1)
            adj = torch.matmul(proto_norm, proto_norm.t())  # Adjacency matrix
            adj = F.softmax(F.relu(adj), dim=1)  # Normalize edges

            aggregated_features = torch.matmul(adj, H)
            H = H + F.relu(gcn_layer(aggregated_features))

        # 6. 聚合GCN输出得到最终原型
        # 【【【核心修正】】】: 对GCN的输出 H 求平均
        #  注意力引导节点聚合
        q_node = H[:1] # 全局节点
        k_nodes = H[1:]

        final_proto = self.CA(q_node, k_nodes)


        weight = F.sigmoid(self.alpha)
        # print(weight)
        # print(f"GPG weight: {weight}")

        return glob_proto*(1-weight) + final_proto*(weight)