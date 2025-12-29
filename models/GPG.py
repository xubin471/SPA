import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import kmeans

class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, dim // 2)
        self.key = nn.Linear(dim, dim // 2)
        self.value = nn.Linear(dim, dim)
        self.mapping = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(dim**0.5), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q_init, k_init):
        """
        Inputs:
            q_init: [1, dim] (global node)
            k_init: [N, dim] (local nodes)
        """
        assert q_init.dim() == 2 and k_init.dim() == 2, f"input dimension error: q_init={q_init.shape}, k_init={k_init.shape}"

        # 1. linear proj
        q = self.query(F.normalize(q_init))  # [1, dim//2]
        k = self.key(F.normalize(k_init))  # [N, dim//2]
        v = self.value(F.normalize(k_init))  # [N, dim]

        # 2. attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = (scores - scores.mean())/(scores.std()+1e-8)# [1, N]

        # 3. Softmax
        attn = F.softmax(scores.clamp(-10, 10), dim=-1)
        attn = self.dropout(attn)

        # 4. Mapping
        out = self.mapping(torch.matmul(attn, v))  # [1, dim]

        gamma = torch.sigmoid(self.gamma)
        final = gamma*q_init + (1-gamma) * out  # res
        return final


class GPG(nn.Module):
    def __init__(self, feature_dim=512, num_nodes=50):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes

        self.gcn_layers = nn.ModuleList()
        self.gcn_layer1 = nn.Linear(self.feature_dim, self.feature_dim)
        self.gcn_layer2 = nn.Linear(self.feature_dim, self.feature_dim)

        self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_nodes)
        self.alpha = nn.Parameter(torch.tensor(-4.0), requires_grad=True)
        self.CA = CrossAttention(dim=feature_dim)

    def forward(self, fts, msk):
        """
        fts: [1, 512, H, W]
        msk: [1, H, W]
        """
        # 1. align
        if fts.shape[-2:] != msk.shape[-2:]:
            fts = F.interpolate(fts, size=msk.shape[-2:], mode='bilinear', align_corners=True)

        # 2. (Global Prototype)
        msk_expanded = msk.unsqueeze(1)  # [1, 1, H, W]
        glob_proto = torch.sum(fts * msk_expanded, dim=(-2, -1)) / (torch.sum(msk_expanded) + 1e-8)  # [1, 512]
        if msk_expanded.sum() < self.num_nodes:
            return glob_proto

        # 3. (Local Nodes)
        fg_fts_pixels1 = fts.masked_select(msk_expanded.bool()).view(1, self.feature_dim, -1)  # [1, 512, N]
        local_protos1 = self.adaptive_pooling(fg_fts_pixels1)  # [1, 512, num_nodes//2]
        local_protos = local_protos1.permute(0, 2, 1).squeeze(0)  # [num_nodes, 512]

        # 4.  (Initial Node Features)
        H_in = torch.cat((glob_proto, local_protos), dim=0)  # [num_nodes+1, 512]

        # ==========================================
        # Step 2: (GCN Reasoning)
        # ==========================================

        # 2.1   (Adjacency Matrix Construction  --- A)
        H_norm = F.normalize(H_in, p=2, dim=1)
        adj = torch.matmul(H_norm, H_norm.t())
        adj = F.relu(adj)

        # Top-k optim
        topk = min(self.num_nodes // 4, adj.shape[1] - 1)  # 避免k超过节点数
        topk_values, topk_indices = torch.topk(adj, topk, dim=1)
        mask_sparse = torch.zeros_like(adj).to(fts.device)
        mask_sparse.scatter_(1, topk_indices, 1.0)
        adj = mask_sparse

        # 2.2 (Laplacian Normalization)
        row_sum = adj.sum(dim=1) + 1e-8  # D
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

        # 2.3 two-layers GCN
        h_hidden = torch.matmul(adj_normalized, H_in)
        h_hidden = self.gcn_layer1(h_hidden)
        h_hidden = F.layer_norm(h_hidden, [self.feature_dim])
        h_hidden = F.relu(h_hidden)

        h_out = torch.matmul(adj_normalized, h_hidden)
        h_out = self.gcn_layer2(h_out)
        h_out = F.layer_norm(h_out, [self.feature_dim])  # 层归一化

        # ==========================================
        # Step 3: (Global-Guided Aggregation)
        # ==========================================
        final_proto = self.CA(q_init=glob_proto, k_init=h_out[1:])
        return final_proto