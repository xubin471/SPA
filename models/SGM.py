import math

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch import nn
import cv2
import numpy as np
import math
import torch.nn.functional as F

def get_mask_edges_distances(mask, direction_num=16):
    """
    Input: ===> mask: (256,256)
    Output: ===> distance_list : []
    """

    # 获取掩码的中心点（图像几何中心）
    h, w = mask.shape[-2:]
    center = (w // 2, h // 2)

    # 检查中心点是否在掩码内，如果不在则使用掩码的质心
    if mask[center[1], center[0]] == 0:
        M = cv2.moments(mask.astype(float))
        if M["m00"] == 0:  # 掩码为空
            return None
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center = (cX, cY)

    distances = []
    angles = np.linspace(0, 2 * np.pi, direction_num, endpoint=False)  # 16个均等方向

    for angle in angles:
        # 计算方向向量
        dx = math.cos(angle)
        dy = math.sin(angle)

        # 从中心点沿方向向量移动，直到到达掩码边缘或图像边界
        distance = 0
        x, y = center

        while True:
            # 沿方向向量移动一小步
            x += dx
            y += dy
            distance += 1

            # 检查是否超出图像边界
            if round(x) < 0 or round(x) >= w or round(y) < 0 or round(y) >= h:
                break

            # 检查是否到达掩码边缘（从前景到背景）
            if mask[int(round(y)), int(round(x))] == 0:
                break

        distances.append(distance)

    return {
        "center": center,
        "angles": angles,  # 弧度
        "distances": distances
    }


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=512,
        dropout=0.2
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim*input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return (
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )




class ShapeProto(nn.Module):
    def __init__(self,emb_dim=512,protos_num=1,init_shape_dim=20):
        super().__init__()
        self.emb_dim = emb_dim
        self.protos_num = protos_num
        self.shape_encoder = nn.Sequential(
            nn.Linear(init_shape_dim,64),
            nn.ReLU(),
            nn.Linear(64,emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim,emb_dim*self.protos_num),
        )

    def forward(self,msk):
        """
        fts: (1 512 64 64)
        content_proto: glob proto (1 512)
        msk: (1 256 256)
        """
        shape_proto = self.shape_proto(msk)
        return shape_proto

    def shape_proto(self,msk):
        h,w = msk.shape[-2:]
        msk = msk.view(-1,h,w)[0]
        h_indices , w_indices = torch.where(msk==1)
        height = h_indices.max() - h_indices.min()
        width = w_indices.max() - w_indices.min()

        results = get_mask_edges_distances(msk.detach().cpu().numpy())
        distances = results["distances"]
        shape_proto = torch.tensor([[height,width,*distances,np.mean(distances),np.std(distances)]]).to(msk.device).float()
        shape_proto = self.shape_encoder(F.normalize(shape_proto,dim=1)).view(self.emb_dim,self.protos_num).permute(1,0)
        return shape_proto #[1 512]


class SGM(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape_fts_generator = MLP(input_dim=256,hidden_dim=512,dropout=0.2)
        self.scalar = 20.0
        self.shape_proto = ShapeProto()

    def forward(self, sup_fts, sup_msk):
        """

        :param sup_fts: [1 512 256 256]
        :param sup_msk: [1 256 256]
        :param shape_proto: [1 512]
        :return:
        """

        # =================================================
        sup_fts = F.interpolate(sup_fts,size=(256,256),mode="bilinear") #[1 512 256 256]
        sup_msk = (F.interpolate(sup_msk[None,...].float(),size=(256,256),mode="bilinear").squeeze(0)>0).int()
        # =================================================
        positive_sup_fts = (sup_fts * sup_msk[None,...]) #[1 512 256 256]
        c,h,w = positive_sup_fts.shape[-3:]
        positive_sup_fts = positive_sup_fts.reshape(c,-1)
        shape_sup_fts = self.shape_fts_generator(positive_sup_fts) #[512, 512] 前者是通道数量， 后者形状特征通道数量
        # =================================================
        shape_proto = self.shape_proto(sup_msk)
        # =================================================
        shape_sup_fts_norm = F.normalize(shape_sup_fts,dim=1)
        shape_proto_norm = F.normalize(shape_proto,dim=1)
        sim_score = torch.sum(shape_sup_fts_norm * shape_proto_norm,dim=-1)[:,None] #[512 1]
        sim_score = F.sigmoid(sim_score*self.scalar).permute(1,0) #[1 512]  (0-1)
        return sim_score+0.1






"""

from PIL import Image
msk = Image.open("/home/cs4007/code/Unet_seg/imgs/lb.png")
msk = np.array(msk)
msk = (msk!=0).astype(int)
msk = torch.from_numpy(msk).unsqueeze(0)

# fts = torch.randn(1,512,64,64)
content_proto = torch.randn(1,512)
shape_embedded_proto = ShapeAdaptor(emb_dim=512,protos_num=1)(content_proto,msk)
print(shape_embedded_proto.shape)
print("end")
"""

