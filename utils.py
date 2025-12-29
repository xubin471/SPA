"""
Utils for Dataset
Extended from ADNet code by Hansen et al.
"""
import random
import torch
import numpy as np
import operator
import os
import logging


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]),  # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]),  # lower_abdomen
    },
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()


def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


class Scores():

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


import matplotlib.pyplot as plt
def show_img(imgs:list,row=None):
    if len(imgs) == 0:
        return

    if (row is not None and row==1) or len(imgs)==1:
        iterm_per_row = len(imgs)
    else:
        iterm_per_row = len(imgs)//2

    row = len(imgs)//iterm_per_row if len(imgs)%iterm_per_row == 0 else 1+len(imgs)//iterm_per_row


    for idx,img in enumerate(imgs):
        img = img.cpu().detach().numpy() if type(img) == torch.Tensor else img
        h,w = img.shape[-2:]
        img = img.reshape(-1,h,w)[0] #(h,w)
        plt.subplot(row,iterm_per_row,idx+1);plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.show()

import torch
def kmeans_plusplus(features, n_clusters):
    # features (B, N, D)
    centroids = features[:, -1, :].unsqueeze(1)  # (B, 1, D)
    for i in range(n_clusters - 1):
        features_ex = features.unsqueeze(1).expand(-1, i + 1, -1, -1)  # (B, N, D) -> (B, i, N, D)
        dis = torch.sqrt(torch.sum((features_ex - centroids.unsqueeze(2)) ** 2, dim=-1))  # (B, C, N, D) -> (B, C, N)
        new_centroid_id = torch.argmax(torch.min(dis, dim=1).values, dim=-1)  # (B, C, N) -> (B, N) -> (B)
        new_centroid = torch.gather(features, 1, new_centroid_id.unsqueeze(-1).unsqueeze(-1) \
                                    .expand(-1, -1, features.size(-1)))  # (B, 1, D)
        centroids = torch.cat([centroids, new_centroid], dim=1)
    return centroids


def kmeans(features, n_clusters=2, max_iter=50, device='cuda'):
    features = features.to(device)  # features (B, N, D)
    centroids = kmeans_plusplus(features, n_clusters)  # (B, C, D)
    features_ex = features.unsqueeze(1).expand(-1, n_clusters, -1, -1)  # (B, N, D) -> (B, C, N, D)
    cluster_label = torch.tensor(0)
    label_matrix = torch.tensor(0)
    converged = False
    for i in range(max_iter):
        pre_centroids = centroids
        dis = torch.sqrt(torch.sum((features_ex - centroids.unsqueeze(2)) ** 2, dim=-1))  # (B, C, N, D) -> (B, C, N)
        cluster_label = torch.argmin(dis, dim=1)  # (B, C, N) -> (B, N)
        label_matrix = torch.zeros(cluster_label.size(0), n_clusters, cluster_label.size(-1)).to(device)  # (B, C, N)
        label_matrix.scatter_(1, cluster_label.unsqueeze(1), 1)
        label_sum = torch.sum(label_matrix, dim=-1).unsqueeze(-1)  # (B, C, N) -> (B, C, 1)
        label_matrix /= label_sum
        centroids = torch.bmm(label_matrix, features)  # (B, C, N)*(B, N, D)*  -> (B, C, D)
        if torch.allclose(pre_centroids, centroids):
            converged = True
            break
    # if not converged:
    #     print('Warning: Clustering is not converged.')
    return cluster_label, label_matrix, centroids