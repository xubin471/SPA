import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label, distance_transform_edt
import math
class PredLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion = nn.NLLLoss(ignore_index=255,weight=torch.FloatTensor([0.1, 1.0]).cuda())

    def forward(self,query_pred,query_labels):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        query_loss = self.criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                     1 - torch.finfo(torch.float32).eps)), query_labels.long())

        return query_loss
class L1Loss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion =nn.L1Loss().cuda()

    def forward(self,a,b):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        L1Loss = self.criterion(a,b)

        return L1Loss
class AlignLoss(nn.Module):

    def __init__(self,):
        super().__init__()
        self.criterion = nn.NLLLoss()

    def forward(self,query_pred,query_labels):
        """
        Args:
            pred_mask: (1 2 h w)
            label: (1 h w)
        """
        eps = torch.finfo(torch.float32).eps
        log_prob = torch.log(torch.clamp(query_pred, eps, 1 - eps))
        align_loss = self.criterion(log_prob, query_labels.long())
        return align_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseConsistencyLoss(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.size = size
        self.criterion = nn.BCELoss().cuda()

    def forward(self, pred_mask, label):
        """
        Args:
            pred_mask: (B, 2, H, W) or (B, 1, H, W) - 概率值 [0, 1]
            label: (B, H, W) or (B, 1, H, W) - 标签值 {0, 1}
        """
        if label.dim() == 3:
            label = label.unsqueeze(1)

        label_resized = F.interpolate(label.float(), size=(self.size, self.size), mode='bilinear', align_corners=True)
        label_resized = (label_resized > 0.5).float()

        batch_size = label_resized.shape[0]
        lb_flat = label_resized.view(batch_size, -1)
        target_matrix = torch.bmm(lb_flat.unsqueeze(2), lb_flat.unsqueeze(1))

        pred_resized = F.interpolate(pred_mask, size=(self.size, self.size), mode='bilinear', align_corners=True)

        if pred_resized.shape[1] == 2:
            fg_prob = pred_resized[:, 1, ...]
        else:  
            fg_prob = pred_resized[:, 0, ...]

        fg_prob_flat = fg_prob.view(batch_size, -1)
        pred_prob_matrix = torch.bmm(fg_prob_flat.unsqueeze(2), fg_prob_flat.unsqueeze(1))

        loss = self.criterion(
            torch.clamp(pred_prob_matrix, min=1e-7, max=1 - 1e-7),
            target_matrix
        )
        return loss

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_loss = PredLoss()
        self.align_loss = AlignLoss()
        self.l1_loss = L1Loss()

        self.pair_wise_consistency_loss = PairwiseConsistencyLoss()
