import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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