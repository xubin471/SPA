import torch
import torch.nn as nn
import torch.nn.functional as F

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