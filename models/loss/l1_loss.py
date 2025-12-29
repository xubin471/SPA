import torch.nn as nn

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