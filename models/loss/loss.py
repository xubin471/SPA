import torch
from torch import nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, label, distance_transform_edt
import math
# from .hist_kl_loss import DistributionConsistencyLoss
# from .boundary_loss import BoundaryLoss
from .align_loss import AlignLoss
from .pairwise_consistency_loss import PairwiseConsistencyLoss
from .boundary_shape_loss import BoundaryShapeLoss
# from .kl_shape_loss import KLShapeLoss
# from .msk_loss import MskLoss
from .l1_loss import L1Loss
from .pred_loss import PredLoss



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_loss = PredLoss()
        # self.boundary_loss = BoundaryLoss(theta0=3, theta=5)
        self.align_loss = AlignLoss()
        # self.msk_loss = MskLoss()
        self.l1_loss = L1Loss()
        # self.kl_shape_loss = KLShapeLoss()
        self.bd_shape_loss = BoundaryShapeLoss()
        self.pair_wise_consistency_loss = PairwiseConsistencyLoss()
        # self.hist_kl_loss = DistributionConsistencyLoss()


