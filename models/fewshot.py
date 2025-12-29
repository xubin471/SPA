from torch import nn
from torch.nn import functional as F
import torch
from .encoder import Res50Encoder as Encoder
# from scipy.ndimage import binary_erosion, label, distance_transform_edt
from .loss.loss import Loss
import numpy as np
import matplotlib.pyplot as plt
from .SGM import SGM
from .SAE import SAE as SAE
# from .ca import CA
from .GPG import GPG as GPG
class FewShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(replace_stride_with_dilation=[True, True, False])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = 20
        self.emb_dim = 512
        self.loss = Loss()
        self.SGM = SGM(direction_num=16)
        self.criterion = nn.NLLLoss()
        self.SAE = SAE()
        self.GPG = GPG(num_nodes=50)


    def forward(self, sup_img,qry_img, sup_msk,qry_msk,mode="train"):
        # ================================================================================
        img_size = sup_msk.shape[-2:]
        img_concat = torch.cat([sup_img,qry_img],dim=0)
        # ================================================================================
        fts,tao = self.encoder(img_concat)
        sup_fts,qry_fts = fts[:1],fts[1:]
        sup_t, qry_t = tao[:1], tao[1:]
        self.sup_t, self.qry_t = sup_t, qry_t
        # ================================================================================
        pred_loss = torch.zeros(1).to(self.device)
        align_loss = torch.zeros(1).to(self.device)
        shape_loss = torch.zeros(1).to(self.device)
        # ================================================================================
        qry_fts = self.SAE(qry_fts)
        sup_fts = self.SAE(sup_fts)
        # ================================================================================
        sup_fts_bak = sup_fts.clone()
        qry_fts_bak = qry_fts.clone()
        shape_scores=None
        if sup_msk.sum() > 200 :
            shape_scores = self.SGM(sup_fts,sup_msk)
            # ================================================================================
            sup_fts =  sup_fts * shape_scores[:,:,None,None]
            qry_fts =  qry_fts * shape_scores[:,:,None,None]
        # ================================================================================
        sup_glob_proto = self.GPG(sup_fts, sup_msk)  # [1 512]
        qry_sim = self.getPred(qry_fts, sup_glob_proto, self.qry_t) #[1 1 64 64]
        qry_pred = F.interpolate(qry_sim, size=img_size, mode='bilinear', align_corners=True) #[1 1 256 256]
        qry_pred_logit = torch.cat([1-qry_pred,qry_pred],dim=1) #[1 2 256 256]


        if mode == "train":
            align_loss_item, shape_loss_item = self.align_shape_loss(sup_fts_bak,qry_fts_bak,sup_msk,qry_pred_logit,shape_scores)
            align_loss += align_loss_item
            shape_loss += shape_loss_item
            pred_loss += self.loss.pred_loss(qry_pred_logit,qry_msk)
            if torch.argmax(qry_pred_logit,dim=1).sum() > 1000:
                shape_loss += 0.005 * self.loss.bd_shape_loss(qry_pred,sup_msk[None,...])
                shape_loss += self.loss.pair_wise_consistency_loss(qry_pred_logit, qry_msk)

            return qry_pred_logit, pred_loss,align_loss, shape_loss
        else:
            return qry_pred_logit
        # ================================================================================

    def align_shape_loss(self, sup_fts, qry_fts, sup_msk, qry_pred,shape_scores1):
        """
        Args:
            supp_fts: (1 512 64 64)
            qry_fts : (1 512 64 64)
            sup_msk: (1 256 256)
            pred: (1, 2, 256, 256)
            sup_fg_pts:  (N_fg 512)
            sup_bg_pts: (N_bg 512)
        """
        qry_pred_mask = qry_pred.argmax(dim=1, keepdim=True).squeeze(1)  # (1 256 256]
        skip = qry_pred_mask.sum() == 0

        # Define loss
        align_loss = torch.zeros(1).to(self.device)
        shape_loss = torch.zeros(1).to(self.device)
        if skip:
            return align_loss, shape_loss

        # ================================================================================
        shape_scores = self.SGM(qry_fts,qry_pred_mask)
        if shape_scores1 is not None:
            shape_loss += self.loss.l1_loss(shape_scores1,shape_scores)
        # # ================================================================================
        sup_fts = sup_fts * shape_scores[:,:,None,None]
        qry_fts = qry_fts * shape_scores[:,:,None,None]
        # ================================================================================
        qry_glob_proto = self.GPG(qry_fts, qry_pred_mask)  # [1 512]
        sup_sim = self.getPred(sup_fts, qry_glob_proto, self.sup_t) #[1 1 64 64]
        sup_pred = F.interpolate(sup_sim, size=sup_msk.shape[-2:], mode='bilinear', align_corners=True) #[1 1 256 256]
        sup_pred_logit = torch.cat([1-sup_pred,sup_pred],dim=1) #[1 2 256 256]

        # Compute query loss
        eps = torch.finfo(torch.float32).eps
        log_prob = torch.log(torch.clamp(sup_pred_logit, eps, 1 - eps))
        align_loss += self.criterion(log_prob, sup_msk.long())
        return align_loss,shape_loss

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred[None,...] #(1 1 64 64)
