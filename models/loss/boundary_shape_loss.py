import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiableBoundaryExtractor(nn.Module):
    """
    一个可微分的模块,用于从概率掩码中提取边界距离向量.

    特征向量是质心到N个预设方向上"最远"边界的距离.
    """

    def __init__(self, num_directions=32, sharpness=10.0):
        """
        初始化模块.
        Args:
            num_directions (int): 方向数量.
            sharpness (float): 控制softmax的锐度. 值越大, "soft max"越接近真实的"hard max".
                               这是一个可以调整的超参数.
        """
        super().__init__()
        self.num_directions = num_directions
        self.sharpness = sharpness
        # 预先计算32个方向的单位向量
        self.directions = self._create_direction_vectors(num_directions)

    def _create_direction_vectors(self, num_directions):
        """生成均匀分布在360度内的单位方向向量."""
        angles = torch.linspace(0, 2 * math.pi, steps=num_directions, dtype=torch.float32)
        # directions shape: [num_directions, 2]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        return directions


    def forward(self, mask):
        """
        从输入的掩码中提取边界距离特征.
        Args:
            mask (torch.Tensor): 输入的预测掩码, 形状为 [B, 1, H, W], 值在[0, 1]之间.
        Returns:
            torch.Tensor: 边界距离向量, 形状为 [B, num_directions].
        """
        if mask.dim() != 4 or mask.size(1) != 1:
            raise ValueError("输入掩码的期望形状是 [B, 1, H, W]")

        batch_size, _, h, w = mask.shape
        device = mask.device

        # 将方向向量移动到与掩码相同的设备上
        self.directions = self.directions.to(device)

        # 1. 创建像素坐标网格
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                        torch.arange(w, device=device, dtype=torch.float32),
                                        indexing='ij')
        # coords 形状: [H*W, 2] -> [[y_0, x_0], [y_1, x_1], ...]
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)

        # 2. 准备掩码
        mask_flat = mask.view(batch_size, -1)
        # 归一化掩码,使其总和为1, 作为概率分布
        mask_normalized = mask_flat / (mask_flat.sum(dim=1, keepdim=True) + 1e-8)

        # 3. 计算可微分的质心 (均值)
        centroid = torch.matmul(mask_normalized, coords)  # 形状: [B, 2]

        # 4. 计算所有点相对于质心的坐标
        centered_coords = coords.unsqueeze(0) - centroid.unsqueeze(1)  # 形状: [B, H*W, 2]

        # 5. 计算所有点在所有方向上的投影距离
        # projections 形状: [B, H*W, num_directions]
        projections = torch.matmul(centered_coords, self.directions.t())

        # 6. 计算可微分的最大边界距离 (核心步骤)
        # 我们希望找到在每个方向上, 投影距离最大的那个点.
        # Softmax可以帮我们创建一个"软"的 argmax.
        # 关键: 我们将掩码的对数加到投影上. 这能确保只有在掩码区域内的点(mask > 0)
        # 才对soft argmax有贡献, 掩码外的点(mask -> 0, log(mask) -> -inf)会被抑制.
        # log_mask = torch.log(mask_flat.unsqueeze(-1) + 1e-10) + 1 # 形状: [B, H*W, 1]
        #
        # # 将log_mask广播加到projections上
        # # 这样, 投影距离远且掩码概率高的点, 会有更大的值
        # # weighted_projections = projections + log_mask
        # weighted_projections = F.relu(projections) * log_mask

        log_mask = torch.log(mask_flat.unsqueeze(-1) + 1e-10)  # 形状: [B, H*W, 1]

        # 注意: 这里是 "加法" 并且 "没有 relu"
        # 这样才能同时考虑所有方向(正向和负向)的边界
        weighted_projections = projections + log_mask

        # 在像素维度(dim=1)上计算softmax. sharpness参数让分布更尖锐.
        # softmax_weights 形状: [B, H*W, num_directions]
        softmax_weights = F.softmax(weighted_projections * self.sharpness, dim=1)

        # 使用这些权重对原始的(无偏置的)投影距离进行加权求和, 得到"软"的最大距离
        # boundary_distances 形状: [B, num_directions]
        boundary_distances = torch.sum(softmax_weights * projections, dim=1)

        return boundary_distances


class BoundaryShapeLoss(nn.Module):
    """
    计算预测掩码的边界距离向量和目标向量之间的损失.
    """

    def __init__(self, num_directions=32, sharpness=10.0):
        super().__init__()
        self.shape_extractor = DifferentiableBoundaryExtractor(
            num_directions=num_directions,
            sharpness=sharpness
        )
        # 使用 L1 Loss 来对比两个向量. L2 (MSELoss) 也是一个不错的选择.
        self.loss_fn = nn.L1Loss()

    def forward(self, pred_msk, sup_msk):
        """
        Args:
            pred_msk: (1 1 h w)
            sup_msk: (1 1 h w)
        Returns:
            torch.Tensor: 形状损失值.
        """



        # 从预测掩码中提取可微分的边界距离向量
        predicted_distance_vector = self.shape_extractor(pred_msk)
        # return predicted_distance_vector
        with torch.no_grad():
            sup_distance_vector = self.shape_extractor(sup_msk)
        # 计算预测向量和目标向量之间的L1损失
        loss = self.loss_fn(predicted_distance_vector, sup_distance_vector)
        # loss = 1-F.cosine_similarity(predicted_distance_vector,sup_distance_vector,dim=1)
        return loss
