import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SAE(nn.Module):
    """
    一个改进版的自注意力模块 (Non-local block)
    """

    def __init__(self, in_channels=512, inter_channels=None):
        super().__init__()

        self.in_channels = in_channels
        # 如果未指定中间通道数，则默认为输入通道数的一半
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        # 确保中间通道数至少为1
        if self.inter_channels == 0:
            self.inter_channels = 1

        # 查询（Query）卷积
        self.q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # 键（Key）卷积
        self.k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # 值（Value）卷积
        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # 用于将加权后的特征重新映射回原始输入通道数
        self.conv_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # gamma 是一个可学习的参数，用于缩放注意力模块的输出
        # 初始化为0，使得网络在初始阶段更依赖于原始特征
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 生成 Q, K, V
        # q 和 k 的通道数被压缩，以减少计算量
        q = self.q(x).view(batch_size, self.inter_channels, -1)
        q = q.permute(0, 2, 1)  # (B, H*W, C_inter)

        k = self.k(x).view(batch_size, self.inter_channels, -1)  # (B, C_inter, H*W)

        v = self.v(x).view(batch_size, self.in_channels, -1)
        v = v.permute(0, 2, 1)  # (B, H*W, C_in)

        # 计算注意力权重
        # (B, H*W, C_inter) @ (B, C_inter, H*W) -> (B, H*W, H*W)
        attention = torch.matmul(q, k)

        # 缩放
        scale = self.inter_channels ** -0.5
        attention = attention * scale

        # Softmax
        attention = F.softmax(attention, dim=-1)

        # 加权求和
        # (B, H*W, H*W) @ (B, H*W, C_in) -> (B, H*W, C_in)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.in_channels, height, width)

        # 通过一个1x1卷积进行特征变换
        out = self.conv_out(out)

        # 残差连接
        # gamma * out + x
        return self.gamma * out + x


# 示例
if __name__ == '__main__':
    # 创建一个输入张量
    input_tensor = torch.randn(1, 512, 64, 64)  # (batch_size, channels, height, width)

    # 改进的自注意力模块
    improved_sa = SelfAttention(in_channels=512)
    output_improved = improved_sa(input_tensor)
    print("改进模块输出尺寸:", output_improved.shape)