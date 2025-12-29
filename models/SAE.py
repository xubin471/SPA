import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SAE(nn.Module):
    def __init__(self, in_channels=512, inter_channels=None):
        super(SAE, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if self.inter_channels == 0:
            self.inter_channels = 1

        # Query Conv
        self.q = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # Key Conv
        self.k = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # Value Conv
        self.v = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        self.conv_out = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1)

        # gamma (a learnable parameter)
        # initial 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Generate Q, K, V
        q = self.q(x).view(batch_size, self.inter_channels, -1)
        q = q.permute(0, 2, 1)  # (B, H*W, C_inter)

        k = self.k(x).view(batch_size, self.inter_channels, -1)  # (B, C_inter, H*W)

        v = self.v(x).view(batch_size, self.in_channels, -1)
        v = v.permute(0, 2, 1)  # (B, H*W, C_in)

        # attn weights
        # (B, H*W, C_inter) @ (B, C_inter, H*W) -> (B, H*W, H*W)
        attention = torch.matmul(q, k)

        # scale
        scale = self.inter_channels ** -0.5
        attention = attention * scale

        # Softmax
        attention = F.softmax(attention, dim=-1)

        # (B, H*W, H*W) @ (B, H*W, C_in) -> (B, H*W, C_in)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.in_channels, height, width)

        # 1x1  Conv
        out = self.conv_out(out)

        # res-cat
        # gamma * out + x
        return self.gamma * out + x
