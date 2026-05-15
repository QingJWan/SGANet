import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=1):#
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))#1 32 1 1/1 128 1 1
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out#1 32 1 1
        return self.sigmoid(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = DepthwiseSeparableConv(dim, 2 * dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = DepthwiseSeparableConv(dim, 2 * dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = DepthwiseSeparableConv(dim, 2 * dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(2*dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.dw = DepthwiseSeparableConv(6 * dim, 2 * dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, *inputs):
        x_fused = torch.cat(inputs, dim=1)
        x_fused = self.down(x_fused)#
        x_3x3 = self.conv_3x3(x_fused)
        x_3x3 = self.spatial_attention(x_3x3) * x_3x3
        x_5x5 = self.conv_5x5(x_fused)#
        x_5x5 = self.spatial_attention(x_5x5) *x_5x5
        x_7x7 = self.conv_7x7(x_fused)#
        x_7x7 = self.spatial_attention(x_7x7 ) * x_7x7
        SA = torch.cat([x_3x3, x_5x5, x_7x7], dim=1)
        out = self.dw(SA)
        CA = self.channel_attention(out)

        out = CA * out
        x_out=self.up(x_fused)
        x_out = x_out + out

        return x_out

class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels,num_inputs=1):
        super(MSAA3, self).__init__()
        self.fusion_conv = FusionConv(in_channels * 3, out_channels)
        self.num_inputs = num_inputs
        if num_inputs == 1:
            self.fusion_conv = FusionConv(in_channels, out_channels)
        elif num_inputs == 2:
            self.fusion_conv = FusionConv(in_channels * 2, out_channels)
        elif num_inputs == 3:
            self.fusion_conv = FusionConv(in_channels * 3, out_channels)
        elif num_inputs == 4:
            self.fusion_conv = FusionConv(in_channels * 4, out_channels)
        else:
            raise ValueError("num_inputs must be 1, 2 or 3")
    def forward(self, *inputs,last=False):
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {len(inputs)}")
        if self.num_inputs == 1:
            x_fused = self.fusion_conv(inputs[0])
        else:
            x_fused = self.fusion_conv(*inputs)
        return x_fused
