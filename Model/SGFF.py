import torch
import torch.nn as nn
import torch.nn.functional as F
class SFTLayer(nn.Module):
    def __init__(self, cond_channels=32, out_channels=64):
        super(SFTLayer, self).__init__()
        self.scale_conv = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cond_channels, out_channels, kernel_size=1)
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cond_channels, out_channels, kernel_size=1)) 
    def forward(self, fea, cond):
        scale = self.scale_conv(cond)
        shift = self.shift_conv(cond)
        return fea * scale + shift
class SpatialFeatureTransform(nn.Module):

    def __init__(self, channels):
        super(SpatialFeatureTransform, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0)

    def forward(self, x, cond):
        gamma_beta = self.conv(cond)  
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)  
        return x * gamma + beta


class SelectiveFusionModule(nn.Module):
    def __init__(self, channels):
        super(SelectiveFusionModule, self).__init__()
        self.sft = SFTLayer(cond_channels=channels,out_channels=channels)
        self.conv1x1 = nn.Conv2d(channels * 4, channels, kernel_size=1, padding=0)
        # self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)

    def forward(self, X_trans, X_cnn):
        B, C, t, f = X_trans.shape
        spatial_cos = F.cosine_similarity(X_trans, X_cnn, dim=1) 
        X_trans_flat = X_trans.view(B, C, -1)  
        X_cnn_flat = X_cnn.view(B, C, -1)
        channel_cos= F.cosine_similarity(X_trans_flat, X_cnn_flat, dim=2) 
        channel_cos = torch.sigmoid(channel_cos)  
        spatial_cos =  torch.sigmoid(spatial_cos)  
        spatial_map = spatial_cos.unsqueeze(1)  
        channel_map = channel_cos.unsqueeze(-1).unsqueeze(-1)  
        M = channel_map * spatial_map  
        X_sim = X_trans * M
        Y_sim = X_cnn * M
        sim_fuse = torch.cat([X_sim, Y_sim], dim=1) 
        X_dis = X_trans * (1 - M)
        Y_dis = X_cnn * (1 - M)
        X_dis_mod = self.sft(X_dis, Y_dis)  
        Y_dis_mod = self.sft(Y_dis, X_dis)  
        fused = torch.cat([ X_dis_mod, Y_dis_mod], dim=1)  
        fused =torch.cat( (fused,sim_fuse), dim=1)
        out = self.conv1x1(fused) 
        return out
