import torch
import torch.nn as nn
import parameters
from WFF import WeightedFeatureFusion
from FD import FD
from SGFF import SelectiveFusionModule
from MSAA import MSAA
from conformer.conformer.encoder import ConformerBlock

class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None,p_dropout: float = 0.0):
        super().__init__()
        self.nb_classes = 13
        self.params=params
        self.input = nn.Sequential(
            nn.Conv2d(7, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.99),
            nn.GELU()
        )
        self.input1 = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            nn.GELU(),
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.99),
            nn.GELU()
        )

        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.gelu2 = nn.GELU()
        #
        #
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)
        self.gelu3 = nn.GELU()


        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))
        self.linear_layer2 = nn.Linear(768, 128)
        # self.linear_layer2 = nn.Linear(768,257)
        self.FD1 = FD(cin=64, cout=128, K=(1, 1), S=(1, 1), P=(0, 0))
        self.FD2 = FD(cin=64, cout=256, K=(1, 1), S=(1, 1), P=(0, 0))
        self.FD3 = FD(cin=128, cout=256, K=(1, 1), S=(1, 1), P=(0, 0))
        self.glu = nn.GELU()
        self.dropout = nn.Dropout(p=0.05)
        self.WFF = WeightedFeatureFusion(input_dim_x=256, input_dim_pooled=256)
        self.sfm1 = SelectiveFusionModule(channels=64)
        self.sfm2 = SelectiveFusionModule(channels=128)
        self.sfm3 = SelectiveFusionModule(channels=256)
        self.pool1 = nn.MaxPool2d([5, 2])
        self.pool2 = nn.MaxPool2d([1, 2])
        self.pool3=  nn.MaxPool2d([1, 2])
        self.pool4 = nn.MaxPool2d([1, 4])
        self.match_conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.99),
            nn.GELU()
        )
        self.match_conv8 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.99),
            nn.GELU()
        )
        self.MSAA1 = MSAA(in_channels=64, out_channels=64, num_inputs=1)
        self.MSAA2 = MSAA(in_channels=128, out_channels=128, num_inputs=2)
        self.MSAA3 = MSAA(in_channels=256, out_channels=256, num_inputs=3)
        self.conformer_block = ConformerBlock(
            encoder_dim=256,
            num_attention_heads=8,
            feed_forward_expansion_factor=2,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.05,
            attention_dropout_p=0.05,
            conv_dropout_p=0.05,
            conv_kernel_size=3,
            half_step_residual=True
        )
    def forward(self, x,pre_feat):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        x1 = self.input(x)
        x1=self.dropout(self.pool1(x1))
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.gelu2(x2)
        x2 = self.dropout(self.pool2(x2))
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.gelu3(x3)
        x3 = self.dropout(self.pool3(x3))
        MSAA1 = self.MSAA1(x1)
        x1_upsampled = self.pool2(self.FD1(x1))
        MSAA2 = self.MSAA2(x2, x1_upsampled)
        x1_upsampled =self.pool4(self.FD2(x1))
        x2_upsampled =self.pool3(self.FD3(x2))
        MSAA3 = self.MSAA3(x3, x1_upsampled, x2_upsampled)
        pre_feat = self.glu(self.linear_layer2(pre_feat))
        pre_feat1 = self.input1(pre_feat)
        pre_feat1= self.dropout(self.pool1(pre_feat1))
        Y1 = self.match_conv4(self.sfm1(pre_feat1, MSAA1))
        Y1= self.dropout(self.pool2(Y1))
        Y2 = self.match_conv8(self.sfm2(Y1,MSAA2 ))
        Y2 = self.dropout(self.pool3(Y2))
        Y3 = self.sfm(Y2 ,MSAA3)
        Y3 =  self.WFF(x3, Y3)

        Y3_3 = torch.mean(Y3, dim=3).permute(0, 2, 1)

        x = self.conformer_block( Y3_3 )

        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x9 = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x9)
        return doa




