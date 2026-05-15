import torch
import torch.nn as nn
class FD(nn.Module): 
    def __init__(self, cin, cout, K, S, P):
        super(FD, self).__init__()
        self.fd = nn.Sequential(
            nn.Conv2d(cin, cout, K, S, P, groups=2),
            nn.BatchNorm2d(cout),
            # nn.ReLU(cout)
            nn.GELU()
        )

    def forward(self, x):
        """
            inp: B x C x T x F
        """
        return self.fd(x)
