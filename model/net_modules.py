import torch
from torch import nn


class SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats-2, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        att_map = self.att2(self.relu(self.att1(x)))
        return att_map


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, HSI_channels, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=HSI_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=HSI_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x
