import torch
from torch import nn
from model.net_modules import SpatialAttentionModule, RDB

class LRTC_Block(nn.Module):
    def __init__(self, HSI_channels):
        super(LRTC_Block, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.Proximal = RDB(HSI_channels=HSI_channels, growRate0=64, growRate=32, nConvLayers=8)

    def tensor_product(self, L, R):
        Lf = torch.fft.fft(torch.squeeze(L), n=L.shape[-1], dim=2).permute(2, 0, 1)
        Rf = torch.fft.fft(torch.squeeze(R), n=R.shape[-1], dim=2).permute(2, 0, 1)
        Gf = torch.matmul(Lf, Rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(Gf, n=R.shape[-1], dim=2), 0)

    def decom_solution(self, L_k, R_k, C_k):
        C = torch.fft.fft(torch.squeeze(C_k), n=C_k.shape[-1], dim=2).permute(2, 0, 1)
        L = torch.fft.fft(torch.squeeze(L_k), n=L_k.shape[-1], dim=2).permute(2, 0, 1)
        R = torch.fft.fft(torch.squeeze(R_k), n=R_k.shape[-1], dim=2).permute(2, 0, 1)

        Li = torch.matmul(torch.matmul(C, torch.transpose(torch.conj(R), 1, 2)),
                          torch.linalg.pinv(torch.matmul(R, torch.transpose(torch.conj(R), 1, 2)), rcond=1e-4)).permute(1, 2, 0)

        Ri = torch.matmul(torch.matmul(torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(L), 1, 2), L), rcond=1e-4),
                          torch.transpose(torch.conj(L), 1, 2)), C).permute(1, 2, 0)

        return torch.unsqueeze(torch.fft.irfft(Li, n=L_k.shape[-1], dim=2), 0), \
               torch.unsqueeze(torch.fft.irfft(Ri, n=R_k.shape[-1], dim=2), 0)

    def forward(self, L, R, C, G, Lg, cs_comp):

        # Update C
        psi_c = 1 + self.lamb + self.alpha
        Psi_C = self.lamb * cs_comp + self.alpha * G - Lg
        C_k = torch.div(self.tensor_product(L, R) + Psi_C, psi_c)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, C_k)

        # Update G
        G_k = self.Proximal(C_k + Lg / (self.alpha + 1e-6))

        # Update Lambda
        Lg_k = Lg + self.alpha * (C_k - G_k)

        return L_k, R_k, C_k, G_k, Lg_k


class LRTC_Net(nn.Module):
    def __init__(self, HSI_channels, N_iter=10):
        super(LRTC_Net, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter
        self.HSI_channels = HSI_channels

        # CS modules
        self.att_module = SpatialAttentionModule(HSI_channels+2)
        self.I_l_conv = nn.Conv2d(HSI_channels, 1, kernel_size=1, bias=False)

        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(LRTC_Block(HSI_channels=HSI_channels))
        self.network = nn.ModuleList(blocks_list)

    def forward(self, interp_ms, pan_image):

        # CS modules
        Il = self.I_l_conv(interp_ms)
        Gi = self.att_module(torch.cat((interp_ms, pan_image, Il), dim=1))
        P_Il = torch.Tensor.repeat(pan_image - Il, (1, interp_ms.shape[1], 1, 1))
        cs_comp = interp_ms + torch.mul(Gi, P_Il)

        # Optimal variables
        C  = interp_ms
        G  = torch.zeros(C.size(), device=torch.device('cuda'))
        Lg = torch.zeros(C.size(), device=torch.device('cuda'))
        # Init L/R
        L = torch.ones((self.HSI_channels, self.HSI_channels//2, C.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((self.HSI_channels//2, C.shape[-2], C.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        for i in range(0, self.N_iter):
            L, R, C, G, Lg = self.network[i](L, R, C, G, Lg, cs_comp)

        return cs_comp, C


if __name__ == '__main__':
    # Initialize model
    model = LRTC_Net(HSI_channels=4).cuda()
    # Syntax: model(upsampled_ms_image, pan_image)
    _, hrhs = model(torch.rand(1,4,256,256).cuda(), torch.rand(1,1,256,256).cuda())
