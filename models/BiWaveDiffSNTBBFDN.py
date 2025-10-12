import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import torch
from ldm.ddpm import DDPM
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class CE(nn.Module):
    def __init__(self, in_channels, n_feats=64):
        super(CE, self).__init__()
        E3 = [nn.Conv2d(in_channels, n_feats //2, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.Conv2d(n_feats//2, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.AdaptiveAvgPool2d(1)]
        self.E = nn.Sequential(*E3)
        self.mlp = nn.Sequential(
            nn.Linear(n_feats, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats),
        )

    def forward(self, pan, lms):
        x = torch.cat((pan, lms), dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1

class ResMLP(nn.Module):
    def __init__(self, n_feats=512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats, n_feats),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        res = self.resmlp(x)
        return res


class denoise(nn.Module):
    def __init__(self, n_feats=64, n_denoise_res=5, timesteps=5):
        super(denoise, self).__init__()
        self.max_period = timesteps * 10
        resmlp = [
            nn.Linear(n_feats * 2 + 1, n_feats),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_feats))
        self.resmlp = nn.Sequential(*resmlp)

    def forward(self, x, t, c):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        # print(x.shape, c.shape, t.shape)
        c = torch.cat([c, t, x], dim=1)
        fea = self.resmlp(c)

        return fea

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        # out_e1 = (x^2 + 2*x)
        # out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True),
                                    dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding)

        return y

class BinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super(BinaryConv2d, self).__init__()
        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(kernel_size // 2),
                                          bias=bias)
        self.relu = RPReLU(out_channels)
        self.crossnorm = CrossLayerNorm2d(out_channels)
    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.relu(out)
        out = self.crossnorm(out, x)
        return out

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class CrossLayerNorm2d(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None
        self.b2 = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, bi, fu):
        eps = 1e-5 if fu.dtype == torch.float32 else 1e-3
        var1 = torch.var(fu, dim=1, keepdim=True)
        mean1 = torch.mean(fu, dim=1, keepdim=True)
        var2 = torch.var(bi, dim=1, keepdim=True)
        mean2 = torch.mean(bi, dim=1, keepdim=True)
        return (bi - mean2+self.b2) * (var2 + eps).rsqrt() * self.g * (var1).sqrt() + mean1 + self.b1

class LL_residual_blocks(nn.Module):
    def __init__(self, in_channels=12, out_channels=12):
        super(LL_residual_blocks, self).__init__()
        self.conv1 = BinaryConv2d(in_channels, out_channels)
        self.conv2 = BinaryConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x

class HH_residual_blocks(nn.Module):
    def __init__(self, in_channels=12*3, out_channels=12*3):
        super(HH_residual_blocks, self).__init__()
        self.conv1 = BinaryConv2d(in_channels, out_channels)
        self.conv2 = BinaryConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x


class Encoder_GT(nn.Module):
    def __init__(self, lms_channels, hid_channels):
        super(Encoder_GT, self).__init__()
        E3 = [nn.Conv2d(lms_channels, hid_channels // 2, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.Conv2d(hid_channels//2, hid_channels, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.Conv2d(hid_channels, hid_channels, kernel_size=3, padding=1),
              nn.LeakyReLU(0.1, True),
              nn.AdaptiveAvgPool2d(1)]
        self.E = nn.Sequential(*E3)
        self.mlp = nn.Sequential(
            nn.Linear(hid_channels, hid_channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hid_channels * 4, hid_channels))
    def forward(self, gt):
        out = self.E(gt).squeeze(-1).squeeze(-1)
        # print(out.shape)
        out = self.mlp(out)
        return out

class WaveletBlcok(nn.Module):
    def __init__(self, hid_channels):
        super(WaveletBlcok, self).__init__()
        # self.scale_factor = args.scale
        self.n_feats = hid_channels
        self.DWTForward = DWTForward(wave='db1', mode='symmetric')
        self.DWTInverse = DWTInverse(wave='db1', mode='symmetric')
        self.body_LL = LL_residual_blocks(in_channels=self.n_feats, out_channels=self.n_feats)
        self.body_HH = HH_residual_blocks(in_channels=self.n_feats * 3, out_channels=self.n_feats * 3)

    def forward(self, x):
        input_wavelet_LL, input_wavelet_H = self.DWTForward(x)
        b, c, n, h, w = input_wavelet_H[0].shape
        # res = input_wavelet_H[0]
        input_wavelet_H = input_wavelet_H[0].view(b, -1, h, w)
        input_wavelet_H = self.body_HH(input_wavelet_H)
        out_H = input_wavelet_H.view(b, c, n, h, w)
        out_LL = self.body_LL(input_wavelet_LL)
        out = self.DWTInverse((out_LL, [out_H]))
        # return out, out_H, out_LL, res, input_wavelet_LL
        return out

class DACA(nn.Module):
    def __init__(self, channels):
        super(DACA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channels, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(3, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, eps=1e-6):
        b, c, h, w = x.shape
        channel_mean = x.view(b, c, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(b, c, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        channel_abs = self.avg_pool(torch.abs(x)).squeeze(-1)

        y = torch.cat([channel_abs, channel_mean, channel_std], dim=2)

        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(y)


class  BodyBlock(nn.Module):
    def __init__(self, in_channels):
        super(BodyBlock, self).__init__()
        self.linear = nn.Linear(in_channels, in_channels * 2)
        self.wavelet = WaveletBlcok(in_channels)
        self.ca = DACA(in_channels)

    def fuse_prior(self, rs, prior, prior_fc):
        prior = prior_fc(prior)
        prior_mul, prior_add = torch.chunk(prior, 2, dim=1)
        prior_mul = prior_mul.unsqueeze(-1).unsqueeze(-1)
        prior_add = prior_add.unsqueeze(-1).unsqueeze(-1)
        rs = rs * prior_mul + prior_add  # BxDxHxW
        return rs

    def forward(self, x, z):
        x = self.wavelet(x) + x
        # y, out_H, out_LL, in_H, input_wavelet_LL = self.wavelet(x)
        # y = self.wavelet(x)
        # x = x + y
        res = x
        x = self.fuse_prior(x, z, self.linear)
        # res2 = x
        x = self.ca(x) * x
        x = x + res
        # return x, res, res2, out_H, out_LL, in_H, input_wavelet_LL
        return x

class Body(nn.Module):
    def __init__(self, pan_channels, lms_channels, hid_channels, layers=4):
        super(Body, self).__init__()
        self.layers = layers
        self.n_feats = hid_channels
        self.conv_head = nn.Conv2d(pan_channels + lms_channels, self.n_feats, 3, 1, 1)
        self.bodyblocks = nn.ModuleList()
        for i in range(self.layers):
            self.bodyblocks.append(BodyBlock(hid_channels))
        self.conv_last_h = nn.Conv2d(self.n_feats, lms_channels, 3, 1, 1)

    def forward(self, lms, pan, z):
        # diff_ins, diff_outs, out_Hs, out_LLs, in_Hs, in_LLs = [], [], [], [], [], []
        # ys = []
        x = self.conv_head(torch.cat((lms, pan), dim=1))
        for i in range(self.layers):
            # x, diff_in, diff_out, out_H, out_LL, in_H, input_wavelet_LL = self.bodyblocks[i](x, z)
            # diff_ins.append(diff_in)
            # diff_outs.append(diff_out)
            # out_Hs.append(out_H)
            # out_LLs.append(out_LL)
            # in_Hs.append(in_H)
            # in_LLs.append(input_wavelet_LL)
            x = self.bodyblocks[i](x, z)
            # ys.append(x)
        out = self.conv_last_h(x)
        # return out, torch.cat(diff_ins, dim=0),  torch.cat(diff_outs, dim=0),  torch.cat(out_Hs, dim=0), \
        #     torch.cat(out_LLs, dim=0),  torch.cat(in_Hs, dim=0),  torch.cat(in_LLs, dim=0)
        return out

class MS2PAN(nn.Module):
    def __init__(self, panch: int =1, msch: int = 8, dim: int = 32):
        super().__init__()
        _Net = [nn.Linear(msch, dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(dim, dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(dim, panch),]
        self.Net = nn.Sequential(*_Net)
        self.conv = nn.Conv2d(panch, dim, 3, 1, 1)
        self.DWTForward = DWTForward(wave='db1', mode='symmetric')

    def forward(self, pan, ms):
        pan_hat = self.Net(ms.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pan = self.conv(pan - pan_hat)
        panl, _ = self.DWTForward(pan)
        return panl

class DM_arch(nn.Module):
    def __init__(self, pan_channels, lms_channels, num_feat=64, n_denoise_res=1, linear_start=0.1, linear_end=0.99, timesteps=4):
        super().__init__()
        self.num_feat = num_feat
        self.condition = CE(pan_channels + lms_channels, num_feat)
        # self.denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)
        self.denoise = denoise(num_feat, n_denoise_res, timesteps)
        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition, n_feats=num_feat, linear_start=linear_start,
                              linear_end=linear_end, timesteps=timesteps)

    def forward(self, pan, lms, gt=None):
        if self.training:
            IPR, IPR_list = self.diffusion(pan, lms, gt)
            return IPR, IPR_list
        else:
            IPR = self.diffusion(pan, lms)
        return IPR

class BiWaveDiffSNTBBFDNS1(nn.Module):
    def __init__(self, pan_channels, lms_channels, hid_channels, layers=4):
        super(BiWaveDiffSNTBBFDNS1, self).__init__()
        self.LE = Encoder_GT(lms_channels, hid_channels)
        self.net = Body(pan_channels, lms_channels, hid_channels, layers)

    def forward(self, lms, ms, pan, gt):
        z = self.LE(gt)
        out = self.net(lms, pan, z)
        return out + lms

class BiWaveDiffSNTBBFDNS2(nn.Module):
    def __init__(self, pan_channels, lms_channels, hid_channels, layers=4, n_denoise_res = 1, linear_start= 0.1, linear_end= 0.99, timesteps = 4):
        super(BiWaveDiffSNTBBFDNS2, self).__init__()
        self.LE = Encoder_GT(lms_channels, hid_channels)
        self.DM = DM_arch(pan_channels, lms_channels, num_feat=hid_channels, n_denoise_res = n_denoise_res,
                          linear_start= linear_start, linear_end= linear_end, timesteps = timesteps)
        self.net = Body(pan_channels, lms_channels, hid_channels, layers)

    def forward(self, lms, ms, pan, gt=None):
        # input = F.interpolate(input, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        if self.training:
            z = self.LE(gt)
            z_hat, _ = self.DM(pan, lms, z)
            return z, z_hat
        else:
            z_hat = self.DM(pan, lms)
            out = self.net(lms, pan, z_hat)
            return out + lms

class BiWaveDiffSNTBBFDNS3(BiWaveDiffSNTBBFDNS2):
    def forward(self, lms, ms, pan, gt=None):
        if self.training:
            z = self.LE(gt)
            z_hat, _ = self.DM(pan, lms, z)
            out = self.net(lms, pan, z_hat)
            return out + lms, z, z_hat
        else:
            z_hat = self.DM(pan, lms)
            # out, diff_in, diff_out, out_H, out_LL, in_H, input_wavelet_LL = self.net(lms, pan, z_hat)
            # return out + lms, diff_in, diff_out, out_H, out_LL, in_H, input_wavelet_LL
            out = self.net(lms, pan, z_hat)
            return out + lms