"""
UKNet‑Gray : 低光/红外图像单通道增强网络
改动点：
1. 默认 in_dim = 1（灰度）
2. 第一层卷积 Conv2d(1, feat_dim, …)
3. 最后两层输出 Conv2d(feat_dim, 1, …)
其余结构保持不变，可无缝迁移除输入/输出层外的权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from thop import profile

# ---------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([.1]))

    def forward(self, x):
        return F.prelu(x, self.weight)


class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * mult, 1, 1, bias=False),
            PReLU(),
            nn.Conv2d(in_dim * mult, in_dim * mult, 3, 1, 1,
                      bias=False, groups=in_dim * mult),
            PReLU(),
            nn.Conv2d(in_dim * mult, out_dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# GCM / ISP / LEB 等核心模块（保持不变）
# ---------------------------------------------------------------------
class GCM(nn.Module):
    def __init__(self, dim, kernel_size=4, stride=4,
                 drop_path=0., ffn_ratio=4, norm_layer=nn.LayerNorm, bias=False):
        super(GCM, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.UpsamplingBilinear2d(scale_factor=1.0 / stride))
        self.norm1 = norm_layer(dim)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size // 2), groups=dim, bias=bias))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.reweight = FFN(dim, dim * 2, ffn_ratio)
        self.norm2 = norm_layer(dim)
        self.ffn = FFN(dim, dim, ffn_ratio)

    def forward(self, x):
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        d_x = self.down(x)
        x = self.drop_path(self.block(x))
        f = x + d_x
        B, C, H, W = f.shape
        f = F.adaptive_avg_pool2d(f, output_size=1)
        f = self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = f[0] * x + f[1] * d_x
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class ISP_Estimator(nn.Module):
    def __init__(self, dim, scale=2, norm_layer=nn.LayerNorm):
        super(ISP_Estimator, self).__init__()
        self.norm = norm_layer(dim)
        hidden_dim = dim * scale
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False),
            PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=7 // 2, bias=True, groups=dim),
            PReLU(),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1, bias=True))

    def forward(self, x):
        x = x + self.net(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class CrossPatchModule(nn.Module):
    def __init__(self, patches=[8, 8]):
        super().__init__()
        self.patch_h = patches[0]
        self.patch_w = patches[1]
        self.patch_n = self.patch_h * self.patch_w
        self.step = 1  # 每次只处理一组
        absolute_pos_embed = nn.Parameter(torch.zeros(1, self.step,
                                                      self.patch_n, self.patch_n, 1, 1))
        self.abs_pos = nn.init.trunc_normal_(absolute_pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        kernel_h = H // self.patch_h
        kernel_w = W // self.patch_w
        unfold = torch.nn.Unfold(kernel_size=(kernel_h, kernel_w),
                                 stride=(kernel_h, kernel_w))
        x = unfold(x)  # [N, C*kh*kw, patch_n]
        x = x.view(B, -1, kernel_h, kernel_w, self.patch_h, self.patch_w)
        x = x.view(B, -1, kernel_h, kernel_w, self.patch_n).permute(0, 1, 4, 2, 3)
        x = x.view(B, self.step, x.shape[1] // self.step,
                   self.patch_n, kernel_h, kernel_w)
        x = x + self.abs_pos
        for st in range(self.step):
            for m in range(self.patch_n):
                idx_i = list(range(m, self.patch_n)) + list(range(m))
                x[:, st, m, :] = x[:, st, m, idx_i]
        x = x.view(B, -1, self.patch_n, kernel_h, kernel_w)
        x = x.permute(0, 1, 3, 4, 2).view(B, -1, kernel_h * kernel_w, self.patch_h * self.patch_w)
        x = x.view(B, -1, self.patch_h * self.patch_w)
        fold = torch.nn.Fold(output_size=(H, W),
                             kernel_size=(kernel_h, kernel_w),
                             stride=(kernel_h, kernel_w))
        x = fold(x)
        return x


class LEB_sub(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, proj_drop=0., ffn_ratio=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias),
            Tanh(),
            nn.Conv2d(out_dim, out_dim, kernel_size=9, stride=1,
                      padding=9 // 2, groups=out_dim, bias=bias)
        )
        self.wise_features = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias),
            PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=9, stride=1,
                      padding=9 // 2, groups=out_dim, bias=bias)
        )
        self.reweight = FFN(out_dim, out_dim * 2, ffn_ratio)
        self.proj = nn.Conv2d(out_dim, out_dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x_wf = self.wise_features(x)
        x_gate = self.gate(x)
        f = x_wf + x_gate
        B, C, H, W = f.shape
        f = F.adaptive_avg_pool2d(f, output_size=1)
        f = self.reweight(f).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = f[0] * x_gate + f[1] * x_wf
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LEB(nn.Module):
    def __init__(self, dim, ffn_ratio=2., bias=False,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.block = LEB_sub(dim, dim, bias=bias, ffn_ratio=ffn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ffn = FFN(dim, dim, ffn_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.block(self.norm1(x.permute(0, 2, 3, 1))
                                          .permute(0, 3, 1, 2)))
        x = x + self.drop_path(self.ffn(self.norm2(x.permute(0, 2, 3, 1))
                                        .permute(0, 3, 1, 2)))
        return x


# ---------------------------------------------------------------------
# 主干网络 UKNet（灰度版）
# ---------------------------------------------------------------------
class UKNet(nn.Module):
    """UKNet‑Gray"""

    def __init__(self, g_layers=3, l_layers=3, patches=[8, 8],
                 in_dim=1,                # 改：默认 1 通道
                 feat_dim=96, ffn_ratio=2,
                 bias=False, drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # ---------- 编码 ----------
        self.conv_feat = nn.Conv2d(in_dim, feat_dim, 3, 1, 1, bias=False)
        self.conv_g = nn.Conv2d(feat_dim, patches[0] * patches[1], 3, 1, 1, bias=False)
        self.cpm = CrossPatchModule(patches=patches)
        self.global_cfc = nn.Conv2d(patches[0] * patches[1], feat_dim, 1, 1, 0)

        # ---------- Global Branch ----------
        self.global_branch = nn.ModuleList([
            GCM(feat_dim, kernel_size=7, stride=2, ffn_ratio=ffn_ratio)
            for _ in range(5)
        ])

        self.ch_ex1 = nn.Conv2d(feat_dim // 16, 256, 3, 1, 1, bias=False)
        self.ch_ex2 = nn.Conv2d(4, feat_dim, 1, 1, 0)

        self.net_g = nn.ModuleList([
            ISP_Estimator(dim=feat_dim, norm_layer=norm_layer)
            for _ in range(g_layers)
        ])

        # ---------- Local Branch ----------
        self.net_l = nn.ModuleList([
            LEB(dim=feat_dim, ffn_ratio=ffn_ratio, norm_layer=norm_layer)
            for _ in range(l_layers)
        ])

        # ---------- 输出层（1 通道） ----------
        self.out_g = nn.Conv2d(feat_dim, in_dim, 3, 1, 1, bias=True)
        self.out_l = nn.Conv2d(feat_dim, in_dim, 3, 1, 1, bias=True)

    def forward(self, x):
        l_x = self.conv_feat(x)           # Local features
        g_x = self.conv_g(l_x)            # Global tokens

        # Local branch
        for block in self.net_l:
            l_x = block(l_x)

        # Global branch
        B, C, H, W = g_x.shape
        g_x = self.cpm(g_x)
        g_x = self.global_cfc(g_x)
        for block in self.global_branch:
            g_x = block(g_x)

        # Channel expansion / reshape magic
        B, c, h, w = g_x.shape
        g_x = g_x.reshape(B, c // 16, h * 4, w * 4)
        g_x = self.ch_ex1(g_x)
        B, c, h, w = g_x.shape
        scale = (H // h) * (W // w)
        g_x = g_x.reshape(B, c // scale, h * (H // h), w * (W // w))
        g_x = self.ch_ex2(g_x)

        # ISP refinement
        for block in self.net_g:
            g_x = block(g_x)

        # 输出融合
        out = self.out_g(g_x) * x + self.out_l(l_x)
        return out


# ---------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UKNet(in_dim=1).to(device)
    dummy = torch.randn(2, 1, 256, 256, device=device)

    flops, params = profile(model, inputs=(dummy,))
    print(f'FLOPs: {flops/1e9:.2f} G | Params: {params/1e6:.2f} M')
