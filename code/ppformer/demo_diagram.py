# demo_diagram_style.py
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import cv2  # 用固定伪彩 LUT

from model.UKNet import UKNet  # 需要 UKNet 中已实现 forward_diagram()（我们之前给过）

DEFAULT_CKPT = "checkpoints/LOL2/net_g_BEST_PSNR_26.27_SSIM_0.95.pth"


# ---------- IO & pad ----------
def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return ten

def pad_to_multiple(x, multiple=32):
    _, _, h, w = x.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)

def unpad(x, ph, pw):
    return x[..., :x.shape[-2] - (ph or 0), :x.shape[-1] - (pw or 0)]

def save_img(tensor, path):
    t = tensor.detach().cpu().clamp(0, 1)
    if t.dim() == 4: t = t[0]
    if t.size(0) == 1: t = t.repeat(3, 1, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    TF.to_pil_image(t).save(path)


# ---------- 颜色风格：固定伪彩 ----------
def percentile_norm(x, p_lo=1.0, p_hi=99.0, eps=1e-8):
    """按百分位归一化到[0,1]，稳住不同图的亮度范围"""
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    x = (x - lo) / (hi - lo + eps)
    return np.clip(x, 0.0, 1.0)

def to_colormap(gray01, cmap="turbo", gamma=1.0):
    """
    gray01: H×W，0-1
    cmap: 'turbo' | 'inferno' | 'magma' | 'viridis' | 'plasma' | 'jet'
    gamma: 伽马压缩，<1 提升亮部
    """
    x = np.power(gray01, gamma)
    x8 = (x * 255.0 + 0.5).astype(np.uint8)
    table = {
        "turbo":   cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma":   cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma":  cv2.COLORMAP_PLASMA,
        "jet":     cv2.COLORMAP_JET,
    }
    cm = table.get(cmap, cv2.COLORMAP_TURBO)
    bgr = cv2.applyColorMap(x8, cm)             # H×W×3 (BGR, uint8)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return rgb


def tensor_feat_to_gray01(feat):
    """
    把 [B,C,H,W] 特征转成单通道 0-1 灰度：
    - 先按通道取均值（更稳），
    - 再百分位归一化。
    """
    f = feat.detach().cpu().mean(dim=1, keepdim=False)  # [B,H,W]
    if f.dim() == 3: f = f[0]                           # [H,W]
    arr = f.numpy()
    arr = percentile_norm(arr, 1.0, 99.0)
    return arr


def style_visualize(feat, style="turbo", gamma=0.9):
    """
    对中间特征应用固定风格（颜色稳定）：
    style: 'turbo'（多彩）/ 'magma'（红系）/ 'inferno' 等
    """
    gray01 = tensor_feat_to_gray01(feat)
    return to_colormap(gray01, cmap=style, gamma=gamma)


def tone_map_rgb01(t, gamma=0.9, gain=1.1):
    """给 3 通道张量做轻度 tone-map，提升对比但不改变颜色空间"""
    t = t.detach().cpu().clamp(0,1)
    if t.dim()==4: t=t[0]
    # 以每图均值为中心微调
    mean = t.mean(dim=(1,2), keepdim=True)
    out = (t ** gamma - mean) * gain + mean
    return out.clamp(0,1).unsqueeze(0)


# ---------- 跑图 ----------
@torch.no_grad()
def run(img_path, ckpt, outdir, device="cuda"):
    os.makedirs(outdir, exist_ok=True)
    pb = tqdm(total=6, desc="Infer", ascii=True) if tqdm else None

    # 1) 读图 + pad
    x = load_image(img_path).to(device);            pb and pb.update(1)
    x_pad, (ph, pw) = pad_to_multiple(x, 32);       pb and pb.update(1)

    # 2) 模型
    net = UKNet().to(device).eval()
    if os.path.isfile(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        net.load_state_dict(sd, strict=False)
    pb and pb.update(1)

    # 3) 严格(a)图前向 + 拿中间量
    outs = net.forward_diagram(x_pad, return_all=True);  pb and pb.update(1)
    outs = {k:(unpad(v,ph,pw) if isinstance(v,torch.Tensor) else v) for k,v in outs.items()};  pb and pb.update(1)

    # 4) 保存（每个节点使用固定风格）
    # 输入
    save_img(x, os.path.join(outdir, "0_input.png"))

    # F（多彩热力）
    save_img(style_visualize(outs["F"], style="turbo",  gamma=0.9),
             os.path.join(outdir, "1_F_turbo.png"))

    # LEM 裸特征（偏红风格）
    save_img(style_visualize(outs["L_lem"], style="magma", gamma=0.9),
             os.path.join(outdir, "2_LEM_magma.png"))

    # L = out_l(LEM后卷积)，既保存真实 RGB，也保存红系伪彩
    save_img(tone_map_rgb01(outs["L"], gamma=0.9, gain=1.10),
             os.path.join(outdir, "3_L_rgb_tonemapped.png"))
    save_img(style_visualize(outs["L"], style="magma", gamma=0.9),
             os.path.join(outdir, "3_L_magma.png"))

    # PSM（多彩热力）
    save_img(style_visualize(outs["G_psm"], style="turbo", gamma=0.9),
             os.path.join(outdir, "4_PSM_turbo.png"))

    # STM堆叠后的特征（多彩）
    save_img(style_visualize(outs["G_stm"], style="turbo", gamma=0.9),
             os.path.join(outdir, "5_STM_turbo.png"))

    # G（右支 3 通道增益），保存其 sigmoid 后的“增益图”的多彩风格
    save_img(style_visualize(torch.sigmoid(outs["G"]), style="turbo", gamma=0.9),
             os.path.join(outdir, "6_G_gain_turbo.png"))

    # 最终 Y：保存真实 RGB，同时给一个多彩的伪彩便于对比
    save_img(outs["Y"], os.path.join(outdir, "7_final_Y_rgb.png"))
    save_img(style_visualize(outs["Y"], style="turbo", gamma=0.9),
             os.path.join(outdir, "7_final_Y_turbo.png"))

    pb and pb.update(1) and pb.close()
    print(f"✅ Done. results -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="单张图片路径")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="权重路径")
    ap.add_argument("--out",  default="outputs/diagram_style", help="输出目录")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if tqdm is None:
        print("Tip: 显示进度条请先安装 tqdm： pip install tqdm")
    run(args.img, args.ckpt, args.out, dev)
