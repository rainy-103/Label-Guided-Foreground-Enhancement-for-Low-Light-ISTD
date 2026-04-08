import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob 
import cv2
import argparse
from model.UKNet import UKNet  # 确保 UKNet 这个模型存在并正确导入

# 解析命令行参数
parser = argparse.ArgumentParser(description='Demo Image Restoration')

parser.add_argument('--input_dir', default='dataset/LOLv2/Synthetic/Test/Low', type=str, help='Input images')
parser.add_argument('--result_dir', default='./visual/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/SID/net_g_BEST_PSNR_23.02_SSIM_0.7.pth', 
                    type=str, help='Path to weights')

args = parser.parse_args()

def match_size(input_, mul=8):
    """ 确保输入尺寸是 mul 的倍数 """
    h, w = input_.shape[1], input_.shape[2]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_

def save_img(filepath, img):
    """ 保存图像到文件 """
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def load_checkpoint(model, weights, device):
    """ 加载模型权重，适配不同格式 """
    print(f"Loading checkpoint from: {weights}")
    checkpoint = torch.load(weights, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "params" in checkpoint:
            model.load_state_dict(checkpoint["params"])
        else:
            # 只加载匹配的部分
            model_keys = set(model.state_dict().keys())
            filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_keys}
            model.load_state_dict(filtered_checkpoint, strict=False)
            print("Checkpoint loaded with partial matching keys.")
    else:
        model.load_state_dict(checkpoint)

    print("Checkpoint loaded successfully.")

# 确保输出文件夹存在
inp_dir = args.input_dir
out_dir = args.result_dir
os.makedirs(out_dir, exist_ok=True)

# 获取所有待处理的图片
files = natsorted(
    glob(os.path.join(inp_dir, '*.jpg')) +
    glob(os.path.join(inp_dir, '*.png')) +
    glob(os.path.join(inp_dir, '*.bmp'))
)

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# **加载模型**
print('==> Build the model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = UKNet()

# 加载权重
load_checkpoint(model, args.weights, device)

# 迁移到 GPU/CPU 并设置为 eval 模式
model.to(device)
model.eval()

# **开始处理图像**
print('Restoring images...')
index = 0

for file_ in files:
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img)

    C, H, W = input_.shape
    input_ = match_size(input_, mul=32)

    restored = input_.unsqueeze(0).to(device)

    with torch.no_grad():
        restored = model(restored)

    # 处理输出
    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :H, :W]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    # 保存图像
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(out_dir, f + '.png'), restored)

    index += 1
    print(f'Processed {index}/{len(files)}')

print(f"Files saved at {out_dir}")
print('Processing finished!')
