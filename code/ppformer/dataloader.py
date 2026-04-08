# from pathlib import Path
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms as TF

# class SIRSTPair(Dataset):
#     def __init__(self, root, split_txt, size=256):
#         self.root = Path(root)
#         self.size = size

#         with open(self.root / "Splits" / split_txt, 'r') as f:
#             self.names = [line.strip() for line in f if line.strip()]

#         self.to_tensor = TF.ToTensor()

#     def __len__(self):
#         return len(self.names)

#     def __getitem__(self, idx):
#         name = self.names[idx]

#         low_path = self.root / 'PNGImages' / f'{name}.png'
#         gt_path  = self.root / 'SIRST' / 'BinaryMask' / f'{name}_pixels0.png'

#         if not low_path.exists():
#             raise FileNotFoundError(f'[LOW] {low_path} 不存在')
#         if not gt_path.exists():
#             raise FileNotFoundError(f'[GT]  {gt_path} 不存在')

#         low = Image.open(low_path).convert("L").resize((self.size, self.size))
#         gt  = Image.open(gt_path).convert("L").resize((self.size, self.size))

#         mask = gt.point(lambda p: 255 if p > 0 else 0)

#         low  = self.to_tensor(low)
#         gt   = self.to_tensor(gt)
#         mask = (self.to_tensor(mask) > 0).float()

#         return low, gt, mask, f"{name}_pixels0"  # 
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SIRSTPair(Dataset):
    def __init__(self, root, split_file, size=256):
        self.root = root
        self.size = size

        # 读取文件名列表（不带扩展名也能处理）
        split_path = os.path.join(root, 'Splits', split_file)
        with open(split_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        self.img_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        
        # 🔧 如果没有 .png 后缀，自动补全
        if not name.endswith('.png'):
            name += '.png'

        # 构造图像路径和标签路径
        img_path = os.path.join(self.root, 'PNGImages', name)
        label_path = os.path.join(self.root, 'IRSTD1k_Label', name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[IMG] {img_path} 不存在")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"[LABEL] {label_path} 不存在")

        # 加载为灰度图（L 通道）
        img = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        img = self.img_transform(img)
        label = self.mask_transform(label)

        return img, img.clone(), label, name.replace('.png', '')

