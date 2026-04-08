import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class InfraredDataset(Dataset):
    def __init__(self, dataset_path, image_size=(256, 256)):
        self.image_dir = os.path.join(dataset_path, "SIRSTdevkit", "PNGImages")
        self.sky_mask_dir = os.path.join(dataset_path, "SIRSTdevkit", "SkySeg", "BinaryMask")
        self.point_label_dir = os.path.join(dataset_path, "SIRSTdevkit", "Point_label")
        self.image_size = image_size
        self.image_list = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        filename = self.image_list[idx]
        image_path = os.path.join(self.image_dir, filename)
        sky_mask_path = os.path.join(self.sky_mask_dir, filename)
        point_label_path = os.path.join(self.point_label_dir, filename.replace(".png", ".txt"))

        # ✅ 读取红外图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.stack([image] * 3, axis=-1)  # 复制灰度通道变 3 通道
        image = torch.from_numpy(image).permute(2, 0, 1)  # 转换为 [C, H, W]

        # ✅ 读取天空掩码
        if os.path.exists(sky_mask_path):
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            sky_mask = np.ones(self.image_size, dtype=np.float32)  # 默认全 1
        sky_mask = torch.from_numpy(sky_mask).unsqueeze(0)  # 转换为 [1, H, W]

        # ✅ 读取目标点掩码
        if os.path.exists(point_label_path):
            points = np.loadtxt(point_label_path, dtype=int)
            point_label = np.zeros(self.image_size, dtype=np.float32)
            for (x, y) in points:
                point_label[y, x] = 1
        else:
            point_label = np.zeros(self.image_size, dtype=np.float32)  # 默认全 0
        point_label = torch.from_numpy(point_label).unsqueeze(0)  # 转换为 [1, H, W]

        return image, sky_mask, point_label, image  # 目标就是原图

