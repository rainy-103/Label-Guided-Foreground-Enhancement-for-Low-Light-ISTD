import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
from tqdm import tqdm
from model.UKNet import UKNet
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

# =============== 训练超参数 ===============
epochs = 5
batch_size = 1
learning_rate = 1e-4
img_size = (320, 320)
dataset_path = "dataset/DenseSIRST-master/data/SIRSTdevkit"
output_path = "checkpoints"
os.makedirs(output_path, exist_ok=True)

# ============= 自定义数据集类 =============
class SIRSTDataset(Dataset):
    def __init__(self, dataset_path, img_size):
        self.dataset_path = dataset_path
        image_dir = os.path.join(dataset_path, 'PNGImages')
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size).astype(np.float32) / 255.0

        sky_mask_path = os.path.join(self.dataset_path, 'SkySeg', 'BinaryMask', os.path.basename(image_path))
        sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
        sky_mask = cv2.resize(sky_mask, self.img_size).astype(np.float32) / 255.0 if sky_mask is not None else np.ones(self.img_size, dtype=np.float32)

        point_label = np.zeros(self.img_size, dtype=np.float32)
        point_label_path = os.path.join(self.dataset_path, 'Point_label', os.path.basename(image_path).replace('.png', '.txt'))
        if os.path.exists(point_label_path):
            points = np.loadtxt(point_label_path, dtype=int)
            points = [points] if points.ndim == 1 else points
            for (x, y) in points:
                x, y = int(x * self.img_size[0] / image.shape[1]), int(y * self.img_size[1] / image.shape[0])
                if 0 <= x < self.img_size[0] and 0 <= y < self.img_size[1]:
                    point_label[y, x] = 1

        return (torch.from_numpy(image).permute(2, 0, 1),
                torch.from_numpy(sky_mask).unsqueeze(0),
                torch.from_numpy(point_label).unsqueeze(0))

# ============= 损失函数定义 =============
class WeightedMSELoss(nn.Module):
    def __init__(self, target_weight=5.0, bg_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.target_weight = target_weight
        self.bg_weight = bg_weight

    def forward(self, output, target, point_label):
        loss = self.mse(output, target)
        weights = torch.where(point_label > 0, self.target_weight, self.bg_weight)
        return (loss * weights).mean()

class SSIMLoss(nn.Module):
    def forward(self, output, target):
        return 1 - ssim(output, target, data_range=1.0, size_average=True)

class CombinedLoss(nn.Module):
    def __init__(self, target_weight=5.0, bg_weight=1.0, alpha=0.8):
        super().__init__()
        self.weighted_mse = WeightedMSELoss(target_weight, bg_weight)
        self.ssim_loss = SSIMLoss()
        self.alpha = alpha

    def forward(self, output, target, point_label):
        loss_wmse = self.weighted_mse(output, target, point_label)
        loss_ssim = self.ssim_loss(output, target)
        return self.alpha * loss_wmse + (1 - self.alpha) * loss_ssim

# ============= 主程序入口 =============
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SIRSTDataset(dataset_path, img_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = UKNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CombinedLoss().to(device)

    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, sky_mask, point_label in pbar:
            inputs, sky_mask, point_label = inputs.to(device), sky_mask.to(device), point_label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, sky_mask, point_label)
            loss = criterion(outputs, inputs, point_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / (pbar.n + 1):.6f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        torch.save(model.state_dict(), os.path.join(output_path, f"model_epoch_{epoch+1}.pth"))

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, '-o', label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png", dpi=300)
    plt.show()
