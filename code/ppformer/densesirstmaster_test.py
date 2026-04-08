# import os
# from pathlib import Path
# from tqdm import tqdm
# import argparse
# import matplotlib.pyplot as plt

# import torch
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image

# from uknet_gray import UKNet
# from target_enhancer import TargetEnhancer
# from dataloader import SIRSTPair
# from loss_fn import MaskedLoss


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--data_root', default='dataset/DenseSIRST-master/data/SIRSTdevkit')
#     ap.add_argument('--split', default='val_v1.txt')
#     ap.add_argument('--size', type=int, default=256)
#     ap.add_argument('--batch', type=int, default=1)
#     ap.add_argument('--ckpt', default='checkpoints/model_epoch_9.pth')
#     ap.add_argument('--save_dir', default='outputs')
#     ap.add_argument('--workers', type=int, default=4)
#     return ap.parse_args()


# def main():
#     args = parse_args()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     Path(args.save_dir).mkdir(parents=True, exist_ok=True)

#     ds = SIRSTPair(args.data_root, args.split, args.size)
#     dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

#     backbone = UKNet(in_dim=1)
#     model = TargetEnhancer(backbone).to(device)

#     ckpt = torch.load(args.ckpt, map_location='cpu')
#     model.load_state_dict(ckpt['model'])
#     model.eval()

#     loss_fn = MaskedLoss(1.0, 1.0)
#     loss_total = 0
#     losses = []

#     print("🚀 Start Testing...")
#     for low, _, mask, name in tqdm(dl):
#         low, mask = low.to(device), mask.to(device)
#         gt = low

#         with torch.no_grad():
#             pred = model(low, mask)

#         # ✅ 使用原始 mask 文件名保存
#         save_image(pred, f'{args.save_dir}/{name[0]}_pred.png')
#         save_image(mask, f'{args.save_dir}/{name[0]}.png')

#         pred_clamp = torch.clamp(pred, 0, 1)
#         loss = loss_fn(pred_clamp, gt, mask).item()
#         loss_total += loss
#         losses.append(loss)

#     avg_loss = loss_total / len(losses) if losses else 0
#     print(f"✅ Avg Loss: {avg_loss:.4f}")
#     print(f"💾 所有图像保存在: {args.save_dir}")

#     plt.plot(losses)
#     plt.xlabel('Image Index')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve (per image)')
#     plt.grid(True)
#     plt.savefig(f'{args.save_dir}/loss_curve.png')
#     print(f"📈 Loss 曲线已保存：{args.save_dir}/loss_curve.png")


# if __name__ == '__main__':
#     main()
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from uknet_gray import UKNet
from target_enhancer import TargetEnhancer
from dataloader import SIRSTPair
from loss_fn import MaskedLoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='dataset/IRSTD-1k')
    ap.add_argument('--split', default='test.txt')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--ckpt', default='checkpoints/model_epoch_9.pth')
    ap.add_argument('--save_dir', default='outputs')
    ap.add_argument('--workers', type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    ds = SIRSTPair(args.data_root, args.split, args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    backbone = UKNet(in_dim=1)
    model = TargetEnhancer(backbone).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    loss_fn = MaskedLoss(1.0, 1.0)
    loss_total = 0
    losses = []

    print("🚀 Start Testing...")
    for low, _, mask, name in tqdm(dl):
        low, mask = low.to(device), mask.to(device)
        gt = low

        with torch.no_grad():
            pred = model(low, mask)

        # ✅ 只保存预测图像，不保存 mask
        save_image(pred, f'{args.save_dir}/{name[0]}_pred.png')

        pred_clamp = torch.clamp(pred, 0, 1)
        loss = loss_fn(pred_clamp, gt, mask).item()
        loss_total += loss
        losses.append(loss)

    avg_loss = loss_total / len(losses) if losses else 0
    print(f"✅ Avg Loss: {avg_loss:.4f}")
    print(f"💾 所有图像保存在: {args.save_dir}")

    plt.plot(losses)
    plt.xlabel('Image Index')
    plt.ylabel('Loss')
    plt.title('Loss Curve (per image)')
    plt.grid(True)
    plt.savefig(f'{args.save_dir}/loss_curve.png')
    print(f"📈 Loss 曲线已保存：{args.save_dir}/loss_curve.png")


if __name__ == '__main__':
    main()
