


import argparse, random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # ✅ 用旧版本兼容写法

from uknet_gray import UKNet
from target_enhancer import TargetEnhancer
from dataloader import SIRSTPair
from loss_fn import MaskedLoss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='dataset/DenseSIRST-master/data/SIRSTdevkit')
    ap.add_argument('--split', default='train_v1.txt')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--save_dir', default='checkpoints')
    ap.add_argument('--resume', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    return ap.parse_args()


def seed_all(seed=2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_losses(loss_list, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve saved: {save_path}")


def main():
    args = parse_args()
    seed_all()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    # ✅ 调整参数名：split_file 而不是 split_txt
    ds = SIRSTPair(root=args.data_root, split_file=args.split, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

    # 模型加载
    backbone = UKNet(in_dim=1)
    net = TargetEnhancer(backbone)
    if torch.cuda.device_count() > 1:
        print(f'==> Using {torch.cuda.device_count()} GPUs')
        net = torch.nn.DataParallel(net)
    net = net.to(device)

    # 断点恢复
    start_ep = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['model'])
        start_ep = ckpt['epoch'] + 1
        print(f'🔄 Resumed from {args.resume} @ epoch {start_ep}')

    # 优化器、调度器、损失函数
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)
    loss_fn = MaskedLoss(1.0, 1.0)
    scaler = GradScaler(enabled=args.amp)

    # 📉 记录每个 epoch 的 loss
    epoch_losses = []

    for epoch in range(start_ep, args.epochs):
        net.train()
        pbar = tqdm(dl, desc=f'🚀 Epoch {epoch}/{args.epochs - 1}')
        total_loss = 0
        valid_count = 0

        # ✅ 这里改成接收 4 个值
        for low, gt, mask, name in pbar:
            low, gt, mask = low.to(device), gt.to(device), mask.to(device)

            # 跳过没有目标的图像
            if mask.sum() == 0:
                continue

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):  # ✅ 兼容旧版本 PyTorch
                pred = net(low, mask)
                pred = pred.clamp(0.0, 1.0)
                loss = loss_fn(pred, gt, mask)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * low.size(0)
            valid_count += low.size(0)
            pbar.set_postfix(loss=loss.item())

        if valid_count == 0:
            print(f'⚠️ Epoch {epoch}: 没有任何有效目标样本被训练，跳过。')
            continue

        avg_loss = total_loss / valid_count
        epoch_losses.append(avg_loss)
        print(f'✅ Epoch {epoch}: 平均 Loss = {avg_loss:.4f}（有效样本数：{valid_count}）')

        sch.step()

        # 模型保存
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = f'{args.save_dir}/model_epoch_{epoch}.pth'
            torch.save({'epoch': epoch, 'model': net.state_dict()}, ckpt_path)
            print(f'💾 模型已保存：{ckpt_path}')

    # 绘制 Loss 曲线
    plot_losses(epoch_losses, f"{args.save_dir}/loss_curve.png")


if __name__ == '__main__':
    main()
