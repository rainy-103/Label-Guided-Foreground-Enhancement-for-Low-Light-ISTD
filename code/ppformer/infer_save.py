# infer_save.py
import argparse
from pathlib import Path
from tqdm import tqdm
import torch, torchvision.utils as vutils
from torch.utils.data import DataLoader

from uknet_gray import UKNet
from target_enhancer import TargetEnhancer
from dataloader import SIRSTPair


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='dataset/DenseSIRST-master/data/SIRSTdevkit')
    ap.add_argument('--split',     default='test_v1.txt')
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--out_dir',   default='enhanced_images')
    ap.add_argument('--size',      type=int, default=256)
    ap.add_argument('--batch',     type=int, default=8)
    ap.add_argument('--workers',   type=int, default=4)
    return ap.parse_args()


@torch.no_grad()
def main():
    p = args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(p.out_dir).mkdir(exist_ok=True, parents=True)

    ds = SIRSTPair(root=p.data_root, split_txt=p.split, size=p.size)
    dl = DataLoader(ds, batch_size=p.batch, shuffle=False,
                    num_workers=p.workers, pin_memory=True)

    model = TargetEnhancer(UKNet(in_dim=1))
    ckpt = torch.load(p.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().to(device)
    print(f'Loaded weights from {p.ckpt}')

    idx = 0
    for low, _gt, mask in tqdm(dl, desc='Infer'):
        low, mask = low.to(device), mask.to(device)
        out = model(low, mask)                     # [B,1,H,W]
        for b in range(out.size(0)):
            vutils.save_image(out[b],
                              fp=f'{p.out_dir}/{idx:06d}.png',
                              normalize=True)
            idx += 1
    print(f'Done! {idx} images saved to {p.out_dir}')


if __name__ == '__main__':
    main()
