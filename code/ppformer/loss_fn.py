import torch
import torch.nn.functional as F
from piq import multi_scale_ssim

class MaskedLoss:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, pred, gt, mask):
        # 修复负值问题
        pred = torch.clamp(pred, 0, 1)
        gt   = torch.clamp(gt, 0, 1)

        ssim = multi_scale_ssim(pred * mask, gt * mask, data_range=1.0)
        mse = F.mse_loss(pred * mask, gt * mask)

        return self.alpha * mse + self.beta * (1 - ssim)
import torch.nn.functional as F
from piq import multi_scale_ssim

class MaskedLoss:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, pred, gt, mask):
        # 修复负值问题
        pred = torch.clamp(pred, 0, 1)
        gt   = torch.clamp(gt, 0, 1)

        ssim = multi_scale_ssim(pred * mask, gt * mask, data_range=1.0)
        mse = F.mse_loss(pred * mask, gt * mask)

        return self.alpha * mse + self.beta * (1 - ssim)