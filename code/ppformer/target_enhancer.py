import torch.nn as nn

class TargetEnhancer(nn.Module):
    """
    包装器：只在 mask==1 的像素上替换增强结果，其余保持输入。
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, mask):
        """
        x    : [B,1,H,W]  低光灰度图
        mask : [B,1,H,W]  二值掩码(0/1)——目标=1，背景=0
        """
        enh = self.backbone(x)
        return enh * mask + x * (1. - mask)     # 背景直接复原
