import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_uknet():
    this_dir = Path(__file__).resolve().parent
    uknet_dir = this_dir.parent / "UKNet"
    if not uknet_dir.exists():
        raise FileNotFoundError(f"UKNet directory not found: {uknet_dir}")
    uknet_path = str(uknet_dir)
    if uknet_path not in sys.path:
        sys.path.insert(0, uknet_path)
    from uknet_gray import UKNet  # pylint: disable=import-error

    return UKNet


class ForegroundEnhancer(nn.Module):
    def __init__(
        self,
        img_norm_cfg,
        ckpt_path=None,
        mix_ratio=0.5,
        freeze=True,
        enhance_in_eval=False,
        snr_ema_decay=0.9,
        snr_smax=None,
        noise_gate_kernel=7,
        use_noise_gate=True,
    ):
        super().__init__()
        UKNet = _load_uknet()
        self.backbone = UKNet(in_dim=1)
        self.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))
        self.mix_ratio = mix_ratio
        self.enhance_in_eval = enhance_in_eval
        self.freeze = freeze
        self.snr_ema_decay = snr_ema_decay
        self.use_noise_gate = use_noise_gate
        self.noise_gate_kernel = noise_gate_kernel

        self.register_buffer("snr_ema", torch.zeros(1, dtype=torch.float32))
        if snr_smax is None:
            self.register_buffer("snr_smax", torch.tensor(1.0, dtype=torch.float32))
            self._snr_smax_fixed = False
        else:
            self.register_buffer("snr_smax", torch.tensor(float(snr_smax), dtype=torch.float32))
            self._snr_smax_fixed = True
        gate_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.register_buffer("gate_kernel", gate_kernel)

        mean = float(img_norm_cfg["mean"])
        std = float(img_norm_cfg["std"])
        self.register_buffer("img_mean", torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("img_std", torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 1))

        if ckpt_path:
            self.load_checkpoint(ckpt_path)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Enhancer checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module.backbone."):
                cleaned[key[len("module.backbone."):]] = value
            elif key.startswith("backbone."):
                cleaned[key[len("backbone."):]] = value
            elif key.startswith("module."):
                cleaned[key[len("module."):]] = value
            else:
                cleaned[key] = value
        self.backbone.load_state_dict(cleaned, strict=False)

    def _to_unit_range(self, img):
        raw = img * self.img_std + self.img_mean
        return torch.clamp(raw / 255.0, 0.0, 1.0)

    def _to_detector_range(self, img):
        raw = torch.clamp(img * 255.0, 0.0, 255.0)
        return (raw - self.img_mean) / self.img_std

    def _expand_mask(self, mask, img):
        if mask is None:
            return torch.ones_like(img)
        mask = (mask > 0).float()
        if mask.shape != img.shape:
            mask = torch.nn.functional.interpolate(mask, size=img.shape[-2:], mode="nearest")
        return mask

    def _dilate_mask(self, mask):
        return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    def _compute_snr(self, img_unit, mask):
        eps = 1e-6
        masked = img_unit * mask
        denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = masked.sum(dim=(2, 3), keepdim=True) / denom
        var = ((masked - mean) * mask).pow(2).sum(dim=(2, 3), keepdim=True) / denom
        snr = mean / torch.sqrt(var + eps)
        return snr.squeeze(-1).squeeze(-1)

    def _compute_gain(self, snr):
        if not self._snr_smax_fixed:
            with torch.no_grad():
                snr_p95 = torch.quantile(snr.detach(), 0.95)
                self.snr_smax.mul_(self.snr_ema_decay).add_((1 - self.snr_ema_decay) * snr_p95)
        if self.snr_ema.item() == 0:
            self.snr_ema.copy_(snr.detach().mean())
        else:
            self.snr_ema.mul_(self.snr_ema_decay).add_((1 - self.snr_ema_decay) * snr.detach().mean())
        smax = self.snr_smax.clamp_min(1e-6)
        alpha = 1.0 - torch.clamp(snr / smax, 0.0, 1.0)
        return alpha

    def apply(self, img, mask=None, force=False, return_info=False):
        use_enhancer = self.training or self.enhance_in_eval or force
        if not use_enhancer:
            return (img, None) if return_info else img

        mask = self._expand_mask(mask, img)
        mask = self._dilate_mask(mask)
        enhancer_input = self._to_unit_range(img)

        if self.freeze:
            with torch.no_grad():
                enhanced = self.backbone(enhancer_input)
        else:
            enhanced = self.backbone(enhancer_input)

        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        snr = self._compute_snr(enhancer_input, mask)
        alpha = self._compute_gain(snr).view(-1, 1, 1, 1)

        if self.use_noise_gate:
            avg = F.avg_pool2d(enhancer_input, kernel_size=self.noise_gate_kernel,
                               stride=1, padding=self.noise_gate_kernel // 2)
            residual = enhancer_input - avg
            gate = torch.sigmoid(F.conv2d(residual.abs(), self.gate_kernel, padding=1))
            alpha = alpha * (1.0 - gate)
        else:
            gate = torch.zeros_like(enhancer_input)

        enhanced_fg = enhancer_input + alpha * (enhanced - enhancer_input)
        blended = enhanced_fg * mask + enhancer_input * (1.0 - mask)

        if self.training:
            selector = (torch.rand(img.size(0), 1, 1, 1, device=img.device) < self.mix_ratio).float()
            blended = selector * blended + (1.0 - selector) * enhancer_input

        mixed = self._to_detector_range(blended)
        if return_info:
            info = {
                "raw_unit": enhancer_input.detach(),
                "enhanced_unit": enhanced_fg.detach(),
                "mask": mask.detach(),
                "gain": alpha.detach(),
                "gate": gate.detach(),
            }
            return mixed, info
        return mixed
