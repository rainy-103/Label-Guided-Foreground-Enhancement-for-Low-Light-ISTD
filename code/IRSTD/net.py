from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from enhancement import ForegroundEnhancer
from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode, use_snake=True, use_pwd=False, pwd_wavelet='haar',
                 use_fdsf=False, use_gaussian_attn=False, attn_heads=8,
                 use_enhancer=False, enhancer_ckpt=None, enhancer_mix_ratio=0.5,
                 enhancer_freeze=True, enhancer_infer=False, img_norm_cfg=None,
                 snr_ema_decay=0.9, snr_smax=None, noise_gate_kernel=7,
                 use_noise_gate=True):
        super(Net, self).__init__()
        self.model_name = model_name
        self.mode = mode
        self.enhancer = None

        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
        elif model_name == 'DBCE_U_Net':
            self.model = DBCE_U_Net(
                use_snake=use_snake,
                use_pwd=use_pwd,
                pwd_wavelet=pwd_wavelet,
                use_fdsf=use_fdsf,
                use_gaussian_attn=use_gaussian_attn,
                attn_heads=attn_heads
            )
        elif model_name == 'DBCE_U_Net_Snake':
            self.model = DBCE_U_Net(use_snake=True)
        elif model_name == 'DBCE_U_Net_Original':
            self.model = DBCE_U_Net(use_snake=False)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if use_enhancer:
            if img_norm_cfg is None:
                raise ValueError("img_norm_cfg is required when use_enhancer=True")
            self.enhancer = ForegroundEnhancer(
                img_norm_cfg=img_norm_cfg,
                ckpt_path=enhancer_ckpt,
                mix_ratio=enhancer_mix_ratio,
                freeze=enhancer_freeze,
                enhance_in_eval=enhancer_infer,
                snr_ema_decay=snr_ema_decay,
                snr_smax=snr_smax,
                noise_gate_kernel=noise_gate_kernel,
                use_noise_gate=use_noise_gate
            )

    def forward(self, img, gt_mask=None, return_info=False):
        if self.enhancer is not None:
            if return_info:
                img, info = self.enhancer.apply(img, gt_mask, return_info=True)
            else:
                img = self.enhancer.apply(img, gt_mask)
                info = None
        else:
            info = None
        pred = self.model(img)
        if return_info:
            return pred, info
        return pred

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
