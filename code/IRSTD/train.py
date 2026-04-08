import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['ACM', 'ALCNet'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")              
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, nargs='+', help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--pretrained", default=None, nargs='+', help="Load pretrained checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--intervals", type=int, default=10, help="Intervals for print loss")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--use_snake", type=bool, default=True, help="Use snake convolution for DBCE_U_Net (default: True)")
parser.add_argument("--use_pwd", action='store_true', help="Use PWD (Parametric Wavelet Downsampling) for DBCE_U_Net")
parser.add_argument("--pwd_wavelet", type=str, default='haar', help="Wavelet type for PWD: 'haar', 'db1', 'db2', etc. (default: 'haar')")
parser.add_argument("--use_fdsf", action='store_true', help="Use FDSF (Frequency-Decoupled Skip Fusion) for DBCE_U_Net")
parser.add_argument("--use_gaussian_attn", action='store_true', help="Use Gaussian Kernel Attention for DBCE_U_Net")
parser.add_argument("--attn_heads", type=int, default=8, help="Number of attention heads for Gaussian Attention (default: 8)")
parser.add_argument("--use_enhancer", action='store_true', help="Enable UKNet foreground enhancement during detector training")
parser.add_argument("--enhancer_ckpt", type=str, default=None, help="Path to pretrained UKNet checkpoint")
parser.add_argument("--enhancer_mix_ratio", type=float, default=0.5, help="Probability of using enhanced samples during training")
parser.add_argument("--enhancer_trainable", action='store_true', help="Train enhancer jointly instead of freezing it")
parser.add_argument("--enhancer_infer", action='store_true', help="Apply enhancer during evaluation/inference when masks are unavailable")
parser.add_argument("--snr_ema_decay", type=float, default=0.9, help="EMA decay for SNR smoothing")
parser.add_argument("--snr_smax", type=float, default=None, help="Fixed SNR max for gain normalization (default: auto EMA)")
parser.add_argument("--noise_gate_kernel", type=int, default=7, help="AvgPool kernel for noise gate residual")
parser.add_argument("--disable_noise_gate", action='store_true', help="Disable noise-aware gate")
parser.add_argument("--lambda_bg", type=float, default=1.0, help="Background preservation weight")
parser.add_argument("--lambda_fa", type=float, default=10.0, help="False-alarm penalty weight")

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)

def train():
    print("Loading dataset...")
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    print(f"Dataset loaded: {len(train_set)} samples")
    
    print("Creating DataLoader...")
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    print(f"DataLoader created with batch_size={opt.batchSize}")

    print("Creating model...")
    net = Net(model_name=opt.model_name, mode='train', use_snake=opt.use_snake,
              use_pwd=opt.use_pwd, pwd_wavelet=opt.pwd_wavelet,
              use_fdsf=opt.use_fdsf, use_gaussian_attn=opt.use_gaussian_attn,
              attn_heads=opt.attn_heads, use_enhancer=opt.use_enhancer,
              enhancer_ckpt=opt.enhancer_ckpt, enhancer_mix_ratio=opt.enhancer_mix_ratio,
              enhancer_freeze=not opt.enhancer_trainable,
              enhancer_infer=opt.enhancer_infer, img_norm_cfg=opt.img_norm_cfg,
              snr_ema_decay=opt.snr_ema_decay, snr_smax=opt.snr_smax,
              noise_gate_kernel=opt.noise_gate_kernel,
              use_noise_gate=not opt.disable_noise_gate).cuda()
    net.train()
    print("Model created and moved to CUDA")
    
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.scheduler_settings['step'])):
                    opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
    
    ### Default settings                
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}
        opt.scheduler_settings['epochs'] = opt.nEpochs
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
        opt.scheduler_settings['epochs'] = opt.nEpochs
        
    opt.nEpochs = opt.scheduler_settings['epochs']

    print("Wrapping model with DataParallel...")
    net = torch.nn.DataParallel(net)
    print("Creating optimizer...")
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)
    print(f"Starting training for {opt.nEpochs} epochs...")
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        print(f"\nEpoch {idx_epoch + 1}/{opt.nEpochs}")
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            if idx_iter == 0:
                print(f"  Processing first batch: img shape={img.shape}")
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            if idx_iter == 0:
                print("  Running forward pass...")
            if opt.use_enhancer:
                pred, enh_info = net.forward(img, gt_mask, return_info=True)
            else:
                pred = net.forward(img, gt_mask)
                enh_info = None
            if idx_iter == 0:
                print(f"  Forward pass done, pred shape={pred.shape}")
                print("  Computing loss...")
            loss = net.module.loss(pred, gt_mask)
            if enh_info is not None:
                raw_unit = enh_info["raw_unit"]
                enhanced_unit = enh_info["enhanced_unit"]
                mask = enh_info["mask"]
                lbg = ((1.0 - mask) * (enhanced_unit - raw_unit).abs()).mean()
                if isinstance(pred, (list, tuple)):
                    pred_map = pred[0]
                else:
                    pred_map = pred
                lfa = (pred_map * (1.0 - mask)).mean()
                loss = loss + opt.lambda_bg * lbg + opt.lambda_fa * lfa
            total_loss_epoch.append(loss.detach().cpu())
            
            if idx_iter == 0:
                print("  Running backward pass...")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx_iter == 0:
                print(f"  First iteration complete, loss={loss.item():.4f}")

        scheduler.step()
        if (idx_epoch + 1) % opt.intervals == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            
        if (idx_epoch + 1) % 50 == 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)
            
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.module.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test', use_snake=opt.use_snake,
              use_pwd=opt.use_pwd, pwd_wavelet=opt.pwd_wavelet,
              use_fdsf=opt.use_fdsf, use_gaussian_attn=opt.use_gaussian_attn,
              attn_heads=opt.attn_heads, use_enhancer=opt.use_enhancer,
              enhancer_ckpt=opt.enhancer_ckpt, enhancer_mix_ratio=opt.enhancer_mix_ratio,
              enhancer_freeze=not opt.enhancer_trainable,
              enhancer_infer=opt.enhancer_infer, img_norm_cfg=opt.img_norm_cfg,
              snr_ema_decay=opt.snr_ema_decay, snr_smax=opt.snr_smax,
              noise_gate_kernel=opt.noise_gate_kernel,
              use_noise_gate=not opt.disable_noise_gate).cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
