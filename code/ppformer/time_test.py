import torch
import time

from model.UKNet import UKNet



# Load corresponding models architecture and weights
print('==> Build the model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cpu'
model=UKNet()
#load_checkpoint(model, args.weights)
#model.load_state_dict(torch.load("./checkpoints/IAT-lol/best_Epoch_lol_v1.pth"))
model.to(device)
model.eval()

print('load dataset......')
bt=85
ps_h=256    
ps_w=256
gt=0
if str(device) =='cuda':
    input=torch.randn(bt,3,ps_h,ps_w).cuda()
else:
    input=torch.randn(bt,3,ps_h,ps_w)
    #input_=match_size(input_,mul=32)
t=time.time()
    #input_=transforms.Resize((1088,1920))(input_)
    #print(input_.shape)
with torch.no_grad():
        # p1,p2,restored= model_restored(input_)
        # del p1,p2
    # restored= model(input,gt,flag='eval')
        restored= model(input)
total_time = time.time() - t
avg_time = total_time / bt
fps = 1 / avg_time
ms = avg_time * 1000
print(
    f"Average inference time over image shape: {ps_h}x{ps_w} is:",
    f"{ms:.2f} ms, fps: {fps:.2f}")