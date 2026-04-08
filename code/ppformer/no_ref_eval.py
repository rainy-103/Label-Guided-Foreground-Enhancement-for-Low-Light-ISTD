import torch
import pyiqa
import os
from natsort import natsorted
from glob import glob 
from PIL import Image
import torchvision.transforms as transforms

# 设置输入路径
inp_dir = './visual/VV'

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = 'cuda' 
else:
    device = 'cpu'

# 获取文件列表，包括 .jpg, .png, .bmp 格式的文件
files = natsorted(glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.png')) +
                  glob(os.path.join(inp_dir, '*.bmp')))

# 如果没有找到任何文件，抛出异常
if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")
 
# 初始化NIQE指标
niqe_list = []
sum = 0.0
niqe_metric = pyiqa.create_metric('pi', device=device)

# 图像预处理：使用torchvision将图像转换为张量
transform = transforms.ToTensor()

# 遍历所有文件，计算NIQE分数
for file_ in files:
    # 使用PIL加载图像
    img = Image.open(file_).convert('RGB')
    
    # 将图像转换为张量，并添加一个批量维度
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 计算NIQE分数
    score = niqe_metric(img_tensor)
    
    # 将分数添加到列表中，并打印结果
    niqe_list.append(score.data)
    print(f"{file_}: {score.data}")
    sum += score

# 输出所有图像的平均NIQE分数
print(f'Avg_NIQE: {sum / len(niqe_list)}')
