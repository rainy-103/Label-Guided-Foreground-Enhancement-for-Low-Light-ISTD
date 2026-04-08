import torch
import cv2
import os
import numpy as np
from model.UKNet import UKNet  # 确保 UKNet.py 在 model 目录中

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图像: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

def load_sky_mask(mask_path, image_size):
    if not os.path.exists(mask_path):
        print(f"未找到 {mask_path}，使用默认全天空掩码")
        return torch.ones((1, 1, image_size[0], image_size[1]), dtype=torch.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

def load_point_label(label_path, image_size):
    if not os.path.exists(label_path):
        print(f"未找到 {label_path}，使用默认全 0 目标掩码")
        return torch.zeros((1, 1, image_size[0], image_size[1]), dtype=torch.float32)

    points = np.loadtxt(label_path, dtype=int)
    label_mask = np.zeros(image_size, dtype=np.float32)
    if points.ndim == 1:
        points = [points]
    for (x, y) in points:
        if 0 <= y < image_size[0] and 0 <= x < image_size[1]:
            label_mask[y, x] = 1
    return torch.from_numpy(label_mask).unsqueeze(0).unsqueeze(0)

def process_dataset(dataset_path, output_path, model):
    input_folder = os.path.join(dataset_path, "SIRSTdevkit", "PNGImages")
    sky_mask_folder = os.path.join(dataset_path, "SIRSTdevkit", "SkySeg", "BinaryMask")
    point_label_folder = os.path.join(dataset_path, "SIRSTdevkit", "Point_label")

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"目录不存在: {input_folder}")

    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            sky_mask_path = os.path.join(sky_mask_folder, filename)
            point_label_path = os.path.join(point_label_folder, filename.replace(".png", ".txt"))

            print(f"处理 {filename} ...")

            input_image = load_image(image_path).to(device)
            sky_mask = load_sky_mask(sky_mask_path, input_image.shape[2:]).to(device)
            point_label = load_point_label(point_label_path, input_image.shape[2:]).to(device)

            with torch.no_grad():
                output = model(input_image, sky_mask, point_label)

            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            output_filename = os.path.join(output_path, filename)
            cv2.imwrite(output_filename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            print(f"已保存增强图像: {output_filename}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "dataset", "DenseSIRST-master", "data")
    output_path = os.path.join(current_dir, "enhanced_images")

    print(f"数据集路径: {dataset_path}")
    print(f"输出路径: {output_path}")

    model = UKNet()

    sid_weight_path = os.path.join(current_dir, "checkpoints","model_epoch_1.pth")
    if os.path.exists(sid_weight_path):
        model.load_state_dict(torch.load(sid_weight_path, map_location="cpu"), strict=False)
        print(f"已加载预训练权重: {sid_weight_path}")
    else:
        print(f"未找到权重文件: {sid_weight_path}，将使用未训练模型")
    process_dataset(dataset_path, output_path, model)
    print(f"全部图片处理完成，增强图像保存在: {output_path}")
