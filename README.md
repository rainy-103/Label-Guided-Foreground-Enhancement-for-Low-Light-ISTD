# Label-Guided Foreground Enhancement for Low-Light ISRTD

Label-Guided Foreground Enhancement for Low-Light Infrared Small Target Detection is a low-light infrared small target detection framework proposed by Dr. Wenjun Zhou and his collaborators. This repository contains the PyTorch implementation of the method from the paper **"Label-Guided Foreground Enhancement for Low-Light Infrared Small Target Detection"**.

## Authors and Contributors

This code was implemented by:

- Dr. Wenjun Zhou (Email: zhouwenjun@swpu.edu.cn)
- Mr. Pinyuan Zhao
- Mr. Jiachen Dang
- Mr. Bolin Xiao
- Mr. Bo Peng

From the School of Computer Science and Software Engineering, Southwest Petroleum University.

## Usage Notice

- Feel free to download and use this code for testing your algorithms.
- If you use this code in your publications, please cite our paper properly.

Thank you for your cooperation!

Date: 2026

## Dependencies

Our method utilizes the [BasicIRSTD toolbox](http://github.com/XinyiYing/BasicIRSTD) for training, testing, and evaluation. This open-source toolbox, based on PyTorch, provides a standardized pipeline specifically designed for infrared small target detection tasks.

## Using the Toolbox

Please refer to the instructions in the BasicIRSTD toolbox for training, testing, and evaluation of our method.

## Datasets

We used the following datasets in our experiments:

### DenseSIRST

Used to construct label-guided foreground-enhanced samples for training.
- [Download](https://github.com/GrokCV/BAFE-Net)
- [Paper](https://arxiv.org/abs/2407.20078) Xiao, M., Dai, Q., Zhu, Y., Guo, K., Wang, H., Shu, X., Yang, J., Dai, Y.:
  *Background Semantics Matter: Cross-task Feature Exchange Network for Clustered Infrared Small Target Detection with Sky-Annotated Dataset*. arXiv preprint arXiv:2407.20078 (2024)

### IRSTD-1K

Used as the benchmark dataset for detection evaluation.

- [Download](https://github.com/RuiZhang97/ISNet)
- [Paper](https://ieeexplore.ieee.org/document/9880295) Zhang, M., Zhang, R., Yang, Y., Bai, H., Zhang, J., Guo, J.:
  *ISNet: Shape Matters for Infrared Small Target Detection*. CVPR (2022)

For detailed instructions on how to use these datasets, please refer to the BasicIRSTD toolbox documentation.

## Training

### Baseline Detector Training

bash
python train.py --model_names  --dataset_names 
Training with Label-Guided Foreground Enhancement
python train.py \
  --model_names DBCE_U_Net \
  --dataset_names IRSTD-1K \
  --use_enhancer \
  --enhancer_ckpt PATH/TO/ENHANCER_CHECKPOINT.pth \
  --enhancer_mix_ratio 0.5 \
  --snr_ema_decay 0.9 \
  --noise_gate_kernel 7 \
  --lambda_bg 1.0 \
  --lambda_fa 10.0
## Testing
python test.py --model_names  --dataset_names 
Inference
python inference.py --model_names --dataset_names
## Important Note
When --use_enhancer is enabled, the IRSTD code depends on the foreground enhancement implementation in the code/UKNet directory.
Please make sure the directory structure is preserved correctly. Otherwise, the enhancement module may fail to load.

## Acknowledgement
The code is implemented based on the BasicIRSTD toolbox. We would like to express our sincere thanks to the contributors.
