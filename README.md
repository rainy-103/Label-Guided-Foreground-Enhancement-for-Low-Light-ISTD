# Label-Guided-Foreground-Enhancement-for-Low-Light-Infrared-Small-Target-Detection
# Label-Guided Foreground Enhancement for Low-Light Infrared Small Target Detection

This repository contains the PyTorch implementation of the method proposed in our paper **"Label-Guided Foreground Enhancement for Low-Light Infrared Small Target Detection"**.

Our method introduces a label-guided foreground enhancement strategy for low-light infrared small target detection. By applying enhancement only within target regions while preserving background statistics, the framework improves target visibility and detector robustness under challenging low-light conditions.

## Authors and Contributors

This code was implemented by:

- Dr. Wenjun Zhou (Email: zhouwenjun@swpu.edu.cn)
- Mr. Pinyuan Zhao
- Mr. Jiachen Dang
- Mr. Bolin Xiao
- Mr. Bo Peng

From the School of Computer Science and Software Engineering, Southwest Petroleum University.

## Usage Notice

- Feel free to download and use this code for academic research and algorithm testing.
- If you use this code in your publications, please cite our paper properly.

Thank you for your support and cooperation.

## Dependencies

Our implementation is built on the [BasicIRSTD toolbox](http://github.com/XinyiYing/BasicIRSTD), which provides a standardized PyTorch pipeline for infrared small target detection, including training, testing, and evaluation.

The low-light foreground enhancement module used in this repository is integrated into the IRSTD training framework for label-guided enhancement experiments.

### Main Environment

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pillow
- scikit-image
- matplotlib
- tqdm

Depending on your environment, you may also need:

- timm
- thop
- opencv-python
- piq
- lpips

## Repository Structure

```text
code/
  IRSTD/
    enhancement.py
    net.py
    train.py
    test.py
    inference.py
    dataset.py
    loss.py
    metrics.py
    utils.py
    model/
  UKNet/
    uknet_gray.py
    ...
