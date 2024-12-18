# [NeurIPS 2024] Constrained Diffusion with Trust Sampling 
William Huang, Yifeng Jiang, Tom Van Wouwe, C. Karen Liu.

[[Paper]](https://arxiv.org/abs/2411.10932)
[[Website]](https://will-s-h.github.io/trust-sampling-website/)

## Abstract
Trust sampling effectively balances between following the unconditional diffusion model and adhering to the loss guidance, enabling more flexible and accurate constrained generation. We demonstrate the efficacy of our method through extensive experiments on complex tasks, and in drastically different domains of images and 3D motion generation, showing significant improvements over existing methods in terms of generation quality.

![Teaser Figure](/figures/Teaser%20Figure.png)

## Requirements
Version numbers may not be strict requirements:
- python 3.10.13
- pytorch3d 0.7.4
- torch 2.0.0
- einops
- matplotlib
- numpy 1.24.4
- pandas
- pillow
- scipy 1.9.1
- tensorflow 2.10.0

## Usage

### 1) Download Model Checkpoints
- image tasks:
  - FFHQ: download `ffhq_10m.pt` from [link (DPS 2022)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) and place in `./runs/image`
  - ImageNet: download `imagenet256.pt` from [link (DPS 2022)](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh) and place in `./runs/image`
- motion tasks:
  - Download `exp4-train-4950.pt` (diffusion model parameters) and `motion-encoder-267.pt` (motion encoder parameters) from [link (Trust 2024)](https://drive.google.com/drive/folders/1BFmqhsesWGKsNQBhAX90Rlo8JZr_jmIn?usp=sharing) and place in `./runs/motion`

### 2) Data
- image tasks:
  - FFHQ
    - example dataset located in `./dataset/ffhq256-4`
  - ImageNet
    - example dataset located in `./dataset/imagenet-4`
  - random masks used for box inpainting located in `./dataset/masks.pt`
- motion tasks:
  - AMASS
    - example dataset located in `./data/AMASS_10`

### 3) Demo
Note: there are associated arguments with the following scripts that can be 
- image tasks:
  ```
  cd demo_image
  python demo.py
  ```
- motion tasks:
  ```
  cd demo_motion
  python demo.py
  ```

## Citation
```
@article{huang2024trust,
  author    = {Huang, William and Jiang, Yifeng and Van Wouwe, Tom and Liu, C Karen},
  title     = {Constrained Diffusion with Trust Sampling},
  journal   = {NeurIPS},
  year      = {2024},
}
```