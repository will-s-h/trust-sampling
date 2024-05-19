from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
import lpips
import torch

device = 'cuda:0'
label_root = Path(f'/move/u/willsh/GitHub/trust-sampling/dataset/ffhq256-100')
normal_recon_root = Path(f'/move/u/willsh/GitHub/trust-sampling/demo_image/experiments/SuperResolution_ablation2/ffhq_trust_(999.0,4,1.0)')
files1 = sorted([file for file in label_root.iterdir() if file.is_file()])
files2 = sorted([file for file in normal_recon_root.iterdir() if file.is_file()])

# calculate FID
fid_value = calculate_fid_given_paths([label_root.as_posix(), normal_recon_root.as_posix()],
                                      batch_size=50,
                                      device=device,
                                      dims=2048)

# calculate PSNR and LPIPS
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

psnr_normal_list = []
lpips_normal_list = []

for file1, file2 in zip(files1, files2):
    label = plt.imread(file1)[:, :, :3]
    normal_recon = plt.imread(file2)[:, :, :3]

    psnr_normal = peak_signal_noise_ratio(label, normal_recon)
    psnr_normal_list.append(psnr_normal)

    normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)

    normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    normal_d = loss_fn_vgg(normal_recon, label).item()

    lpips_normal_list.append(normal_d)

psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

print(f'FID score: {fid_value}')
print(f'PSNR: {psnr_normal_avg}')
print(f'LPIPS: {lpips_normal_avg}')