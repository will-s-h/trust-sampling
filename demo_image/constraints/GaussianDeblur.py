import numpy as np
import scipy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class GaussianBlurConstraint():
    def __init__(self, reference_image, kernel_size, intensity, device='cuda', noise=0):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.ref_image = self._format_image(reference_image)
        self.noise_string = "" if noise == 0 else f"_{noise}"
        self.fixed_noise = 0 if noise == 0 else noise * torch.randn_like(self.ref_image)
        
    def __str__(self):
        return "GaussianBlur" + self.noise_string
    
    def _format_image(self, ref_image):
        if isinstance(ref_image, list):
            images = []
            for path in ref_image:
                images.append(self.transform(Image.open(path).convert('RGB')))
            ref_image = torch.stack(images, dim=0)
        
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert('RGB')
        
        if isinstance(ref_image, Image.Image):
            ref_image = self.transform(ref_image)
        
        if isinstance(ref_image, torch.Tensor):
            ref_image = ref_image.to(self.device)
            assert 3 <= ref_image.dim() <= 4 and ref_image.shape[-3] == 3, f"image must be RGB (needs 3 channels, found {ref_image.dim()})"
            if ref_image.shape[-3:] == (3, 256, 256):
                # add blurring
                return self.conv(ref_image)
            else:
                raise ValueError(f"Unknown tensor shape {ref_image.shape}")

    def constraint(self, samples):
        blurred = self.conv(samples)
        difference = blurred - (self.ref_image + self.fixed_noise)
        loss = torch.norm(difference)
        return loss

    def gradient(self, samples, func=None):
        # func should be of the form lambda x: self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
        assert func is not None
        with torch.enable_grad():
            samples.requires_grad_(True)
            next_sample = func(samples)[1]
            loss = -self.constraint(next_sample)
            return torch.autograd.grad(loss, samples)[0]

# Blurkernel code was taken from DPS 2022 (https://github.com/DPS2022/diffusion-posterior-sampling/)

class Blurkernel(nn.Module):
    def __init__(self, kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2,self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
        k = torch.from_numpy(k)
        self.k = k
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k