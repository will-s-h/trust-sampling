import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

class InpaintConstraint():
    def __init__(self, reference_image, mask, device = 'cuda'):
        self.device = device
        self.mask = mask
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.ref_image = self._format_image(reference_image)
    
    def __str__(self):
        return "Inpaint"
    
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
                return ref_image
            else:
                raise ValueError(f"Unknown tensor shape {ref_image.shape}")
    
    def get_reference_image(self):
        ref_image = self.ref_image if self.ref_image.dim() == 4 else self.ref_image.unsqueeze(0)
        return ref_image * self.get_mask(self.ref_image)
    
    def get_mask(self, samples):
        if self.mask.shape == samples.shape: return self.mask
        # if self.mask.dim() == samples.dim(): raise ValueError("Can't provide 4D mask and non-matching 4D samples")
        return self.mask.expand((samples.shape[0],) + self.mask.shape)
    
    def constraint(self, samples):
        # print(self.ref_image.shape, samples.shape)
        # assert self.ref_image.dim() == samples.dim() - 1 or self.ref_image.shape[0] == samples.shape[0]
        difference = (samples - self.ref_image) * self.get_mask(samples)
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


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask
        
def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w