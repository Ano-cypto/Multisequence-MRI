import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim

# Global cache for the LPIPS model
_LPIPS_MODEL = None

def get_lpips_model(device='cuda'):
    """Get the cached LPIPS model (AlexNet-based by default)"""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net='alex').to(device)
    return _LPIPS_MODEL

def denormalize(img):
    """
    Denormalizes an image from [-1, 1] to [0, 1].
    """
    img = img * 0.5 + 0.5
    return img.clamp(0, 1)

def calculate_psnr(img1, img2):
    """
    Calculates PSNR between two images (assumed to be normalized to [-1, 1]).
    
    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        
    Returns:
        float: Average PSNR over the batch.
    """
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean().item()

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two images (assumed to be normalized to [-1, 1]).
    
    Args:
        img1 (Tensor): First image.
        img2 (Tensor): Second image.
        
    Returns:
        float: Average SSIM over the batch.
    """
    img1 = denormalize(img1)
    img2 = denormalize(img2)
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    ssim_values = []
    batch_size = img1_np.shape[0]
    for i in range(batch_size):
        img1_i = np.squeeze(img1_np[i])
        img2_i = np.squeeze(img2_np[i])
        ssim_value = ssim(img1_i, img2_i, data_range=1.0)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

class MetricCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = get_lpips_model(device)  # Uses the cached model
    
    def calculate_lpips(self, real_img, fake_img):
        """Compute LPIPS distance (lower is better)"""
        # Convert to [0,1] range for LPIPS
        real_img_01 = (real_img + 1) / 2.0
        fake_img_01 = (fake_img + 1) / 2.0

        with torch.no_grad():
            # print(f"Input device: {input_tensor.device}")
            # print(f"Model device: {next(model.parameters()).device}")
            return self.lpips_fn(real_img_01, fake_img_01).mean().item()