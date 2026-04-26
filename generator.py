# generator.py
import torch.nn as nn
from latent_space import LatentSpace

class Generator_PMC(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_latent_spaces=4):
        super(Generator, self).__init__()

        # Initial convolution block (no downsampling)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=3),  # Output: 64 x 128 x 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers (stride=2)
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 64 x 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 32 x 64
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Latent space blocks
        self.latent_spaces = nn.Sequential(
            *[LatentSpace(256) for _ in range(n_latent_spaces)]
        )

        # Upsampling layers (stride=2)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 64 x 128
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 128 x 256
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, output_nc, kernel_size=7, padding=3),  # Output: output_nc x 128 x 256
            nn.Tanh()
        )

    def forward(self, x):
        # Encoding path
        initial = self.initial_conv(x)
        downsampled = self.downsampling(initial)
        latent = self.latent_spaces(downsampled)

        # Decoding path
        upsampled = self.upsampling(latent)
        output = self.output_conv(upsampled)

        return output, latent


import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_BRATS(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_latent_spaces=4):
        super(Generator, self).__init__()
        
        # 1. Initial Block (Reflection Padding prevents edge artifacts)
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 2. Downsampling (Stride=2)
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 3. Latent Space (Purely Convolutional Residual Blocks)
        # Note: Ensure your LatentSpace class has no Linear/Flatten layers
        from latent_space import LatentSpace
        self.latent_blocks = nn.Sequential(
            *[LatentSpace(256) for _ in range(n_latent_spaces)]
        )

        # 4. Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 5. Output
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        orig_size = x.shape[2:] # Save original (H, W)
        
        # Encoding
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Processing
        x = self.latent_blocks(x)
        
        # Decoding
        x = self.up1(x)
        x = self.up2(x)
        
        # FINAL FIX: If the upsampled size is slightly off (common for odd inputs),
        # force it to match the original input size exactly.
        if x.shape[2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
            
        return self.output(x), x