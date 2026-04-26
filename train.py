import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np
import random
from PIL import Image
# Custom modules for metrics and visualization
from metrics import calculate_psnr, calculate_ssim ,MetricCalculator
from visualization import save_images, plot_losses_metrics

def calculate_combined_score(psnr, ssim, lpips):
    """Calculates a single score from three metrics."""
    
    # --- Step 1: Define Normalization Ranges and Weights ---
    # These are hyperparameters you can tune.
    TARGET_RANGES = {
        'psnr': {'min': 20.0, 'max': 35.0},
        'ssim': {'min': 0.7, 'max': 0.95},
        'lpips': {'min': 0.5, 'max': 0.95} # Corresponds to original LPIPS of 0.5 down to 0.05
    }
    WEIGHTS = {
        'psnr': 0.25,  # Pixel-level accuracy
        'ssim': 0.40,  # Structural similarity
        'lpips': 0.35  # Perceptual similarity
    }
    
    # --- Step 2: Invert LPIPS so higher is better ---
    perceptual_score = 1.0 - lpips

    # --- Step 3: Normalize all metrics to a [0, 1] scale ---
    def normalize(value, v_min, v_max):
        value = max(min(value, v_max), v_min)
        return (value - v_min) / (v_max - v_min)

    norm_psnr = normalize(psnr, TARGET_RANGES['psnr']['min'], TARGET_RANGES['psnr']['max'])
    norm_ssim = normalize(ssim, TARGET_RANGES['ssim']['min'], TARGET_RANGES['ssim']['max'])
    norm_lpips = normalize(perceptual_score, TARGET_RANGES['lpips']['min'], TARGET_RANGES['lpips']['max'])
    
    # --- Step 4: Calculate the final weighted score ---
    combined_score = (
        WEIGHTS['psnr'] * norm_psnr +
        WEIGHTS['ssim'] * norm_ssim +
        WEIGHTS['lpips'] * norm_lpips
    )
    
    return combined_score

def get_replacement_percentage(epoch, total_epochs):
    """Calculate the percentage of generated images to use based on current epoch"""
    return min(1.0, epoch / total_epochs)

metric_calculator = MetricCalculator(device='cuda:1')

def weights_init(m):
    """Initialize weights using He (Kaiming) initialization suitable for grayscale MRI images."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_loss_optimizers_pmc(G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, D_T1, D_T2, D_PD, lr_G_T1_to_T2, lr_G_T2_to_PD ,lr_G_PD_to_T1, lr_D_T1, lr_D_T2, lr_D_PD):
    """
    Initializes loss functions and optimizers.
    """
    # Basic LSGAN loss for adversarial training
    criterion_GAN = nn.MSELoss()

    # Cycle-consistency and identity losses (L1 loss)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Feature matching: L1 loss
    criterion_feature_matching = nn.L1Loss()


    # Optimizers for Generators
    optimizer_G_T1_to_T2 = optim.Adam(G_T1_to_T2.parameters(), lr=lr_G_T1_to_T2, betas=(0.5, 0.999))
    optimizer_G_T2_to_PD = optim.Adam(G_T2_to_PD.parameters(), lr=lr_G_T2_to_PD, betas=(0.5, 0.999))
    optimizer_G_PD_to_T1 = optim.Adam(G_PD_to_T1.parameters(), lr=lr_G_PD_to_T1, betas=(0.5, 0.999))

    # Optimizers for Discriminators
    optimizer_D_T1 = optim.Adam(D_T1.parameters(), lr=lr_D_T1, betas=(0.5, 0.999))
    optimizer_D_T2 = optim.Adam(D_T2.parameters(), lr=lr_D_T2, betas=(0.5, 0.999))
    optimizer_D_PD = optim.Adam(D_PD.parameters(), lr=lr_D_PD, betas=(0.5, 0.999))

    optimizer_G = {
        "G_T1_to_T2": optimizer_G_T1_to_T2,
        "G_T2_to_PD": optimizer_G_T2_to_PD,
        "G_PD_to_T1": optimizer_G_PD_to_T1,
    }
    optimizer_D = {
        "D_T1": optimizer_D_T1,
        "D_T2": optimizer_D_T2,
        "D_PD": optimizer_D_PD,
    }
    

    # Learning rate schedulers (example: linear decay after epoch 200)
    scheduler_G_T1_to_T2 = optim.lr_scheduler.LambdaLR(
        optimizer_G_T1_to_T2, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_G_T2_to_PD = optim.lr_scheduler.LambdaLR(
        optimizer_G_T2_to_PD, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_G_PD_to_T1 = optim.lr_scheduler.LambdaLR(
        optimizer_G_PD_to_T1, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    
    scheduler_D_T1 = optim.lr_scheduler.LambdaLR(
        optimizer_D_T1, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_D_T2 = optim.lr_scheduler.LambdaLR(
        optimizer_D_T2, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_D_PD = optim.lr_scheduler.LambdaLR(
        optimizer_D_PD, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    
    scheduler_G = {
        "G_T1_to_T2": scheduler_G_T1_to_T2,
        "G_T2_to_PD": scheduler_G_T2_to_PD,
        "G_PD_to_T1": scheduler_G_PD_to_T1,
    }
    scheduler_D = {
        "D_T1": scheduler_D_T1,
        "D_T2": scheduler_D_T2,
        "D_PD": scheduler_D_PD,
    }


    return (criterion_GAN, criterion_cycle, criterion_identity,
            criterion_feature_matching,
            optimizer_G, optimizer_D, scheduler_G, scheduler_D)


def validate(val_loader, G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, 
             D_T2, D_PD, D_T1, criterion_GAN, criterion_cycle, 
             criterion_identity, criterion_feature_matching,
             lambda_dict, device, images_dir, epoch):
    G_T1_to_T2.eval()
    G_T2_to_PD.eval()
    G_PD_to_T1.eval()
    D_T2.eval()
    D_PD.eval()
    D_T1.eval()

    total_G_loss_val = 0.0
    total_D_loss_val = 0.0


    #FROM T1
    total_psnr_T2_T1 = 0.0
    total_ssim_T2_T1 = 0.0
    total_lpips_T2_T1 = 0.0
    total_psnr_PD_T1 = 0.0
    total_ssim_PD_T1 = 0.0
    total_lpips_PD_T1 = 0.0

    #FROM PD
    total_psnr_T1_PD = 0.0
    total_ssim_T1_PD = 0.0
    total_lpips_T1_PD = 0.0
    total_psnr_T2_PD = 0.0
    total_ssim_T2_PD = 0.0
    total_lpips_T2_PD = 0.0

    #FROM T2
    total_psnr_T1_T2 = 0.0
    total_ssim_T1_T2 = 0.0
    total_lpips_T1_T2 = 0.0
    total_psnr_PD_T2 = 0.0
    total_ssim_PD_T2 = 0.0
    total_lpips_PD_T2 = 0.0

    num_batches_val = 0

    # lambda_feature_matching = lambda_dict.get('lambda_feature_matching', 10.0)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue

            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)
            real_PD = batch["pd"].to(device)

            # Generate images
            # Forward cycle: T1 → T2 → PD → T1
            fake_T2, _ = G_T1_to_T2(real_T1)
            fake_PD, _ = G_T2_to_PD(fake_T2) 
            cycled_T1, _ = G_PD_to_T1(fake_PD)

            # Backward cycle: PD → T1 → T2 → PD
            fake_T1_from_PD, _ = G_PD_to_T1(real_PD)
            fake_T2_from_T1, _ = G_T1_to_T2(fake_T1_from_PD)
            cycled_PD, _ = G_T2_to_PD(fake_T2_from_T1)

            # CYCLE FOR T2 T2-->PD-->T1-->T2
            fake_PD_from_T2_inT2, _ = G_T2_to_PD(real_T2)
            fake_T1_from_PD_inT2, _ = G_PD_to_T1(fake_PD_from_T2_inT2)
            cycled_T2_inT2, _ = G_T1_to_T2(fake_T1_from_PD_inT2)

            # Compute cycle loss for T1 (T1→T2→T1) and direct reconstruction loss for T2
            loss_cycle_T1 = criterion_cycle(cycled_T1, real_T1) * lambda_dict['lambda_cycle_T1']
            loss_cycle_PD = criterion_cycle(cycled_PD, real_PD) * lambda_dict['lambda_cycle_PD']
            loss_cycle_T2 = criterion_cycle(cycled_T2_inT2, real_T2) * lambda_dict['lambda_cycle_T2']


            # Identity losses for structure preservation
            idt_T2, _ = G_T1_to_T2(real_T2)
            idt_PD, _ = G_T2_to_PD(real_PD)
            idt_T1, _ = G_PD_to_T1(real_T1)


            # Identity losses
            loss_idt_T2 = criterion_identity(idt_T2, real_T2)* lambda_dict['lambda_identity_T2']
            loss_idt_PD = criterion_identity(idt_PD, real_PD)* lambda_dict['lambda_identity_PD']
            loss_idt_T1 = criterion_identity(idt_T1, real_T1)* lambda_dict['lambda_identity_T1']

            loss_identity = loss_idt_T1 + loss_idt_T2+loss_idt_PD

            # GAN losses
            pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
            loss_GAN_T1_to_T2 = criterion_GAN(
                pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
            ) * lambda_dict['lambda_GAN_T1_T2']

            pred_fake_PD, fake_features_PD = D_PD(fake_PD_from_T2_inT2)
            loss_GAN_T2_to_PD = criterion_GAN(
                pred_fake_PD, torch.ones_like(pred_fake_PD, device=device)
            ) * lambda_dict['lambda_GAN_T2_PD']

            pred_fake_T1, fake_features_T1 = D_T1(fake_T1_from_PD)
            loss_GAN_PD_to_T1 = criterion_GAN(
                pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
                )* lambda_dict['lambda_GAN_PD_T1']

            loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_PD + loss_GAN_PD_to_T1 + loss_cycle_T1 + loss_identity + loss_cycle_PD +loss_cycle_T2

            # Feature Matching Loss
            pred_real_T2, real_features_T2 = D_T2(real_T2)
            pred_real_T1, real_features_T1 = D_T1(real_T1)
            pred_real_PD, real_features_PD = D_PD(real_PD)

            loss_fm_T2 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T2, fake_features_T2)
            )
            loss_fm_T1 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T1, fake_features_T1)
            )
            loss_fm_PD = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_PD, fake_features_PD)
            )
            loss_feature_matching = loss_fm_T2*lambda_dict['lambda_fm_D_T2']+ loss_fm_T1*lambda_dict['lambda_fm_D_T1'] +loss_fm_PD*lambda_dict['lambda_fm_D_PD']
            loss_G += loss_feature_matching

            # Discriminator losses
           
            pred_real_T2, _ = D_T2(real_T2)
            loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
            pred_fake_T2, _ = D_T2(fake_T2.detach())
            loss_D_fake_T2 = criterion_GAN(pred_fake_T2, torch.zeros_like(pred_fake_T2,device=device))
            loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5
            
            pred_real_PD, _ = D_PD(real_PD)
            loss_D_real_PD = criterion_GAN(pred_real_PD, torch.ones_like(pred_real_PD,device=device))
            pred_fake_PD, _ = D_PD(fake_PD_from_T2_inT2.detach())
            loss_D_fake_PD = criterion_GAN(pred_fake_PD, torch.zeros_like(pred_fake_PD,device=device))
            loss_D_PD = (loss_D_real_PD + loss_D_fake_PD) * 0.5
            
            pred_real_T1, _ = D_T1(real_T1)
            loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1,device=device))
            pred_fake_T1, _ = D_T1(fake_T1_from_PD.detach())
            loss_D_fake_T1 = criterion_GAN(pred_fake_T1, torch.zeros_like(pred_fake_T1,device=device))
            loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5
            
            loss_D = loss_D_T2 + loss_D_PD + loss_D_T1
            

            total_G_loss_val += loss_G.item()
            total_D_loss_val += loss_D.item()

            # Compute PSNR, SSIM ,LPIPS metrics

            #FROM T1
            psnr_T2_T1 = calculate_psnr(real_T2, fake_T2)
            ssim_T2_T1 = calculate_ssim(real_T2, fake_T2)
            lpips_T2_T1 = metric_calculator.calculate_lpips(real_T2, fake_T2)
            psnr_PD_T1 = calculate_psnr(real_PD, fake_PD)
            ssim_PD_T1 = calculate_ssim(real_PD, fake_PD)
            lpips_PD_T1 = metric_calculator.calculate_lpips(real_PD, fake_PD)


            #FROM T2
            psnr_T1_T2 = calculate_psnr(real_T1, fake_T1_from_PD_inT2)
            ssim_T1_T2 = calculate_ssim(real_T1, fake_T1_from_PD_inT2)
            lpips_T1_T2 = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD_inT2)
            psnr_PD_T2 = calculate_psnr(real_PD, fake_PD_from_T2_inT2)
            ssim_PD_T2 = calculate_ssim(real_PD, fake_PD_from_T2_inT2)
            lpips_PD_T2 = metric_calculator.calculate_lpips(real_PD, fake_PD_from_T2_inT2)

            #FROM PD
            psnr_T2_PD = calculate_psnr(real_T2, fake_T2_from_T1)
            ssim_T2_PD = calculate_ssim(real_T2, fake_T2_from_T1)
            lpips_T2_PD = metric_calculator.calculate_lpips(real_T2, fake_T2_from_T1)
            psnr_T1_PD = calculate_psnr(real_T1, fake_T1_from_PD)
            ssim_T1_PD = calculate_ssim(real_T1, fake_T1_from_PD)
            lpips_T1_PD = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD)
            
           #FROM T1
            total_psnr_T2_T1 += psnr_T2_T1
            total_ssim_T2_T1 += ssim_T2_T1
            total_lpips_T2_T1 += lpips_T2_T1
            total_psnr_PD_T1 += psnr_PD_T1
            total_ssim_PD_T1 += ssim_PD_T1
            total_lpips_PD_T1 += lpips_PD_T1

            #FROM T2
            total_psnr_T1_T2 += psnr_T1_T2
            total_ssim_T1_T2 += ssim_T1_T2
            total_lpips_T1_T2 += lpips_T1_T2
            total_psnr_PD_T2 += psnr_PD_T2
            total_ssim_PD_T2 += ssim_PD_T2
            total_lpips_PD_T2 += lpips_PD_T2

            #FROM PD
            total_psnr_T1_PD += psnr_T1_PD
            total_ssim_T1_PD += ssim_T1_PD
            total_lpips_T1_PD += lpips_T1_PD
            total_psnr_T2_PD += psnr_T2_PD
            total_ssim_T2_PD += ssim_T2_PD
            total_lpips_T2_PD += lpips_T2_PD



            num_batches_val += 1

            # Save example images for qualitative inspection
            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'val')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, real_T2, real_PD,fake_T2,fake_PD,cycled_T1,fake_T1_from_PD,fake_T2_from_T1,cycled_PD,fake_PD_from_T2_inT2,fake_T1_from_PD_inT2,cycled_T2_inT2, epoch_images_dir, epoch + 1, i, dataset_type='val')
            if i % 10 == 0:
                logging.info(f"Validation Batch {i+1}/{len(val_loader)}: "
                             f"G Loss: {loss_G.item():.4f}, D Loss: {loss_D.item():.4f}, "
                             f"GENERATED FROM T1"
                             f"     PSNR T2: {psnr_T2_T1:.4f}, SSIM T2: {ssim_T2_T1:.4f}, LPIPS T2: {lpips_T2_T1:.4f}, "
                             f"     PSNR PD: {psnr_PD_T1:.4f}, SSIM PD: {ssim_PD_T1:.4f}, LPIPS PD: {lpips_PD_T1:.4f}"
                             f"GENERATED FROM T2"
                             f"     PSNR T1: {psnr_T1_T2:.4f}, SSIM T1: {ssim_T1_T2:.4f}, LPIPS T1: {lpips_T1_T2:.4f}, "
                            # f"     PSNR T2: {psnr_T2_T2:.4f}, SSIM T2: {ssim_T2_T2:.4f}, LPIPS T2: {lpips_T2_T2:.4f}, "
                             f"     PSNR PD: {psnr_PD_T2:.4f}, SSIM PD: {ssim_PD_T2:.4f}, LPIPS PD: {lpips_PD_T2:.4f}"
                             f"GENERATED FROM PD"
                             f"     PSNR T1: {psnr_T1_PD:.4f}, SSIM T1: {ssim_T1_PD:.4f}, LPIPS T1: {lpips_T1_PD:.4f}, "
                             f"     PSNR T2: {psnr_T2_PD:.4f}, SSIM T2: {ssim_T2_PD:.4f}, LPIPS T2: {lpips_T2_PD:.4f}, ")
                            # f"     PSNR PD: {psnr_PD:.4f}, SSIM PD: {ssim_PD:.4f}, LPIPS PD: {lpips_PD:.4f}")   
    
    avg_G_loss_val = total_G_loss_val / num_batches_val if num_batches_val > 0 else 0
    avg_D_loss_val = total_D_loss_val / num_batches_val if num_batches_val > 0 else 0
    

    avg_psnr_T2_T1 = total_psnr_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T2_T1 = total_ssim_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T2_T1 = total_lpips_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_PD_T1 = total_psnr_PD_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_PD_T1 = total_ssim_PD_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_PD_T1 = total_lpips_PD_T1 / num_batches_val if num_batches_val > 0 else 0

    avg_psnr_T1_T2 = total_psnr_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T1_T2 = total_ssim_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T1_T2 = total_lpips_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_PD_T2 = total_psnr_PD_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_PD_T2 = total_ssim_PD_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_PD_T2 = total_lpips_PD_T2 / num_batches_val if num_batches_val > 0 else 0

    avg_psnr_T1_PD = total_psnr_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T1_PD = total_ssim_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T1_PD = total_lpips_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_T2_PD= total_psnr_T2_PD / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T2_PD = total_ssim_T2_PD / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T2_PD = total_lpips_T2_PD / num_batches_val if num_batches_val > 0 else 0
    

   

    return (avg_G_loss_val, avg_D_loss_val,
    avg_psnr_T2_T1 ,avg_ssim_T2_T1 ,avg_lpips_T2_T1 ,avg_psnr_PD_T1 ,avg_ssim_PD_T1 ,avg_lpips_PD_T1,
    avg_psnr_T1_T2 ,avg_ssim_T1_T2 ,avg_lpips_T1_T2 ,avg_psnr_PD_T2 ,avg_ssim_PD_T2 ,avg_lpips_PD_T2,
    avg_psnr_T1_PD ,avg_ssim_T1_PD ,avg_lpips_T1_PD ,avg_psnr_T2_PD ,avg_ssim_T2_PD ,avg_lpips_T2_PD
    )

best_combined_score = 0.0

def train_model_pmc(n_epochs, train_loader, val_loader, 
                G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, 
                D_T1, D_T2, D_PD, device,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching,
                optimizer_G, optimizer_D, scheduler_G, scheduler_D, 
                experiment_dir, lambda_dict, metric_calculator):

    total_epochs = n_epochs

    images_dir = os.path.join(experiment_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    scaler = torch.amp.GradScaler(device=device)

    best_ssim = 0.0
    best_psnr = 0.0
    best_lpips = 0.0
    best_combined_score = 0.0

    lambda_fm_D_T1 = lambda_dict.get('lambda_fm_D_T1', 10.0)
    lambda_fm_D_PD = lambda_dict.get('lambda_fm_D_PD', 10.0)
    lambda_fm_D_T2 = lambda_dict.get('lambda_fm_D_T2', 10.0)

    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")
        epch = int(total_epochs-50)
        current_replacement_pct = get_replacement_percentage(epoch, epch)
        logging.info(f"Epoch {epoch+1}: Using {current_replacement_pct*100:.1f}% generated images")


        G_T1_to_T2.train()
        G_T2_to_PD.train()    
        G_PD_to_T1.train()
        D_T1.train()
        D_T2.train()
        D_PD.train()

        total_G_loss = 0.0
        total_D_loss = 0.0

        
        epoch_psnr_T2_T1 = 0.0
        epoch_ssim_T2_T1 = 0.0
        epoch_lpips_T2_T1 = 0.0
        epoch_psnr_PD_T1 = 0.0
        epoch_ssim_PD_T1 = 0.0
        epoch_lpips_PD_T1 = 0.0

        epoch_psnr_T1_T2 = 0.0
        epoch_ssim_T1_T2 = 0.0
        epoch_lpips_T1_T2 = 0.0
        epoch_psnr_PD_T2 = 0.0
        epoch_ssim_PD_T2 = 0.0
        epoch_lpips_PD_T2 = 0.0

        epoch_psnr_T1_PD = 0.0
        epoch_ssim_T1_PD = 0.0
        epoch_lpips_T1_PD = 0.0
        epoch_psnr_T2_PD = 0.0
        epoch_ssim_T2_PD = 0.0
        epoch_lpips_T2_PD = 0.0
       

        num_batches = 0

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue

            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)
            real_PD = batch["pd"].to(device)

            for opt in optimizer_G.values():
                opt.zero_grad()
            for optD in optimizer_D.values():
                optD.zero_grad()
            

            try:
                with autocast():


                    # Forward cycle: T1 → T2 → PD → T1
                    fake_T2, _ = G_T1_to_T2(real_T1)
                    fake_T2_replaced = fake_T2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T2_replaced = torch.where(replace_mask[:, None, None, None], fake_T2, real_T2)
                    
                    fake_PD, _ = G_T2_to_PD(fake_T2_replaced)
                    fake_PD_replaced = fake_PD
                    
                    # Replace some fake_PD with real_PD based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_PD.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_PD_replaced = torch.where(replace_mask[:, None, None, None], fake_PD, real_PD)
                    
                    cycled_T1, _ = G_PD_to_T1(fake_PD_replaced)
                    


                    # Backward cycle: PD → T1 → T2 → PD
                    fake_T1_from_PD, _ = G_PD_to_T1(real_PD)
                    fake_T1_from_PD_replaced = fake_T1_from_PD
                    
                    # Replace some fake_T1_from_PD with real_T1 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T1_from_PD.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T1_from_PD_replaced = torch.where(replace_mask[:, None, None, None], fake_T1_from_PD, real_T1)
                    
                    fake_T2_from_T1, _ = G_T1_to_T2(fake_T1_from_PD_replaced)
                    fake_T2_from_T1_replaced = fake_T2_from_T1
                    
                    # Replace some fake_T2_from_T1 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T2_from_T1.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T2_from_T1_replaced = torch.where(replace_mask[:, None, None, None], fake_T2_from_T1, real_T2)
                    
                    cycled_PD, _ = G_T2_to_PD(fake_T2_from_T1_replaced)
                    


                    # CYCLE FOR T2 T2-->PD-->T1-->T2

                    fake_PD_from_T2_inT2, _ = G_T2_to_PD(real_T2)
                    fake_PD_from_T2_inT2_replaced = fake_PD_from_T2_inT2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_PD_from_T2_inT2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_PD_from_T2_inT2_replaced = torch.where(replace_mask[:, None, None, None], fake_PD_from_T2_inT2, real_T2)
                    

                    fake_T1_from_PD_inT2, _ = G_PD_to_T1(fake_PD_from_T2_inT2_replaced)
                    fake_T1_from_PD_inT2_replaced = fake_T1_from_PD_inT2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T1_from_PD_inT2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T1_from_PD_inT2_replaced = torch.where(replace_mask[:, None, None, None], fake_T1_from_PD_inT2, real_T2)
                    
                    
                    cycled_T2_inT2, _ = G_T1_to_T2(fake_T1_from_PD_inT2_replaced)



                    # Compute cycle consistency loss for T1 and direct reconstruction loss for T2
                    loss_cycle_T1 = criterion_cycle(cycled_T1, real_T1) * lambda_dict['lambda_cycle_T1']
                    loss_cycle_PD = criterion_cycle(cycled_PD, real_PD) * lambda_dict['lambda_cycle_PD']
                    loss_cycle_T2 = criterion_cycle(cycled_T2_inT2, real_T2) * lambda_dict['lambda_cycle_T2']


                    # Identity losses
                    idt_T1, _ = G_PD_to_T1(real_T1)
                    loss_idt_T1 = criterion_identity(idt_T1, real_T1) * lambda_dict['lambda_identity_T1']
                    idt_T2, _ = G_T1_to_T2(real_T2)
                    loss_idt_T2 = criterion_identity(idt_T2, real_T2) * lambda_dict['lambda_identity_T2']
                    idt_PD, _ = G_T2_to_PD(real_PD)
                    loss_idt_PD = criterion_identity(idt_PD, real_PD) * lambda_dict['lambda_identity_PD']
                
                    loss_identity = loss_idt_T1 + loss_idt_T2 + loss_idt_PD

                    # GAN losses for both generators
                    pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
                    loss_GAN_T1_to_T2 = criterion_GAN(
                        pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
                    ) * lambda_dict['lambda_GAN_T1_T2']


                    pred_fake_PD, fake_features_PD = D_PD(fake_PD_from_T2_inT2)
                    loss_GAN_T2_to_PD = criterion_GAN(
                        pred_fake_PD, torch.ones_like(pred_fake_PD, device=device)
                    ) * lambda_dict['lambda_GAN_T2_PD']
                    

                    pred_fake_T1, fake_features_T1 = D_T1(fake_T1_from_PD)
                    loss_GAN_PD_to_T1 = criterion_GAN(
                        pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
                        )* lambda_dict['lambda_GAN_PD_T1']

                

                    loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_PD + loss_GAN_PD_to_T1 + loss_cycle_T1 + loss_identity + loss_cycle_PD + loss_cycle_T2

                    # Feature Matching Loss
                    pred_real_T2, real_features_T2 = D_T2(real_T2)
                    pred_real_PD, real_features_PD = D_PD(real_PD)
                    pred_real_T1, real_features_T1 = D_T1(real_T1)

                    loss_fm_T2 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T2, fake_features_T2)
                    )
                    loss_fm_PD = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_PD, fake_features_PD)
                    )
                    loss_fm_T1 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T1, fake_features_T1)
                    )

                    loss_feature_matching = loss_fm_T2*lambda_fm_D_T2+ loss_fm_T1*lambda_fm_D_T1 + loss_fm_PD*lambda_fm_D_PD

                    loss_G += loss_feature_matching

            except Exception as e:
                logging.error(f"Error during generator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_G).backward()
                for optimizer in optimizer_G.values():
                    scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during generator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                with autocast():
                    # Discriminator loss for T2
                    pred_real_T2, _ = D_T2(real_T2)
                    loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
                    
                    pred_fake_T2, _ = D_T2(fake_T2.detach())
                    loss_D_fake_T2 = criterion_GAN(pred_fake_T2, torch.zeros_like(pred_fake_T2, device=device))
                    loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5
                   

                    # Discriminator PD
                    pred_real_PD, _ = D_PD(real_PD)
                    loss_D_real_PD = criterion_GAN(pred_real_PD, torch.ones_like(pred_real_PD, device=device))
                    
                    pred_fake_PD, _ = D_PD(fake_PD_from_T2_inT2.detach())
                    loss_D_fake_PD = criterion_GAN(pred_fake_PD, torch.zeros_like(pred_fake_PD, device=device))
                    loss_D_PD = (loss_D_real_PD + loss_D_fake_PD) * 0.5
                    
                    # Discriminator T1
                    pred_real_T1, _ = D_T1(real_T1)
                    loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1, device=device))
                    
                    pred_fake_T1, _ = D_T1(fake_T1_from_PD.detach())
                    loss_D_fake_T1 = criterion_GAN(pred_fake_T1, torch.zeros_like(pred_fake_T1, device=device))
                    loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5
                    
                    loss_D = loss_D_T2 + loss_D_PD + loss_D_T1
                
            except Exception as e:
                logging.error(f"Error during discriminator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_D).backward()
                for optimizer in optimizer_D.values():
                    scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during discriminator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()

            # fake_T2, _ = G_T1_to_T2(real_T1)
            # fake_PD, _ = G_T2_to_PD(fake_T2)
            # cycled_T1, _ = G_PD_to_T1(fake_PD)

            #numbers from T1 
            psnr_T2_T1 = calculate_psnr(real_T2, fake_T2)
            ssim_T2_T1 = calculate_ssim(real_T2, fake_T2)
            lpips_T2_T1 = metric_calculator.calculate_lpips(real_T2, fake_T2)
            psnr_PD_T1 = calculate_psnr(real_PD, fake_PD)
            ssim_PD_T1 = calculate_ssim(real_PD, fake_PD)
            lpips_PD_T1 = metric_calculator.calculate_lpips(real_PD, fake_PD)

            #NUMBERS FROM T2
            psnr_T1_T2 = calculate_psnr(real_T1, fake_T1_from_PD_inT2)
            ssim_T1_T2 = calculate_ssim(real_T1, fake_T1_from_PD_inT2)
            lpips_T1_T2 = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD_inT2)
            psnr_PD_T2 = calculate_psnr(real_PD, fake_PD_from_T2_inT2)
            ssim_PD_T2 = calculate_ssim(real_PD, fake_PD_from_T2_inT2)
            lpips_PD_T2 = metric_calculator.calculate_lpips(real_PD, fake_PD_from_T2_inT2)

            #NUMBERRS FROM PD
            psnr_T1_PD = calculate_psnr(real_T1, fake_T1_from_PD)
            ssim_T1_PD = calculate_ssim(real_T1, fake_T1_from_PD)
            lpips_T1_PD = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD)
            psnr_T2_PD = calculate_psnr(real_T2, fake_T2_from_T1)
            ssim_T2_PD = calculate_ssim(real_T2, fake_T2_from_T1)
            lpips_T2_PD = metric_calculator.calculate_lpips(real_T2, fake_T2_from_T1)            

            
            #FOR T1
            epoch_psnr_T2_T1 += psnr_T2_T1
            epoch_ssim_T2_T1 += ssim_T2_T1
            epoch_lpips_T2_T1 += lpips_T2_T1
            epoch_psnr_PD_T1 += psnr_PD_T1
            epoch_ssim_PD_T1 += ssim_PD_T1
            epoch_lpips_PD_T1 += lpips_PD_T1
            
            #FOR T2
            epoch_psnr_T1_T2 += psnr_T1_T2
            epoch_ssim_T1_T2 += ssim_T1_T2
            epoch_lpips_T1_T2 += lpips_T1_T2
            epoch_psnr_PD_T2 += psnr_PD_T2
            epoch_ssim_PD_T2 += ssim_PD_T2
            epoch_lpips_PD_T2 += lpips_PD_T2

            #FOR PD
            epoch_psnr_T1_PD += psnr_T1_PD
            epoch_ssim_T1_PD += ssim_T1_PD
            epoch_lpips_T1_PD += lpips_T1_PD
            epoch_psnr_T2_PD += psnr_T2_PD
            epoch_ssim_T2_PD += ssim_T2_PD
            epoch_lpips_T2_PD += lpips_T2_PD


            num_batches += 1

            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'train')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, real_T2, real_PD,fake_T2,fake_PD,cycled_T1,fake_T1_from_PD,fake_T2_from_T1,cycled_PD,fake_PD_from_T2_inT2,fake_T1_from_PD_inT2,cycled_T2_inT2
                           ,epoch_images_dir, epoch + 1, i, dataset_type='train')

        avg_G_loss = total_G_loss / num_batches if num_batches > 0 else 0
        avg_D_loss = total_D_loss / num_batches if num_batches > 0 else 0
        
        #FROM T1
        avg_psnr_T2_T1 = epoch_psnr_T2_T1 / num_batches if num_batches > 0 else 0
        avg_ssim_T2_T1 = epoch_ssim_T2_T1 / num_batches if num_batches > 0 else 0
        avg_lpips_T2_T1 = epoch_lpips_T2_T1 / num_batches if num_batches > 0 else 0
        avg_psnr_PD_T1 = epoch_psnr_PD_T1 / num_batches if num_batches > 0 else 0
        avg_ssim_PD_T1 = epoch_ssim_PD_T1 / num_batches if num_batches > 0 else 0
        avg_lpips_PD_T1 = epoch_lpips_PD_T1 / num_batches if num_batches > 0 else 0
        
        #FROM T2
        avg_psnr_T1_T2 = epoch_psnr_T1_T2 / num_batches if num_batches > 0 else 0
        avg_ssim_T1_T2 = epoch_ssim_T1_T2 / num_batches if num_batches > 0 else 0
        avg_lpips_T1_T2 = epoch_lpips_T1_T2 / num_batches if num_batches > 0 else 0
        avg_psnr_PD_T2 = epoch_psnr_PD_T2 / num_batches if num_batches > 0 else 0
        avg_ssim_PD_T2 = epoch_ssim_PD_T2 / num_batches if num_batches > 0 else 0
        avg_lpips_PD_T2 = epoch_lpips_PD_T2 / num_batches if num_batches > 0 else 0

        #FROM PD
        avg_psnr_T1_PD = epoch_psnr_T1_PD / num_batches if num_batches > 0 else 0
        avg_ssim_T1_PD = epoch_ssim_T1_PD / num_batches if num_batches > 0 else 0
        avg_lpips_T1_PD = epoch_lpips_T1_PD / num_batches if num_batches > 0 else 0
        avg_psnr_T2_PD = epoch_psnr_T2_PD / num_batches if num_batches > 0 else 0
        avg_ssim_T2_PD = epoch_ssim_T2_PD / num_batches if num_batches > 0 else 0
        avg_lpips_T2_PD = epoch_lpips_T2_PD / num_batches if num_batches > 0 else 0


        logging.info(f"Epoch {epoch + 1} Training Losses: Gen {avg_G_loss:.4f}, Dis {avg_D_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} Training Metrics:")
        logging.info(f"Metrics from T1")
        logging.info(f"  T2: PSNR {avg_psnr_T2_T1:.4f}, SSIM {avg_ssim_T2_T1:.4f}, LPIPS {avg_lpips_T2_T1:.4f}")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T1:.4f}, SSIM {avg_ssim_PD_T1:.4f}, LPIPS {avg_lpips_PD_T1:.4f}")
        logging.info(f"Metrics from T2")
        logging.info(f"  T1: PSNR {avg_psnr_T1_T2:.4f}, SSIM {avg_ssim_T1_T2:.4f}, LPIPS {avg_lpips_T1_T2:.4f}")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T2:.4f}, SSIM {avg_ssim_PD_T2:.4f}, LPIPS {avg_lpips_PD_T2:.4f}")
        logging.info(f"Metrics from PD")
        logging.info(f"  T1: PSNR {avg_psnr_T1_PD:.4f}, SSIM {avg_ssim_T1_PD:.4f}, LPIPS {avg_lpips_T1_PD:.4f}")
        logging.info(f"  T2: PSNR {avg_psnr_T2_PD:.4f}, SSIM {avg_ssim_T2_PD:.4f}, LPIPS {avg_lpips_T2_PD:.4f}")




        train_losses.append({
            'gen_total_loss_train': avg_G_loss,
            'dis_total_loss_train': avg_D_loss,
            'cycle_loss_T1_train': loss_cycle_T1.item() if num_batches > 0 else 0,
            'cycle_loss_PD_train':loss_cycle_PD.item() if num_batches > 0 else 0,
            'cycle_loss_T2_train':loss_cycle_T2.item() if num_batches > 0 else 0,
            'identity_loss_train': loss_identity.item() if num_batches > 0 else 0,
            'feature_matching_loss_train': loss_feature_matching.item() if num_batches > 0 else 0,
            'gan_loss_T1_to_T2_train': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
            'gan_loss_T2_to_PD_train': loss_GAN_T2_to_PD.item() if num_batches > 0 else 0,
            'gan_loss_PD_to_T1_train': loss_GAN_PD_to_T1.item() if num_batches > 0 else 0,
            'dis_loss_T2_train':loss_D_T2.item() if num_batches > 0 else 0,
            'dis_loss_PD_train':loss_D_PD.item() if num_batches > 0 else 0,
            'dis_loss_T1_train':loss_D_T1.item() if num_batches > 0 else 0

        })

        train_metrics.append({
            
            'PSNR_T2_train_T1': avg_psnr_T2_T1,
            'SSIM_T2_train_T1': avg_ssim_T2_T1,
            'LPIPS_T2_train_T1': avg_lpips_T2_T1,
            'PSNR_PD_train_T1': avg_psnr_PD_T1,
            'SSIM_PD_train_T1': avg_ssim_PD_T1,
            'LPIPS_PD_train_T1': avg_lpips_PD_T1,

            'PSNR_T1_train_T2': avg_psnr_T1_T2,
            'SSIM_T1_train_T2': avg_ssim_T1_T2,
            'LPIPS_T1_train_T2': avg_lpips_T1_T2,
            'PSNR_PD_train_T2': avg_psnr_PD_T2,
            'SSIM_PD_train_T2': avg_ssim_PD_T2,
            'LPIPS_PD_train_T2': avg_lpips_PD_T2,

            'PSNR_T1_train_PD': avg_psnr_T1_PD,
            'SSIM_T1_train_PD': avg_ssim_T1_PD,
            'LPIPS_T1_train_PD': avg_lpips_T1_PD,
            'PSNR_T2_train_PD': avg_psnr_T2_PD,
            'SSIM_T2_train_PD': avg_ssim_T2_PD,
            'LPIPS_T2_train_PD': avg_lpips_T2_PD,
            
        })

        try:
            (avg_G_loss_val, avg_D_loss_val,
             
             avg_psnr_T2_T1_val, avg_ssim_T2_T1_val, avg_lpips_T2_T1_val,
             avg_psnr_PD_T1_val, avg_ssim_PD_T1_val, avg_lpips_PD_T1_val,

             avg_psnr_T1_T2_val, avg_ssim_T1_T2_val, avg_lpips_T1_T2_val,
             avg_psnr_PD_T2_val, avg_ssim_PD_T2_val, avg_lpips_PD_T2_val,

             avg_psnr_T1_PD_val, avg_ssim_T1_PD_val, avg_lpips_T1_PD_val,
             avg_psnr_T2_PD_val, avg_ssim_T2_PD_val, avg_lpips_T2_PD_val,
             ) = validate(
                val_loader, G_T1_to_T2, G_T2_to_PD, G_PD_to_T1,
                D_T2, D_PD, D_T1,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching,
                lambda_dict, device, images_dir, epoch )
        except Exception as e:
            logging.error(f"Error during validation at epoch {epoch+1}: {e}", exc_info=True)
            raise

        logging.info(f"Epoch {epoch + 1} Validation Losses: Gen {avg_G_loss_val:.4f}, Dis {avg_D_loss_val:.4f}\n")
        logging.info(f"Epoch {epoch + 1} Validation Metrics:\n")
        logging.info("generated from T1\n")
        logging.info(f"  T2: PSNR {avg_psnr_T2_T1_val:.4f}, SSIM {avg_ssim_T2_T1_val:.4f}, LPIPS {avg_lpips_T2_T1_val:.4f}\n")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T1_val:.4f}, SSIM {avg_ssim_PD_T1_val:.4f}, LPIPS {avg_lpips_PD_T1_val:.4f}\n")
        logging.info("generated from T2\n")
        logging.info(f"  T1: PSNR {avg_psnr_T1_T2_val:.4f}, SSIM {avg_ssim_T1_T2_val:.4f}, LPIPS {avg_lpips_T1_T2_val:.4f}\n")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T2_val:.4f}, SSIM {avg_ssim_PD_T2_val:.4f}, LPIPS {avg_lpips_PD_T2_val:.4f}\n")
        logging.info("generated from PD\n")
        logging.info(f"  T1: PSNR {avg_psnr_T1_PD_val:.4f}, SSIM {avg_ssim_T1_PD_val:.4f}, LPIPS {avg_lpips_T1_PD_val:.4f}\n")
        logging.info(f"  T2: PSNR {avg_psnr_T2_PD_val:.4f}, SSIM {avg_ssim_T2_PD_val:.4f}, LPIPS {avg_lpips_T2_PD_val:.4f}\n")
        
        val_losses.append({
            'gen_total_loss_val': avg_G_loss_val,
            'dis_total_loss_val': avg_D_loss_val,
            'cycle_loss_T1_val': loss_cycle_T1.item() if num_batches > 0 else 0,
            'cycle_loss_PD_val': loss_cycle_PD.item() if num_batches > 0 else 0,
            'cycle_loss_T2_val': loss_cycle_T2.item() if num_batches > 0 else 0,
            'identity_loss_val': loss_identity.item() if num_batches > 0 else 0,
            'feature_matching_loss_val': loss_feature_matching.item() if num_batches > 0 else 0,
            'gan_loss_T1_to_T2_val': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
            'gan_loss_T2_to_PD_val': loss_GAN_T2_to_PD.item() if num_batches > 0 else 0,
            'gan_loss_PD_to_T1_val': loss_GAN_PD_to_T1.item() if num_batches > 0 else 0,
            'dis_loss_T2_val': loss_D_T2.item() if num_batches > 0 else 0,
            'dis_loss_PD_val': loss_D_PD.item() if num_batches > 0 else 0,
            'dis_loss_T1_val': loss_D_T1.item() if num_batches > 0 else 0
        })
        
        val_metrics.append({
            
            'PSNR_T2_T1_val': avg_psnr_T2_T1_val,
            'SSIM_T2_T1_val': avg_ssim_T2_T1_val,
            'LPIPS_T2_T1_val': avg_lpips_T2_T1_val,
            'PSNR_PD_T1_val': avg_psnr_PD_T1_val,
            'SSIM_PD_T1_val': avg_ssim_PD_T1_val,
            'LPIPS_PD_T1_val': avg_lpips_PD_T1_val,

            'PSNR_T1_T2_val': avg_psnr_T1_T2_val,
            'SSIM_T1_T2_val': avg_ssim_T1_T2_val,
            'LPIPS_T1_T2_val': avg_lpips_T1_T2_val,
            'PSNR_PD_T2_val': avg_psnr_PD_T2_val,
            'SSIM_PD_T2_val': avg_ssim_PD_T2_val,
            'LPIPS_PD_T2_val': avg_lpips_PD_T2_val,

            'PSNR_T1_PD_val': avg_psnr_T1_PD_val,
            'SSIM_T1_PD_val': avg_ssim_T1_PD_val,
            'LPIPS_T1_PD_val': avg_lpips_T1_PD_val,
            'PSNR_T2_PD_val': avg_psnr_T2_PD_val,
            'SSIM_T2_PD_val': avg_ssim_T2_PD_val,
            'LPIPS_T2_PD_val': avg_lpips_T2_PD_val
           

        })

        current_avg_ssim_T1 = (avg_ssim_T2_T1_val + avg_ssim_PD_T1_val) / 2
        current_avg_ssim_T2 = (avg_ssim_T1_T2_val + avg_ssim_PD_T2_val) / 2
        current_avg_ssim_PD = (avg_ssim_T2_PD_val + avg_ssim_T1_PD_val) / 2
        current_avg_ssim = (current_avg_ssim_PD+current_avg_ssim_T1+current_avg_ssim_T2)/3

        current_avg_psnr_T1 = (avg_psnr_T2_T1_val + avg_psnr_PD_T1_val) / 2
        current_avg_psnr_T2 = (avg_psnr_T1_T2_val + avg_psnr_PD_T2_val) / 2
        current_avg_psnr_PD = (avg_psnr_T2_PD_val + avg_psnr_T1_PD_val) / 2
        current_avg_psnr = (current_avg_psnr_PD+current_avg_psnr_T1+current_avg_psnr_T2)/3

        current_avg_lpips_T1 = (avg_lpips_T2_T1_val + avg_lpips_PD_T1_val) / 2
        current_avg_lpips_T2 = (avg_lpips_T1_T2_val + avg_lpips_PD_T2_val) / 2
        current_avg_lpips_PD = (avg_lpips_T2_PD_val + avg_lpips_T1_PD_val) / 2
        current_avg_lpips = (current_avg_lpips_PD+current_avg_lpips_T1+current_avg_lpips_T2)/3
        


        best_lpips = float('inf')
        
        # NEW: Calculate the combined score
        current_score = calculate_combined_score(
            current_avg_psnr, 
            current_avg_ssim, 
            current_avg_lpips
        )

        # Check if this is the best model based on the new combined score
        if current_score > best_combined_score:
            best_combined_score = current_score
            
            logging.info(f"New best model found at epoch {epoch + 1} with Combined Score: {best_combined_score:.4f}")
            logging.info(f"  (PSNR: {current_avg_psnr:.4f}, SSIM: {current_avg_ssim:.4f}, LPIPS: {current_avg_lpips:.4f})")

            model_save_path = os.path.join(experiment_dir, 'best_model-combined.pth')

                
        if current_avg_psnr > best_psnr:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_psnr.pth')

        if current_avg_lpips < best_lpips:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_lpips.pth')

        if current_avg_ssim > best_ssim:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_ssim.pth')


        
        torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_PD': G_T2_to_PD.state_dict(),
                'G_PD_to_T1': G_PD_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'D_PD': D_PD.state_dict(),
                'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
                'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
                'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
                'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
                'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
                'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'best_lpips': best_lpips,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, model_save_path)

        logging.info(f"Best model saved at epoch {epoch + 1} with avg SSIM {best_ssim:.4f} and avg PSNR {best_psnr:.4f} and avg lpips {best_lpips:.4f} based on SSIM")

        for scheduler in scheduler_G.values():
            scheduler.step()
        for scheduler in scheduler_D.values():
            scheduler.step()

        try:
            plot_losses_metrics(train_losses, val_losses, train_metrics, val_metrics, plots_dir)
            logging.info(f"Plots saved for epoch {epoch + 1}.")
        except Exception as e:
            logging.error(f"Error while plotting at epoch {epoch+1}: {e}", exc_info=True)

    return best_ssim, best_psnr, best_lpips


##################################################
# BRATS METHODS 
##################################################

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np
import random
from PIL import Image
# Custom modules for metrics and visualization
from metrics import calculate_psnr, calculate_ssim ,MetricCalculator
from visualization import save_images, plot_losses_metrics

def calculate_combined_score(psnr, ssim, lpips):
    """Calculates a single score from three metrics."""
    
    # --- Step 1: Define Normalization Ranges and Weights ---
    # These are hyperparameters you can tune.
    TARGET_RANGES = {
        'psnr': {'min': 20.0, 'max': 35.0},
        'ssim': {'min': 0.7, 'max': 0.95},
        'lpips': {'min': 0.5, 'max': 0.95} # Corresponds to original LPIPS of 0.5 down to 0.05
    }
    WEIGHTS = {
        'psnr': 0.25,  # Pixel-level accuracy
        'ssim': 0.40,  # Structural similarity
        'lpips': 0.35  # Perceptual similarity
    }
    
    # --- Step 2: Invert LPIPS so higher is better ---
    perceptual_score = 1.0 - lpips

    # --- Step 3: Normalize all metrics to a [0, 1] scale ---
    def normalize(value, v_min, v_max):
        value = max(min(value, v_max), v_min)
        return (value - v_min) / (v_max - v_min)

    norm_psnr = normalize(psnr, TARGET_RANGES['psnr']['min'], TARGET_RANGES['psnr']['max'])
    norm_ssim = normalize(ssim, TARGET_RANGES['ssim']['min'], TARGET_RANGES['ssim']['max'])
    norm_lpips = normalize(perceptual_score, TARGET_RANGES['lpips']['min'], TARGET_RANGES['lpips']['max'])
    
    # --- Step 4: Calculate the final weighted score ---
    combined_score = (
        WEIGHTS['psnr'] * norm_psnr +
        WEIGHTS['ssim'] * norm_ssim +
        WEIGHTS['lpips'] * norm_lpips
    )
    
    return combined_score

def get_replacement_percentage(epoch, total_epochs):
    """Calculate the percentage of generated images to use based on current epoch"""
    return min(1.0, epoch / total_epochs)

metric_calculator = MetricCalculator(device='cuda:2')

def weights_init_brats(m):
    """Initialize weights using He (Kaiming) initialization suitable for grayscale MRI images."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_loss_optimizers_brats(G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, D_T1, D_T2, D_PD, lr_G_T1_to_T2, lr_G_T2_to_PD ,lr_G_PD_to_T1, lr_D_T1, lr_D_T2, lr_D_PD):
    """
    Initializes loss functions and optimizers.
    """
    # Basic LSGAN loss for adversarial training
    criterion_GAN = nn.MSELoss()

    # Cycle-consistency and identity losses (L1 loss)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Feature matching: L1 loss
    criterion_feature_matching = nn.L1Loss()


    # Optimizers for Generators
    optimizer_G_T1_to_T2 = optim.Adam(G_T1_to_T2.parameters(), lr=lr_G_T1_to_T2, betas=(0.5, 0.999))
    optimizer_G_T2_to_PD = optim.Adam(G_T2_to_PD.parameters(), lr=lr_G_T2_to_PD, betas=(0.5, 0.999))
    optimizer_G_PD_to_T1 = optim.Adam(G_PD_to_T1.parameters(), lr=lr_G_PD_to_T1, betas=(0.5, 0.999))

    # Optimizers for Discriminators
    optimizer_D_T1 = optim.Adam(D_T1.parameters(), lr=lr_D_T1, betas=(0.5, 0.999))
    optimizer_D_T2 = optim.Adam(D_T2.parameters(), lr=lr_D_T2, betas=(0.5, 0.999))
    optimizer_D_PD = optim.Adam(D_PD.parameters(), lr=lr_D_PD, betas=(0.5, 0.999))

    optimizer_G = {
        "G_T1_to_T2": optimizer_G_T1_to_T2,
        "G_T2_to_PD": optimizer_G_T2_to_PD,
        "G_PD_to_T1": optimizer_G_PD_to_T1,
    }
    optimizer_D = {
        "D_T1": optimizer_D_T1,
        "D_T2": optimizer_D_T2,
        "D_PD": optimizer_D_PD,
    }
    

    # Learning rate schedulers (example: linear decay after epoch 200)
    scheduler_G_T1_to_T2 = optim.lr_scheduler.LambdaLR(
        optimizer_G_T1_to_T2, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_G_T2_to_PD = optim.lr_scheduler.LambdaLR(
        optimizer_G_T2_to_PD, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_G_PD_to_T1 = optim.lr_scheduler.LambdaLR(
        optimizer_G_PD_to_T1, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    
    scheduler_D_T1 = optim.lr_scheduler.LambdaLR(
        optimizer_D_T1, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_D_T2 = optim.lr_scheduler.LambdaLR(
        optimizer_D_T2, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    scheduler_D_PD = optim.lr_scheduler.LambdaLR(
        optimizer_D_PD, lr_lambda=lambda epoch: max(0.0, 1.0 - epoch / 200.0)
    )
    
    scheduler_G = {
        "G_T1_to_T2": scheduler_G_T1_to_T2,
        "G_T2_to_PD": scheduler_G_T2_to_PD,
        "G_PD_to_T1": scheduler_G_PD_to_T1,
    }
    scheduler_D = {
        "D_T1": scheduler_D_T1,
        "D_T2": scheduler_D_T2,
        "D_PD": scheduler_D_PD,
    }


    return (criterion_GAN, criterion_cycle, criterion_identity,
            criterion_feature_matching,
            optimizer_G, optimizer_D, scheduler_G, scheduler_D)


# Add wandb_run=None at the end
def validate(val_loader, G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, 
             D_T2, D_PD, D_T1, criterion_GAN, criterion_cycle, 
             criterion_identity, criterion_feature_matching,
             lambda_dict, device, images_dir, epoch, wandb_run=None):
    G_T1_to_T2.eval()
    G_T2_to_PD.eval()
    G_PD_to_T1.eval()
    D_T2.eval()
    D_PD.eval()
    D_T1.eval()

    total_G_loss_val = 0.0
    total_D_loss_val = 0.0


    #FROM T1
    total_psnr_T2_T1 = 0.0
    total_ssim_T2_T1 = 0.0
    total_lpips_T2_T1 = 0.0
    total_psnr_PD_T1 = 0.0
    total_ssim_PD_T1 = 0.0
    total_lpips_PD_T1 = 0.0

    #FROM PD
    total_psnr_T1_PD = 0.0
    total_ssim_T1_PD = 0.0
    total_lpips_T1_PD = 0.0
    total_psnr_T2_PD = 0.0
    total_ssim_T2_PD = 0.0
    total_lpips_T2_PD = 0.0

    #FROM T2
    total_psnr_T1_T2 = 0.0
    total_ssim_T1_T2 = 0.0
    total_lpips_T1_T2 = 0.0
    total_psnr_PD_T2 = 0.0
    total_ssim_PD_T2 = 0.0
    total_lpips_PD_T2 = 0.0

    num_batches_val = 0

    # lambda_feature_matching = lambda_dict.get('lambda_feature_matching', 10.0)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue

            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)
            real_PD = batch["flair"].to(device)

            # Generate images
            # Forward cycle: T1 → T2 → PD → T1
            fake_T2, _ = G_T1_to_T2(real_T1)
            fake_PD, _ = G_T2_to_PD(fake_T2) 
            cycled_T1, _ = G_PD_to_T1(fake_PD)

            # Backward cycle: PD → T1 → T2 → PD
            fake_T1_from_PD, _ = G_PD_to_T1(real_PD)
            fake_T2_from_T1, _ = G_T1_to_T2(fake_T1_from_PD)
            cycled_PD, _ = G_T2_to_PD(fake_T2_from_T1)

            # CYCLE FOR T2 T2-->PD-->T1-->T2
            fake_PD_from_T2_inT2, _ = G_T2_to_PD(real_T2)
            fake_T1_from_PD_inT2, _ = G_PD_to_T1(fake_PD_from_T2_inT2)
            cycled_T2_inT2, _ = G_T1_to_T2(fake_T1_from_PD_inT2)

            # Compute cycle loss for T1 (T1→T2→T1) and direct reconstruction loss for T2
            loss_cycle_T1 = criterion_cycle(cycled_T1, real_T1) * lambda_dict['lambda_cycle_T1']
            loss_cycle_PD = criterion_cycle(cycled_PD, real_PD) * lambda_dict['lambda_cycle_PD']
            loss_cycle_T2 = criterion_cycle(cycled_T2_inT2, real_T2) * lambda_dict['lambda_cycle_T2']


            # Identity losses for structure preservation
            idt_T2, _ = G_T1_to_T2(real_T2)
            idt_PD, _ = G_T2_to_PD(real_PD)
            idt_T1, _ = G_PD_to_T1(real_T1)


            # Identity losses
            loss_idt_T2 = criterion_identity(idt_T2, real_T2)* lambda_dict['lambda_identity_T2']
            loss_idt_PD = criterion_identity(idt_PD, real_PD)* lambda_dict['lambda_identity_PD']
            loss_idt_T1 = criterion_identity(idt_T1, real_T1)* lambda_dict['lambda_identity_T1']

            loss_identity = loss_idt_T1 + loss_idt_T2+loss_idt_PD

            # GAN losses
            pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
            loss_GAN_T1_to_T2 = criterion_GAN(
                pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
            ) * lambda_dict['lambda_GAN_T1_T2']

            pred_fake_PD, fake_features_PD = D_PD(fake_PD_from_T2_inT2)
            loss_GAN_T2_to_PD = criterion_GAN(
                pred_fake_PD, torch.ones_like(pred_fake_PD, device=device)
            ) * lambda_dict['lambda_GAN_T2_PD']

            pred_fake_T1, fake_features_T1 = D_T1(fake_T1_from_PD)
            loss_GAN_PD_to_T1 = criterion_GAN(
                pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
                )* lambda_dict['lambda_GAN_PD_T1']

            loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_PD + loss_GAN_PD_to_T1 + loss_cycle_T1 + loss_identity + loss_cycle_PD +loss_cycle_T2

            # Feature Matching Loss
            pred_real_T2, real_features_T2 = D_T2(real_T2)
            pred_real_T1, real_features_T1 = D_T1(real_T1)
            pred_real_PD, real_features_PD = D_PD(real_PD)

            loss_fm_T2 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T2, fake_features_T2)
            )
            loss_fm_T1 = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_T1, fake_features_T1)
            )
            loss_fm_PD = sum(
                criterion_feature_matching(fake, real)
                for real, fake in zip(real_features_PD, fake_features_PD)
            )
            loss_feature_matching = loss_fm_T2*lambda_dict['lambda_fm_D_T2']+ loss_fm_T1*lambda_dict['lambda_fm_D_T1'] +loss_fm_PD*lambda_dict['lambda_fm_D_PD']
            loss_G += loss_feature_matching

            # Discriminator losses
           
            pred_real_T2, _ = D_T2(real_T2)
            loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
            pred_fake_T2, _ = D_T2(fake_T2.detach())
            loss_D_fake_T2 = criterion_GAN(pred_fake_T2, torch.zeros_like(pred_fake_T2,device=device))
            loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5
            
            pred_real_PD, _ = D_PD(real_PD)
            loss_D_real_PD = criterion_GAN(pred_real_PD, torch.ones_like(pred_real_PD,device=device))
            pred_fake_PD, _ = D_PD(fake_PD_from_T2_inT2.detach())
            loss_D_fake_PD = criterion_GAN(pred_fake_PD, torch.zeros_like(pred_fake_PD,device=device))
            loss_D_PD = (loss_D_real_PD + loss_D_fake_PD) * 0.5
            
            pred_real_T1, _ = D_T1(real_T1)
            loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1,device=device))
            pred_fake_T1, _ = D_T1(fake_T1_from_PD.detach())
            loss_D_fake_T1 = criterion_GAN(pred_fake_T1, torch.zeros_like(pred_fake_T1,device=device))
            loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5
            
            loss_D = loss_D_T2 + loss_D_PD + loss_D_T1
            

            total_G_loss_val += loss_G.item()
            total_D_loss_val += loss_D.item()

            # Compute PSNR, SSIM ,LPIPS metrics

            #FROM T1
            psnr_T2_T1 = calculate_psnr(real_T2, fake_T2)
            ssim_T2_T1 = calculate_ssim(real_T2, fake_T2)
            lpips_T2_T1 = metric_calculator.calculate_lpips(real_T2, fake_T2)
            psnr_PD_T1 = calculate_psnr(real_PD, fake_PD)
            ssim_PD_T1 = calculate_ssim(real_PD, fake_PD)
            lpips_PD_T1 = metric_calculator.calculate_lpips(real_PD, fake_PD)


            #FROM T2
            psnr_T1_T2 = calculate_psnr(real_T1, fake_T1_from_PD_inT2)
            ssim_T1_T2 = calculate_ssim(real_T1, fake_T1_from_PD_inT2)
            lpips_T1_T2 = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD_inT2)
            psnr_PD_T2 = calculate_psnr(real_PD, fake_PD_from_T2_inT2)
            ssim_PD_T2 = calculate_ssim(real_PD, fake_PD_from_T2_inT2)
            lpips_PD_T2 = metric_calculator.calculate_lpips(real_PD, fake_PD_from_T2_inT2)

            #FROM PD
            psnr_T2_PD = calculate_psnr(real_T2, fake_T2_from_T1)
            ssim_T2_PD = calculate_ssim(real_T2, fake_T2_from_T1)
            lpips_T2_PD = metric_calculator.calculate_lpips(real_T2, fake_T2_from_T1)
            psnr_T1_PD = calculate_psnr(real_T1, fake_T1_from_PD)
            ssim_T1_PD = calculate_ssim(real_T1, fake_T1_from_PD)
            lpips_T1_PD = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD)
            
           #FROM T1
            total_psnr_T2_T1 += psnr_T2_T1
            total_ssim_T2_T1 += ssim_T2_T1
            total_lpips_T2_T1 += lpips_T2_T1
            total_psnr_PD_T1 += psnr_PD_T1
            total_ssim_PD_T1 += ssim_PD_T1
            total_lpips_PD_T1 += lpips_PD_T1

            #FROM T2
            total_psnr_T1_T2 += psnr_T1_T2
            total_ssim_T1_T2 += ssim_T1_T2
            total_lpips_T1_T2 += lpips_T1_T2
            total_psnr_PD_T2 += psnr_PD_T2
            total_ssim_PD_T2 += ssim_PD_T2
            total_lpips_PD_T2 += lpips_PD_T2

            #FROM PD
            total_psnr_T1_PD += psnr_T1_PD
            total_ssim_T1_PD += ssim_T1_PD
            total_lpips_T1_PD += lpips_T1_PD
            total_psnr_T2_PD += psnr_T2_PD
            total_ssim_T2_PD += ssim_T2_PD
            total_lpips_T2_PD += lpips_T2_PD



            num_batches_val += 1

            # Save example images for qualitative inspection
            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'val')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, real_T2, real_PD,fake_T2,fake_PD,cycled_T1,fake_T1_from_PD,fake_T2_from_T1,cycled_PD,fake_PD_from_T2_inT2,fake_T1_from_PD_inT2,cycled_T2_inT2, epoch_images_dir, epoch + 1, i, dataset_type='val')
            if i % 10 == 0:
                logging.info(f"Validation Batch {i+1}/{len(val_loader)}: "
                             f"G Loss: {loss_G.item():.4f}, D Loss: {loss_D.item():.4f}, "
                             f"GENERATED FROM T1"
                             f"     PSNR T2: {psnr_T2_T1:.4f}, SSIM T2: {ssim_T2_T1:.4f}, LPIPS T2: {lpips_T2_T1:.4f}, "
                             f"     PSNR PD: {psnr_PD_T1:.4f}, SSIM PD: {ssim_PD_T1:.4f}, LPIPS PD: {lpips_PD_T1:.4f}"
                             f"GENERATED FROM T2"
                             f"     PSNR T1: {psnr_T1_T2:.4f}, SSIM T1: {ssim_T1_T2:.4f}, LPIPS T1: {lpips_T1_T2:.4f}, "
                            # f"     PSNR T2: {psnr_T2_T2:.4f}, SSIM T2: {ssim_T2_T2:.4f}, LPIPS T2: {lpips_T2_T2:.4f}, "
                             f"     PSNR PD: {psnr_PD_T2:.4f}, SSIM PD: {ssim_PD_T2:.4f}, LPIPS PD: {lpips_PD_T2:.4f}"
                             f"GENERATED FROM PD"
                             f"     PSNR T1: {psnr_T1_PD:.4f}, SSIM T1: {ssim_T1_PD:.4f}, LPIPS T1: {lpips_T1_PD:.4f}, "
                             f"     PSNR T2: {psnr_T2_PD:.4f}, SSIM T2: {ssim_T2_PD:.4f}, LPIPS T2: {lpips_T2_PD:.4f}, ")
                            # f"     PSNR PD: {psnr_PD:.4f}, SSIM PD: {ssim_PD:.4f}, LPIPS PD: {lpips_PD:.4f}")   
    
    avg_G_loss_val = total_G_loss_val / num_batches_val if num_batches_val > 0 else 0
    avg_D_loss_val = total_D_loss_val / num_batches_val if num_batches_val > 0 else 0
    

    avg_psnr_T2_T1 = total_psnr_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T2_T1 = total_ssim_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T2_T1 = total_lpips_T2_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_PD_T1 = total_psnr_PD_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_PD_T1 = total_ssim_PD_T1 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_PD_T1 = total_lpips_PD_T1 / num_batches_val if num_batches_val > 0 else 0

    avg_psnr_T1_T2 = total_psnr_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T1_T2 = total_ssim_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T1_T2 = total_lpips_T1_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_PD_T2 = total_psnr_PD_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_PD_T2 = total_ssim_PD_T2 / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_PD_T2 = total_lpips_PD_T2 / num_batches_val if num_batches_val > 0 else 0

    avg_psnr_T1_PD = total_psnr_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T1_PD = total_ssim_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T1_PD = total_lpips_T1_PD / num_batches_val if num_batches_val > 0 else 0
    avg_psnr_T2_PD= total_psnr_T2_PD / num_batches_val if num_batches_val > 0 else 0
    avg_ssim_T2_PD = total_ssim_T2_PD / num_batches_val if num_batches_val > 0 else 0
    avg_lpips_T2_PD = total_lpips_T2_PD / num_batches_val if num_batches_val > 0 else 0
    

   

    return (avg_G_loss_val, avg_D_loss_val,
    avg_psnr_T2_T1 ,avg_ssim_T2_T1 ,avg_lpips_T2_T1 ,avg_psnr_PD_T1 ,avg_ssim_PD_T1 ,avg_lpips_PD_T1,
    avg_psnr_T1_T2 ,avg_ssim_T1_T2 ,avg_lpips_T1_T2 ,avg_psnr_PD_T2 ,avg_ssim_PD_T2 ,avg_lpips_PD_T2,
    avg_psnr_T1_PD ,avg_ssim_T1_PD ,avg_lpips_T1_PD ,avg_psnr_T2_PD ,avg_ssim_T2_PD ,avg_lpips_T2_PD
    )



# Add wandb_run=None at the end
def train_model_brats(n_epochs, train_loader, val_loader,  
                G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, 
                D_T1, D_T2, D_PD, device,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching,
                optimizer_G, optimizer_D, scheduler_G, scheduler_D, 
                experiment_dir, lambda_dict, metric_calculator, wandb_run=None):
    total_epochs = n_epochs

    images_dir = os.path.join(experiment_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    scaler = torch.amp.GradScaler(device='cuda')

    best_ssim = 0.0
    best_psnr = 0.0
    best_lpips = 0.0
    best_combined_score = 0.0

    lambda_fm_D_T1 = lambda_dict.get('lambda_fm_D_T1', 10.0)
    lambda_fm_D_PD = lambda_dict.get('lambda_fm_D_PD', 10.0)
    lambda_fm_D_T2 = lambda_dict.get('lambda_fm_D_T2', 10.0)

    for epoch in range(n_epochs):
        logging.info(f"Epoch {epoch + 1}/{n_epochs}")
        epch = int(total_epochs-((2/100)*total_epochs))
        current_replacement_pct = int(1)
        # get_replacement_percentage(epoch, total_epochs)
        logging.info(f"Epoch {epoch+1}: Using {current_replacement_pct*100:.1f}% generated images")


        G_T1_to_T2.train()
        G_T2_to_PD.train()    
        G_PD_to_T1.train()
        D_T1.train()
        D_T2.train()
        D_PD.train()

        total_G_loss = 0.0
        total_D_loss = 0.0

        
        epoch_psnr_T2_T1 = 0.0
        epoch_ssim_T2_T1 = 0.0
        epoch_lpips_T2_T1 = 0.0
        epoch_psnr_PD_T1 = 0.0
        epoch_ssim_PD_T1 = 0.0
        epoch_lpips_PD_T1 = 0.0

        epoch_psnr_T1_T2 = 0.0
        epoch_ssim_T1_T2 = 0.0
        epoch_lpips_T1_T2 = 0.0
        epoch_psnr_PD_T2 = 0.0
        epoch_ssim_PD_T2 = 0.0
        epoch_lpips_PD_T2 = 0.0

        epoch_psnr_T1_PD = 0.0
        epoch_ssim_T1_PD = 0.0
        epoch_lpips_T1_PD = 0.0
        epoch_psnr_T2_PD = 0.0
        epoch_ssim_T2_PD = 0.0
        epoch_lpips_T2_PD = 0.0
       

        num_batches = 0

        for i, batch in enumerate(train_loader):
            if batch is None:
                continue

            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)
            real_PD = batch["flair"].to(device)

            for opt in optimizer_G.values():
                opt.zero_grad()
            for optD in optimizer_D.values():
                optD.zero_grad()
            

            try:
                with autocast():


                    # Forward cycle: T1 → T2 → PD → T1
                    fake_T2, _ = G_T1_to_T2(real_T1)
                    fake_T2_replaced = fake_T2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T2_replaced = torch.where(replace_mask[:, None, None, None], fake_T2, real_T2)
                    
                    fake_PD, _ = G_T2_to_PD(fake_T2_replaced)
                    fake_PD_replaced = fake_PD
                    
                    # Replace some fake_PD with real_PD based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_PD.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_PD_replaced = torch.where(replace_mask[:, None, None, None], fake_PD, real_PD)
                    
                    cycled_T1, _ = G_PD_to_T1(fake_PD_replaced)
                    


                    # Backward cycle: PD → T1 → T2 → PD
                    fake_T1_from_PD, _ = G_PD_to_T1(real_PD)
                    fake_T1_from_PD_replaced = fake_T1_from_PD
                    
                    # Replace some fake_T1_from_PD with real_T1 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T1_from_PD.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T1_from_PD_replaced = torch.where(replace_mask[:, None, None, None], fake_T1_from_PD, real_T1)
                    
                    fake_T2_from_T1, _ = G_T1_to_T2(fake_T1_from_PD_replaced)
                    fake_T2_from_T1_replaced = fake_T2_from_T1
                    
                    # Replace some fake_T2_from_T1 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T2_from_T1.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T2_from_T1_replaced = torch.where(replace_mask[:, None, None, None], fake_T2_from_T1, real_T2)
                    
                    cycled_PD, _ = G_T2_to_PD(fake_T2_from_T1_replaced)
                    


                    # CYCLE FOR T2 T2-->PD-->T1-->T2

                    fake_PD_from_T2_inT2, _ = G_T2_to_PD(real_T2)
                    fake_PD_from_T2_inT2_replaced = fake_PD_from_T2_inT2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_PD_from_T2_inT2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_PD_from_T2_inT2_replaced = torch.where(replace_mask[:, None, None, None], fake_PD_from_T2_inT2, real_PD)
                    

                    fake_T1_from_PD_inT2, _ = G_PD_to_T1(fake_PD_from_T2_inT2_replaced)
                    fake_T1_from_PD_inT2_replaced = fake_T1_from_PD_inT2
                    # Replace some fake_T2 with real_T2 based on current_replacement_pct
                    if current_replacement_pct > 0:
                        replace_mask = torch.rand(fake_T1_from_PD_inT2.shape[0]) < current_replacement_pct
                        replace_mask = replace_mask.to(device)
                        fake_T1_from_PD_inT2_replaced = torch.where(replace_mask[:, None, None, None], fake_T1_from_PD_inT2, real_T1)
                    
                    
                    cycled_T2_inT2, _ = G_T1_to_T2(fake_T1_from_PD_inT2_replaced)



                    # Compute cycle consistency loss for T1 and direct reconstruction loss for T2
                    loss_cycle_T1 = criterion_cycle(cycled_T1, real_T1) * lambda_dict['lambda_cycle_T1']
                    loss_cycle_PD = criterion_cycle(cycled_PD, real_PD) * lambda_dict['lambda_cycle_PD']
                    loss_cycle_T2 = criterion_cycle(cycled_T2_inT2, real_T2) * lambda_dict['lambda_cycle_T2']


                    # Identity losses
                    idt_T1, _ = G_PD_to_T1(real_T1)
                    loss_idt_T1 = criterion_identity(idt_T1, real_T1) * lambda_dict['lambda_identity_T1']
                    idt_T2, _ = G_T1_to_T2(real_T2)
                    loss_idt_T2 = criterion_identity(idt_T2, real_T2) * lambda_dict['lambda_identity_T2']
                    idt_PD, _ = G_T2_to_PD(real_PD)
                    loss_idt_PD = criterion_identity(idt_PD, real_PD) * lambda_dict['lambda_identity_PD']
                
                    loss_identity = loss_idt_T1 + loss_idt_T2 + loss_idt_PD

                    # GAN losses for both generators
                    pred_fake_T2, fake_features_T2 = D_T2(fake_T2)
                    loss_GAN_T1_to_T2 = criterion_GAN(
                        pred_fake_T2, torch.ones_like(pred_fake_T2, device=device)
                    ) * lambda_dict['lambda_GAN_T1_T2']


                    pred_fake_PD, fake_features_PD = D_PD(fake_PD_from_T2_inT2)
                    loss_GAN_T2_to_PD = criterion_GAN(
                        pred_fake_PD, torch.ones_like(pred_fake_PD, device=device)
                    ) * lambda_dict['lambda_GAN_T2_PD']
                    

                    pred_fake_T1, fake_features_T1 = D_T1(fake_T1_from_PD)
                    loss_GAN_PD_to_T1 = criterion_GAN(
                        pred_fake_T1, torch.ones_like(pred_fake_T1, device=device)
                        )* lambda_dict['lambda_GAN_PD_T1']

                

                    loss_G = loss_GAN_T1_to_T2 + loss_GAN_T2_to_PD + loss_GAN_PD_to_T1 + loss_cycle_T1 + loss_identity + loss_cycle_PD + loss_cycle_T2

                    # Feature Matching Loss
                    pred_real_T2, real_features_T2 = D_T2(real_T2)
                    pred_real_PD, real_features_PD = D_PD(real_PD)
                    pred_real_T1, real_features_T1 = D_T1(real_T1)

                    loss_fm_T2 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T2, fake_features_T2)
                    )
                    loss_fm_PD = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_PD, fake_features_PD)
                    )
                    loss_fm_T1 = sum(
                        criterion_feature_matching(fake, real)
                        for real, fake in zip(real_features_T1, fake_features_T1)
                    )

                    loss_feature_matching = loss_fm_T2*lambda_fm_D_T2+ loss_fm_T1*lambda_fm_D_T1 + loss_fm_PD*lambda_fm_D_PD

                    loss_G += loss_feature_matching

            except Exception as e:
                logging.error(f"Error during generator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_G).backward()
                for optimizer in optimizer_G.values():
                    scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during generator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                with autocast():
                    # Discriminator loss for T2
                    pred_real_T2, _ = D_T2(real_T2)
                    loss_D_real_T2 = criterion_GAN(pred_real_T2, torch.ones_like(pred_real_T2, device=device))
                    
                    pred_fake_T2, _ = D_T2(fake_T2.detach())
                    loss_D_fake_T2 = criterion_GAN(pred_fake_T2, torch.zeros_like(pred_fake_T2, device=device))
                    loss_D_T2 = (loss_D_real_T2 + loss_D_fake_T2) * 0.5
                   

                    # Discriminator PD
                    pred_real_PD, _ = D_PD(real_PD)
                    loss_D_real_PD = criterion_GAN(pred_real_PD, torch.ones_like(pred_real_PD, device=device))
                    
                    pred_fake_PD, _ = D_PD(fake_PD_from_T2_inT2.detach())
                    loss_D_fake_PD = criterion_GAN(pred_fake_PD, torch.zeros_like(pred_fake_PD, device=device))
                    loss_D_PD = (loss_D_real_PD + loss_D_fake_PD) * 0.5
                    
                    # Discriminator T1
                    pred_real_T1, _ = D_T1(real_T1)
                    loss_D_real_T1 = criterion_GAN(pred_real_T1, torch.ones_like(pred_real_T1, device=device))
                    
                    pred_fake_T1, _ = D_T1(fake_T1_from_PD.detach())
                    loss_D_fake_T1 = criterion_GAN(pred_fake_T1, torch.zeros_like(pred_fake_T1, device=device))
                    loss_D_T1 = (loss_D_real_T1 + loss_D_fake_T1) * 0.5
                    
                    loss_D = loss_D_T2 + loss_D_PD + loss_D_T1
                
            except Exception as e:
                logging.error(f"Error during discriminator forward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            try:
                scaler.scale(loss_D).backward()
                for optimizer in optimizer_D.values():
                    scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                logging.error(f"Error during discriminator backward pass at epoch {epoch+1}, batch {i+1}: {e}", exc_info=True)
                continue

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()

            # fake_T2, _ = G_T1_to_T2(real_T1)
            # fake_PD, _ = G_T2_to_PD(fake_T2)
            # cycled_T1, _ = G_PD_to_T1(fake_PD)

            #numbers from T1 
            psnr_T2_T1 = calculate_psnr(real_T2, fake_T2)
            ssim_T2_T1 = calculate_ssim(real_T2, fake_T2)
            # print(f"Input device: {input_tensor.device}")
            # print(f"Model device: {next(model.parameters()).device}")
            lpips_T2_T1 = metric_calculator.calculate_lpips(real_T2, fake_T2)
            psnr_PD_T1 = calculate_psnr(real_PD, fake_PD)
            ssim_PD_T1 = calculate_ssim(real_PD, fake_PD)
            lpips_PD_T1 = metric_calculator.calculate_lpips(real_PD, fake_PD)

            #NUMBERS FROM T2
            psnr_T1_T2 = calculate_psnr(real_T1, fake_T1_from_PD_inT2)
            ssim_T1_T2 = calculate_ssim(real_T1, fake_T1_from_PD_inT2)
            lpips_T1_T2 = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD_inT2)
            psnr_PD_T2 = calculate_psnr(real_PD, fake_PD_from_T2_inT2)
            ssim_PD_T2 = calculate_ssim(real_PD, fake_PD_from_T2_inT2)
            lpips_PD_T2 = metric_calculator.calculate_lpips(real_PD, fake_PD_from_T2_inT2)

            #NUMBERRS FROM PD
            psnr_T1_PD = calculate_psnr(real_T1, fake_T1_from_PD)
            ssim_T1_PD = calculate_ssim(real_T1, fake_T1_from_PD)
            lpips_T1_PD = metric_calculator.calculate_lpips(real_T1, fake_T1_from_PD)
            psnr_T2_PD = calculate_psnr(real_T2, fake_T2_from_T1)
            ssim_T2_PD = calculate_ssim(real_T2, fake_T2_from_T1)
            lpips_T2_PD = metric_calculator.calculate_lpips(real_T2, fake_T2_from_T1)            

            
            #FOR T1
            epoch_psnr_T2_T1 += psnr_T2_T1
            epoch_ssim_T2_T1 += ssim_T2_T1
            epoch_lpips_T2_T1 += lpips_T2_T1
            epoch_psnr_PD_T1 += psnr_PD_T1
            epoch_ssim_PD_T1 += ssim_PD_T1
            epoch_lpips_PD_T1 += lpips_PD_T1
            
            #FOR T2
            epoch_psnr_T1_T2 += psnr_T1_T2
            epoch_ssim_T1_T2 += ssim_T1_T2
            epoch_lpips_T1_T2 += lpips_T1_T2
            epoch_psnr_PD_T2 += psnr_PD_T2
            epoch_ssim_PD_T2 += ssim_PD_T2
            epoch_lpips_PD_T2 += lpips_PD_T2

            #FOR PD
            epoch_psnr_T1_PD += psnr_T1_PD
            epoch_ssim_T1_PD += ssim_T1_PD
            epoch_lpips_T1_PD += lpips_T1_PD
            epoch_psnr_T2_PD += psnr_T2_PD
            epoch_ssim_T2_PD += ssim_T2_PD
            epoch_lpips_T2_PD += lpips_T2_PD


            num_batches += 1

            if i < 2:
                epoch_images_dir = os.path.join(images_dir, f'epoch_{epoch+1}', 'train')
                os.makedirs(epoch_images_dir, exist_ok=True)
                save_images(real_T1, real_T2, real_PD,fake_T2,fake_PD,cycled_T1,fake_T1_from_PD,fake_T2_from_T1,cycled_PD,fake_PD_from_T2_inT2,fake_T1_from_PD_inT2,cycled_T2_inT2
                           ,epoch_images_dir, epoch + 1, i, dataset_type='train')

        avg_G_loss = total_G_loss / num_batches if num_batches > 0 else 0
        avg_D_loss = total_D_loss / num_batches if num_batches > 0 else 0
        
        #FROM T1
        avg_psnr_T2_T1 = epoch_psnr_T2_T1 / num_batches if num_batches > 0 else 0
        avg_ssim_T2_T1 = epoch_ssim_T2_T1 / num_batches if num_batches > 0 else 0
        avg_lpips_T2_T1 = epoch_lpips_T2_T1 / num_batches if num_batches > 0 else 0
        avg_psnr_PD_T1 = epoch_psnr_PD_T1 / num_batches if num_batches > 0 else 0
        avg_ssim_PD_T1 = epoch_ssim_PD_T1 / num_batches if num_batches > 0 else 0
        avg_lpips_PD_T1 = epoch_lpips_PD_T1 / num_batches if num_batches > 0 else 0
        
        #FROM T2
        avg_psnr_T1_T2 = epoch_psnr_T1_T2 / num_batches if num_batches > 0 else 0
        avg_ssim_T1_T2 = epoch_ssim_T1_T2 / num_batches if num_batches > 0 else 0
        avg_lpips_T1_T2 = epoch_lpips_T1_T2 / num_batches if num_batches > 0 else 0
        avg_psnr_PD_T2 = epoch_psnr_PD_T2 / num_batches if num_batches > 0 else 0
        avg_ssim_PD_T2 = epoch_ssim_PD_T2 / num_batches if num_batches > 0 else 0
        avg_lpips_PD_T2 = epoch_lpips_PD_T2 / num_batches if num_batches > 0 else 0

        #FROM PD
        avg_psnr_T1_PD = epoch_psnr_T1_PD / num_batches if num_batches > 0 else 0
        avg_ssim_T1_PD = epoch_ssim_T1_PD / num_batches if num_batches > 0 else 0
        avg_lpips_T1_PD = epoch_lpips_T1_PD / num_batches if num_batches > 0 else 0
        avg_psnr_T2_PD = epoch_psnr_T2_PD / num_batches if num_batches > 0 else 0
        avg_ssim_T2_PD = epoch_ssim_T2_PD / num_batches if num_batches > 0 else 0
        avg_lpips_T2_PD = epoch_lpips_T2_PD / num_batches if num_batches > 0 else 0


        logging.info(f"Epoch {epoch + 1} Training Losses: Gen {avg_G_loss:.4f}, Dis {avg_D_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} Training Metrics:")
        logging.info(f"Metrics from T1")
        logging.info(f"  T2: PSNR {avg_psnr_T2_T1:.4f}, SSIM {avg_ssim_T2_T1:.4f}, LPIPS {avg_lpips_T2_T1:.4f}")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T1:.4f}, SSIM {avg_ssim_PD_T1:.4f}, LPIPS {avg_lpips_PD_T1:.4f}")
        logging.info(f"Metrics from T2")
        logging.info(f"  T1: PSNR {avg_psnr_T1_T2:.4f}, SSIM {avg_ssim_T1_T2:.4f}, LPIPS {avg_lpips_T1_T2:.4f}")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T2:.4f}, SSIM {avg_ssim_PD_T2:.4f}, LPIPS {avg_lpips_PD_T2:.4f}")
        logging.info(f"Metrics from PD")
        logging.info(f"  T1: PSNR {avg_psnr_T1_PD:.4f}, SSIM {avg_ssim_T1_PD:.4f}, LPIPS {avg_lpips_T1_PD:.4f}")
        logging.info(f"  T2: PSNR {avg_psnr_T2_PD:.4f}, SSIM {avg_ssim_T2_PD:.4f}, LPIPS {avg_lpips_T2_PD:.4f}")




        train_losses.append({
            'gen_total_loss_train': avg_G_loss,
            'dis_total_loss_train': avg_D_loss,
            'cycle_loss_T1_train': loss_cycle_T1.item() if num_batches > 0 else 0,
            'cycle_loss_PD_train':loss_cycle_PD.item() if num_batches > 0 else 0,
            'cycle_loss_T2_train':loss_cycle_T2.item() if num_batches > 0 else 0,
            'identity_loss_train': loss_identity.item() if num_batches > 0 else 0,
            'feature_matching_loss_train': loss_feature_matching.item() if num_batches > 0 else 0,
            'gan_loss_T1_to_T2_train': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
            'gan_loss_T2_to_PD_train': loss_GAN_T2_to_PD.item() if num_batches > 0 else 0,
            'gan_loss_PD_to_T1_train': loss_GAN_PD_to_T1.item() if num_batches > 0 else 0,
            'dis_loss_T2_train':loss_D_T2.item() if num_batches > 0 else 0,
            'dis_loss_PD_train':loss_D_PD.item() if num_batches > 0 else 0,
            'dis_loss_T1_train':loss_D_T1.item() if num_batches > 0 else 0

        })

        train_metrics.append({
            
            'PSNR_T2_train_T1': avg_psnr_T2_T1,
            'SSIM_T2_train_T1': avg_ssim_T2_T1,
            'LPIPS_T2_train_T1': avg_lpips_T2_T1,
            'PSNR_PD_train_T1': avg_psnr_PD_T1,
            'SSIM_PD_train_T1': avg_ssim_PD_T1,
            'LPIPS_PD_train_T1': avg_lpips_PD_T1,

            'PSNR_T1_train_T2': avg_psnr_T1_T2,
            'SSIM_T1_train_T2': avg_ssim_T1_T2,
            'LPIPS_T1_train_T2': avg_lpips_T1_T2,
            'PSNR_PD_train_T2': avg_psnr_PD_T2,
            'SSIM_PD_train_T2': avg_ssim_PD_T2,
            'LPIPS_PD_train_T2': avg_lpips_PD_T2,

            'PSNR_T1_train_PD': avg_psnr_T1_PD,
            'SSIM_T1_train_PD': avg_ssim_T1_PD,
            'LPIPS_T1_train_PD': avg_lpips_T1_PD,
            'PSNR_T2_train_PD': avg_psnr_T2_PD,
            'SSIM_T2_train_PD': avg_ssim_T2_PD,
            'LPIPS_T2_train_PD': avg_lpips_T2_PD,
            
        })

        try:
            (avg_G_loss_val, avg_D_loss_val,
             
             avg_psnr_T2_T1_val, avg_ssim_T2_T1_val, avg_lpips_T2_T1_val,
             avg_psnr_PD_T1_val, avg_ssim_PD_T1_val, avg_lpips_PD_T1_val,

             avg_psnr_T1_T2_val, avg_ssim_T1_T2_val, avg_lpips_T1_T2_val,
             avg_psnr_PD_T2_val, avg_ssim_PD_T2_val, avg_lpips_PD_T2_val,

             avg_psnr_T1_PD_val, avg_ssim_T1_PD_val, avg_lpips_T1_PD_val,
             avg_psnr_T2_PD_val, avg_ssim_T2_PD_val, avg_lpips_T2_PD_val,
             ) = validate(
                val_loader, G_T1_to_T2, G_T2_to_PD, G_PD_to_T1,
                D_T2, D_PD, D_T1,
                criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching,
                lambda_dict, device, images_dir, epoch, wandb_run=wandb_run ) # Pass the wandb_run object
        except Exception as e:
            logging.error(f"Error during validation at epoch {epoch+1}: {e}", exc_info=True)
            raise

        logging.info(f"Epoch {epoch + 1} Validation Losses: Gen {avg_G_loss_val:.4f}, Dis {avg_D_loss_val:.4f}\n")
        logging.info(f"Epoch {epoch + 1} Validation Metrics:\n")
        logging.info("generated from T1\n")
        logging.info(f"  T2: PSNR {avg_psnr_T2_T1_val:.4f}, SSIM {avg_ssim_T2_T1_val:.4f}, LPIPS {avg_lpips_T2_T1_val:.4f}\n")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T1_val:.4f}, SSIM {avg_ssim_PD_T1_val:.4f}, LPIPS {avg_lpips_PD_T1_val:.4f}\n")
        logging.info("generated from T2\n")
        logging.info(f"  T1: PSNR {avg_psnr_T1_T2_val:.4f}, SSIM {avg_ssim_T1_T2_val:.4f}, LPIPS {avg_lpips_T1_T2_val:.4f}\n")
        logging.info(f"  PD: PSNR {avg_psnr_PD_T2_val:.4f}, SSIM {avg_ssim_PD_T2_val:.4f}, LPIPS {avg_lpips_PD_T2_val:.4f}\n")
        logging.info("generated from PD\n")
        logging.info(f"  T1: PSNR {avg_psnr_T1_PD_val:.4f}, SSIM {avg_ssim_T1_PD_val:.4f}, LPIPS {avg_lpips_T1_PD_val:.4f}\n")
        logging.info(f"  T2: PSNR {avg_psnr_T2_PD_val:.4f}, SSIM {avg_ssim_T2_PD_val:.4f}, LPIPS {avg_lpips_T2_PD_val:.4f}\n")
        
        val_losses.append({
            'gen_total_loss_val': avg_G_loss_val,
            'dis_total_loss_val': avg_D_loss_val,
            'cycle_loss_T1_val': loss_cycle_T1.item() if num_batches > 0 else 0,
            'cycle_loss_PD_val': loss_cycle_PD.item() if num_batches > 0 else 0,
            'cycle_loss_T2_val': loss_cycle_T2.item() if num_batches > 0 else 0,
            'identity_loss_val': loss_identity.item() if num_batches > 0 else 0,
            'feature_matching_loss_val': loss_feature_matching.item() if num_batches > 0 else 0,
            'gan_loss_T1_to_T2_val': loss_GAN_T1_to_T2.item() if num_batches > 0 else 0,
            'gan_loss_T2_to_PD_val': loss_GAN_T2_to_PD.item() if num_batches > 0 else 0,
            'gan_loss_PD_to_T1_val': loss_GAN_PD_to_T1.item() if num_batches > 0 else 0,
            'dis_loss_T2_val': loss_D_T2.item() if num_batches > 0 else 0,
            'dis_loss_PD_val': loss_D_PD.item() if num_batches > 0 else 0,
            'dis_loss_T1_val': loss_D_T1.item() if num_batches > 0 else 0
        })
        
        val_metrics.append({
            
            'PSNR_T2_T1_val': avg_psnr_T2_T1_val,
            'SSIM_T2_T1_val': avg_ssim_T2_T1_val,
            'LPIPS_T2_T1_val': avg_lpips_T2_T1_val,
            'PSNR_PD_T1_val': avg_psnr_PD_T1_val,
            'SSIM_PD_T1_val': avg_ssim_PD_T1_val,
            'LPIPS_PD_T1_val': avg_lpips_PD_T1_val,

            'PSNR_T1_T2_val': avg_psnr_T1_T2_val,
            'SSIM_T1_T2_val': avg_ssim_T1_T2_val,
            'LPIPS_T1_T2_val': avg_lpips_T1_T2_val,
            'PSNR_PD_T2_val': avg_psnr_PD_T2_val,
            'SSIM_PD_T2_val': avg_ssim_PD_T2_val,
            'LPIPS_PD_T2_val': avg_lpips_PD_T2_val,

            'PSNR_T1_PD_val': avg_psnr_T1_PD_val,
            'SSIM_T1_PD_val': avg_ssim_T1_PD_val,
            'LPIPS_T1_PD_val': avg_lpips_T1_PD_val,
            'PSNR_T2_PD_val': avg_psnr_T2_PD_val,
            'SSIM_T2_PD_val': avg_ssim_T2_PD_val,
            'LPIPS_T2_PD_val': avg_lpips_T2_PD_val
           

        })

        current_avg_ssim_T1 = (avg_ssim_T2_T1_val + avg_ssim_PD_T1_val) / 2
        current_avg_ssim_T2 = (avg_ssim_T1_T2_val + avg_ssim_PD_T2_val) / 2
        current_avg_ssim_PD = (avg_ssim_T2_PD_val + avg_ssim_T1_PD_val) / 2
        current_avg_ssim = (current_avg_ssim_PD+current_avg_ssim_T1+current_avg_ssim_T2)/3

        current_avg_psnr_T1 = (avg_psnr_T2_T1_val + avg_psnr_PD_T1_val) / 2
        current_avg_psnr_T2 = (avg_psnr_T1_T2_val + avg_psnr_PD_T2_val) / 2
        current_avg_psnr_PD = (avg_psnr_T2_PD_val + avg_psnr_T1_PD_val) / 2
        current_avg_psnr = (current_avg_psnr_PD+current_avg_psnr_T1+current_avg_psnr_T2)/3

        current_avg_lpips_T1 = (avg_lpips_T2_T1_val + avg_lpips_PD_T1_val) / 2
        current_avg_lpips_T2 = (avg_lpips_T1_T2_val + avg_lpips_PD_T2_val) / 2
        current_avg_lpips_PD = (avg_lpips_T2_PD_val + avg_lpips_T1_PD_val) / 2
        current_avg_lpips = (current_avg_lpips_PD+current_avg_lpips_T1+current_avg_lpips_T2)/3
        


        best_lpips = float('inf')
        
        # NEW: Calculate the combined score
        current_score = calculate_combined_score(
            current_avg_psnr, 
            current_avg_ssim, 
            current_avg_lpips
        )

        
        # Check if this is the best model based on the new combined score
        if current_score > best_combined_score:
            best_combined_score = current_score
            
            logging.info(f"New best model found at epoch {epoch + 1} with Combined Score: {best_combined_score:.4f}")
            logging.info(f"  (PSNR: {current_avg_psnr:.4f}, SSIM: {current_avg_ssim:.4f}, LPIPS: {current_avg_lpips:.4f})")

            model_save_path = os.path.join(experiment_dir, 'best_model-combined.pth')
            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_PD': G_T2_to_PD.state_dict(),
                'G_PD_to_T1': G_PD_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'D_PD': D_PD.state_dict(),
                'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
                'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
                'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
                'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
                'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
                'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'best_lpips': best_lpips,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, model_save_path)
                
        if current_avg_psnr > best_psnr:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model for psnr found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_psnr.pth')
            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_PD': G_T2_to_PD.state_dict(),
                'G_PD_to_T1': G_PD_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'D_PD': D_PD.state_dict(),
                'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
                'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
                'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
                'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
                'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
                'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'best_lpips': best_lpips,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, model_save_path)
        if current_avg_lpips < best_lpips:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model for lpips found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_lpips.pth')
            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_PD': G_T2_to_PD.state_dict(),
                'G_PD_to_T1': G_PD_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'D_PD': D_PD.state_dict(),
                'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
                'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
                'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
                'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
                'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
                'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'best_lpips': best_lpips,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, model_save_path)
        if current_avg_ssim > best_ssim:
            best_ssim = current_avg_ssim
            best_psnr = current_avg_psnr
            best_lpips = current_avg_lpips
            logging.info(f"New best model for ssim found at epoch {epoch + 1} with avg SSIM {best_ssim:.4f}, PSNR {best_psnr:.4f}, LPIPS {best_lpips:.4f}")
            model_save_path = os.path.join(experiment_dir, 'best_model_ssim.pth')
            torch.save({
                'G_T1_to_T2': G_T1_to_T2.state_dict(),
                'G_T2_to_PD': G_T2_to_PD.state_dict(),
                'G_PD_to_T1': G_PD_to_T1.state_dict(),
                'D_T1': D_T1.state_dict(),
                'D_T2': D_T2.state_dict(),
                'D_PD': D_PD.state_dict(),
                'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
                'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
                'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
                'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
                'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
                'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
                'epoch': epoch + 1,
                'best_ssim': best_ssim,
                'best_psnr': best_psnr,
                'best_lpips': best_lpips,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, model_save_path)

        
            # torch.save({
            #         'G_T1_to_T2': G_T1_to_T2.state_dict(),
            #         'G_T2_to_PD': G_T2_to_PD.state_dict(),
            #         'G_PD_to_T1': G_PD_to_T1.state_dict(),
            #         'D_T1': D_T1.state_dict(),
            #         'D_T2': D_T2.state_dict(),
            #         'D_PD': D_PD.state_dict(),
            #         'optimizer_G_T1_to_T2': optimizer_G["G_T1_to_T2"].state_dict(),
            #         'optimizer_G_T2_to_PD': optimizer_G["G_T2_to_PD"].state_dict(),
            #         'optimizer_G_PD_to_T1': optimizer_G["G_PD_to_T1"].state_dict(),
            #         'optimizer_D_T1': optimizer_D["D_T1"].state_dict(),
            #         'optimizer_D_T2': optimizer_D["D_T2"].state_dict(),
            #         'optimizer_D_PD': optimizer_D["D_PD"].state_dict(),
            #         'epoch': epoch + 1,
            #         'best_ssim': best_ssim,
            #         'best_psnr': best_psnr,
            #         'best_lpips': best_lpips,
            #         'train_losses': train_losses,
            #         'val_losses': val_losses,
            #         'train_metrics': train_metrics,
            #         'val_metrics': val_metrics
            #     }, model_save_path)

        logging.info(f"Best model saved at epoch {epoch + 1} with avg SSIM {best_ssim:.4f} and avg PSNR {best_psnr:.4f} and avg lpips {best_lpips:.4f} based on SSIM")

        for scheduler in scheduler_G.values():
            scheduler.step()
        for scheduler in scheduler_D.values():
            scheduler.step()

        try:
            plot_losses_metrics(train_losses, val_losses, train_metrics, val_metrics, plots_dir)
            logging.info(f"Plots saved for epoch {epoch + 1}.")
        except Exception as e:
            logging.error(f"Error while plotting at epoch {epoch+1}: {e}", exc_info=True)

    return best_ssim, best_psnr, best_lpips
