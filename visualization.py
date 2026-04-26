# visualization.py
import os
import matplotlib.pyplot as plt
import logging
import torch
import numpy as np

def denormalize(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1] for visualization."""
    return tensor * 0.5 + 0.5

def save_images(real_T1, real_T2, real_PD,fake_T2,fake_PD,cycled_T1,fake_T1_from_PD,fake_T2_from_T1,cycled_PD,fake_PD_from_T2_inT2,fake_T1_from_PD_inT2,cycled_T2_inT2, images_dir, epoch, batch_idx, dataset_type='train'):
    """
    Saves a structured set of real, fake, and cycled images for clear visualization.
    Rows represent cycles, and columns represent the modality.
    """
    os.makedirs(images_dir, exist_ok=True)
    # Use the first image in the batch for visualization
    i = 0  

    # Denormalize all images for proper display
    real_T1_vis = denormalize(real_T1[i].cpu().detach())
    real_T2_vis = denormalize(real_T2[i].cpu().detach())
    real_PD_vis = denormalize(real_PD[i].cpu().detach())

    # Images from the T1 -> T2 -> PD -> T1 cycle
    fake_T2_from_T1_vis = denormalize(fake_T2[i].cpu().detach())
    fake_PD_from_T2_vis = denormalize(fake_PD[i].cpu().detach())
    cycled_T1_vis = denormalize(cycled_T1[i].cpu().detach())

    # Images from the T2 -> PD -> T1 -> T2 cycle
    fake_PD_from_T2_inT2_vis = denormalize(fake_PD_from_T2_inT2[i].cpu().detach())
    fake_T1_from_PD_inT2_vis = denormalize(fake_T1_from_PD_inT2[i].cpu().detach())
    cycled_T2_inT2_vis = denormalize(cycled_T2_inT2[i].cpu().detach())

    # Images from the PD -> T1 -> T2 -> PD cycle
    fake_T1_from_PD_vis = denormalize(fake_T1_from_PD[i].cpu().detach())
    fake_T2_from_T1_inPD_vis = denormalize(fake_T2_from_T1[i].cpu().detach())
    cycled_PD_vis = denormalize(cycled_PD[i].cpu().detach())

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    plt.suptitle(f'{dataset_type.capitalize()} Epoch {epoch}, Batch {batch_idx}', fontsize=16)

    # --- Row 1: Real Images ---
    axes[0, 0].imshow(real_T1_vis.squeeze(), cmap='gray')
    axes[0, 0].set_title('Real T1')
    axes[0, 1].imshow(real_T2_vis.squeeze(), cmap='gray')
    axes[0, 1].set_title('Real T2')
    axes[0, 2].imshow(real_PD_vis.squeeze(), cmap='gray')
    axes[0, 2].set_title('Real PD')

    # --- Row 2: Cycle T1 -> T2 -> PD -> T1 ---
    axes[1, 0].imshow(cycled_T1_vis.squeeze(), cmap='gray')
    axes[1, 0].set_title('Start: cycled T1')
    axes[1, 1].imshow(fake_T2_from_T1_vis.squeeze(), cmap='gray')
    axes[1, 1].set_title('-> Fake T2')
    axes[1, 2].imshow(fake_PD_from_T2_vis.squeeze(), cmap='gray')
    axes[1, 2].set_title('-> Fake PD')

    # --- Row 3: Cycle T2 -> PD -> T1 -> T2 ---
    axes[2, 0].imshow(fake_T1_from_PD_inT2_vis.squeeze(), cmap='gray')
    axes[2, 0].set_title('Fake T1')
    axes[2, 1].imshow(cycled_T2_inT2_vis.squeeze(), cmap='gray')
    axes[2, 1].set_title('-> Start: cycled T2')
    axes[2, 2].imshow(fake_PD_from_T2_inT2_vis.squeeze(), cmap='gray')
    axes[2, 2].set_title('-> fake PD')

    # --- Row 4: Cycle PD -> T1 -> T2 -> PD ---
    axes[3, 0].imshow(fake_T1_from_PD_vis.squeeze(), cmap='gray')
    axes[3, 0].set_title('fake T1')
    axes[3, 1].imshow(fake_T2_from_T1_inPD_vis.squeeze(), cmap='gray')
    axes[3, 1].set_title('-> Fake T2')
    axes[3, 2].imshow(cycled_PD_vis.squeeze(), cmap='gray')
    axes[3, 2].set_title('-> Start: Cycled PD')
    
    # Turn off axes for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    save_path = os.path.join(images_dir, f"{dataset_type}_epoch_{epoch}_batch_{batch_idx}.png")
    
    try:
        plt.savefig(save_path)
        logging.info(f"Image grid saved at {save_path}")
    except Exception as e:
        logging.error(f"Failed to save image grid at {save_path}: {e}")
    finally:
        plt.close(fig)


def plot_losses_metrics(train_losses, val_losses, train_metrics, val_metrics, plots_dir):
    """
    Plots and saves loss and metric curves over training epochs.
    This version is simplified for clarity and robustness.
    """
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # --- Plotting Individual Loss Components ---
    loss_groups = {
        'Generator_Loss': ['gen_total_loss_train', 'gen_total_loss_val'],
        'Discriminator_Loss': ['dis_total_loss_train', 'dis_total_loss_val'],
        'GAN_Loss': ['gan_loss_T1_to_T2_train', 'gan_loss_T2_to_PD_train', 'gan_loss_PD_to_T1_train'],
        'Cycle_Consistency_Loss': ['cycle_loss_T1_train', 'cycle_loss_PD_train', 'cycle_loss_T2_train'],
        'Identity_Loss': ['identity_loss_train', 'identity_loss_val'],
        'Feature_Matching_Loss': ['feature_matching_loss_train', 'feature_matching_loss_val'],
    }
    
    for title, keys in loss_groups.items():
        plt.figure(figsize=(10, 6))
        for key in keys:
            if 'train' in key and train_losses and key in train_losses[0]:
                values = [d[key] for d in train_losses]
                plt.plot(epochs, values, label=f'Train {key.replace("_train", "")}', linestyle='-')
            elif 'val' in key and val_losses and key in val_losses[0]:
                values = [d[key] for d in val_losses]
                plt.plot(epochs, values, label=f'Validation {key.replace("_val", "")}', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'{title.lower()}.png'), bbox_inches='tight')
        plt.close()

    # --- Plotting Metrics (PSNR, SSIM, LPIPS) ---
    metric_types = ['PSNR', 'SSIM', 'LPIPS']
    # Format: (Key in dict, Legend Label)
    metric_keys_to_plot = [
        ('T2_train_T1', 'T1 -> T2 (Train)'), ('T2_T1_val', 'T1 -> T2 (Val)'),
        ('PD_train_T1', 'T1 -> PD (Train)'), ('PD_T1_val', 'T1 -> PD (Val)'),
        ('T1_train_T2', 'T2 -> T1 (Train)'), ('T1_T2_val', 'T2 -> T1 (Val)'),
        ('PD_train_T2', 'T2 -> PD (Train)'), ('PD_T2_val', 'T2 -> PD (Val)'),
        ('T1_train_PD', 'PD -> T1 (Train)'), ('T1_PD_val', 'PD -> T1 (Val)'),
        ('T2_train_PD', 'PD -> T2 (Train)'), ('T2_PD_val', 'PD -> T2 (Val)'),
    ]

    for metric_type in metric_types:
        plt.figure(figsize=(12, 8))
        for key_suffix, label in metric_keys_to_plot:
            full_key = f'{metric_type}_{key_suffix}'
            
            source_dict = train_metrics if 'train' in key_suffix else val_metrics
            linestyle = '-' if 'train' in key_suffix else '--'

            if source_dict and full_key in source_dict[0]:
                values = [d[full_key] for d in source_dict]
                plt.plot(epochs, values, label=label, linestyle=linestyle)

        plt.title(f'{metric_type} Metrics Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_type)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'{metric_type.lower()}_metrics.png'), bbox_inches='tight')
        plt.close()

    logging.info(f"All plots saved successfully to {plots_dir}")