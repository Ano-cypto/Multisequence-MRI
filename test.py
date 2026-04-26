import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# --- NEW IMPORT FOR FID ---
from torchmetrics.image.fid import FrechetInceptionDistance

# --- CUSTOM MODULES ---
from generator import Generator_PMC as Generator
from dataset import MRIImageDataset
from metrics import calculate_psnr, calculate_ssim, MetricCalculator
# from generator import AttentionUNet

# Initialize global metric calculator
metric_calculator = MetricCalculator(device='cuda:1')

def setup_test_logging(output_dir):
    """Sets up logging for the test script."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'test_run.log')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])

def tensor_to_pil(tensor):
    """Converts a single PyTorch image tensor to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    img_np = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)
    return Image.fromarray(img_np)

def prepare_for_fid(tensor):
    """
    Prepares a batch of tensors for FID calculation.
    FID expects: uint8 [0, 255], 3 channels (RGB).
    Input: [-1, 1], 1 channel (Grayscale).
    """
    # 1. Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    # 2. Convert to [0, 255] uint8
    tensor = (tensor * 255).to(dtype=torch.uint8)
    # 3. Expand 1 channel to 3 channels (Grayscale -> RGB)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor

def save_test_outputs(real_input, fake_step1, real_step1, fake_step2, real_step2,
                      input_modality, step1_modality, step2_modality,
                      output_dir, batch_idx):
    """Saves separate image files for a transformation chain."""
    base_prefix = f"batch_{batch_idx}_chain_{input_modality}"
    images_to_save = {
        f"{base_prefix}_1_input.png": real_input,
        f"{base_prefix}_2_fake_{step1_modality}.png": fake_step1,
        f"{base_prefix}_3_real_{step1_modality}.png": real_step1,
        f"{base_prefix}_4_fake_{step2_modality}.png": fake_step2,
        f"{base_prefix}_5_real_{step2_modality}.png": real_step2,
    }
    for filename, tensor_img in images_to_save.items():
        try:
            full_path = os.path.join(output_dir, filename)
            pil_img = tensor_to_pil(tensor_img)
            pil_img.save(full_path)
        except Exception as e:
            logging.error(f"Could not save image {filename}: {e}")
    logging.info(f"Saved images for batch {batch_idx}, chain {input_modality}.")

def test_single_model(model_path, test_loader, device, output_dir, num_samples_to_save, model_idx):
    """
    Evaluates a SINGLE model checkpoint.
    Returns average metrics including FID.
    """
    logging.info(f"--- Testing Model {model_idx+1}: {os.path.basename(model_path)} ---")

    # --- Initialize Generators ---
    G_T1_to_T2 = Generator().to(device)
    G_T2_to_PD = Generator().to(device)
    G_PD_to_T1 = Generator().to(device)

    # --- Load Weights ---
    try:
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        G_T1_to_T2.load_state_dict(checkpoint['G_T1_to_T2'])
        G_T2_to_PD.load_state_dict(checkpoint['G_T2_to_PD'])
        G_PD_to_T1.load_state_dict(checkpoint['G_PD_to_T1'])
        logging.info("Successfully loaded generator weights.")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return None

    G_T1_to_T2.eval()
    G_T2_to_PD.eval()
    G_PD_to_T1.eval()
    
    samples_dir = os.path.join(output_dir, f'samples_model_{model_idx+1}')
    os.makedirs(samples_dir, exist_ok=True)
    
    metrics = defaultdict(list)

    # --- FID INITIALIZATION ---
    fid_calcs = {
        'fid_T1_to_T2': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T1_to_PD': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T2_to_PD': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T2_to_T1': FrechetInceptionDistance(feature=2048).to(device),
        'fid_PD_to_T1': FrechetInceptionDistance(feature=2048).to(device),
        'fid_PD_to_T2': FrechetInceptionDistance(feature=2048).to(device),
    }

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None: continue
            
            real_T1, real_T2, real_PD = batch["t1"].to(device), batch["t2"].to(device), batch["pd"].to(device)
            
            # --- Generate ---
            # Path 1: T1 -> T2 -> PD
            fake_T2_from_T1, _ = G_T1_to_T2(real_T1)
            fake_PD_from_T2_from_T1, _ = G_T2_to_PD(fake_T2_from_T1)
            
            # Path 2: T2 -> PD -> T1
            fake_PD_from_T2, _ = G_T2_to_PD(real_T2)
            fake_T1_from_PD_from_T2, _ = G_PD_to_T1(fake_PD_from_T2)
            
            # Path 3: PD -> T1 -> T2
            fake_T1_from_PD, _ = G_PD_to_T1(real_PD)
            fake_T2_from_T1_from_PD, _ = G_T1_to_T2(fake_T1_from_PD)
            
            # --- Record Basic Metrics ---
            def record_metrics(real, fake, prefix):
                metrics[f'psnr_{prefix}'].append(calculate_psnr(real, fake))
                metrics[f'ssim_{prefix}'].append(calculate_ssim(real, fake))
                metrics[f'lpips_{prefix}'].append(metric_calculator.calculate_lpips(real, fake))

            record_metrics(real_T2, fake_T2_from_T1, 'T1_to_T2')
            record_metrics(real_PD, fake_PD_from_T2_from_T1, 'T1_to_PD')
            record_metrics(real_PD, fake_PD_from_T2, 'T2_to_PD')
            record_metrics(real_T1, fake_T1_from_PD_from_T2, 'T2_to_T1')
            record_metrics(real_T1, fake_T1_from_PD, 'PD_to_T1')
            record_metrics(real_T2, fake_T2_from_T1_from_PD, 'PD_to_T2')
            
            # --- Update FID Stats ---
            fid_calcs['fid_T1_to_T2'].update(prepare_for_fid(real_T2), real=True)
            fid_calcs['fid_T1_to_T2'].update(prepare_for_fid(fake_T2_from_T1), real=False)
            
            fid_calcs['fid_T1_to_PD'].update(prepare_for_fid(real_PD), real=True)
            fid_calcs['fid_T1_to_PD'].update(prepare_for_fid(fake_PD_from_T2_from_T1), real=False)
            
            fid_calcs['fid_T2_to_PD'].update(prepare_for_fid(real_PD), real=True)
            fid_calcs['fid_T2_to_PD'].update(prepare_for_fid(fake_PD_from_T2), real=False)
            
            fid_calcs['fid_T2_to_T1'].update(prepare_for_fid(real_T1), real=True)
            fid_calcs['fid_T2_to_T1'].update(prepare_for_fid(fake_T1_from_PD_from_T2), real=False)
            
            fid_calcs['fid_PD_to_T1'].update(prepare_for_fid(real_T1), real=True)
            fid_calcs['fid_PD_to_T1'].update(prepare_for_fid(fake_T1_from_PD), real=False)
            
            fid_calcs['fid_PD_to_T2'].update(prepare_for_fid(real_T2), real=True)
            fid_calcs['fid_PD_to_T2'].update(prepare_for_fid(fake_T2_from_T1_from_PD), real=False)

            # if i < num_samples_to_save:
            save_test_outputs(real_T1, fake_T2_from_T1, real_T2, fake_PD_from_T2_from_T1, real_PD, "T1", "T2", "PD", samples_dir, i)
            save_test_outputs(real_T2, fake_PD_from_T2, real_PD, fake_T1_from_PD_from_T2, real_T1, "T2", "PD", "T1", samples_dir, i)
            save_test_outputs(real_PD, fake_T1_from_PD, real_T1, fake_T2_from_T1_from_PD, real_T2, "PD", "T1", "T2", samples_dir, i)

    # --- Finalize Metrics ---
    final_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    logging.info("Computing FID scores...")
    for key, fid_obj in fid_calcs.items():
        final_metrics[key] = fid_obj.compute().item()
        
    logging.info(f"Finished Model {model_idx+1}")
    return final_metrics

def main_pmc():
    parser = argparse.ArgumentParser(description="Multi-Model Test Script for Tri-Modal GAN.")
    # --- CHANGED: Accepts multiple paths ---
    parser.add_argument('--model_weights_paths', nargs='+', required=True, 
                        help='List of paths to best_model.pth files (e.g. path1.pth path2.pth).')
    
    parser.add_argument('--output_dir', type=str, default='/DATA1/RINKU/BRATS/brats_cycle/test_withoutfeaturematching/new_2048', help='Directory to save results.')
    parser.add_argument('--dataroot', type=str, default='/DATA1/RINKU/BRATS/PMCdataset/PMC dataset/2D_PNG_Format/test', help='Dataset root.')
    parser.add_argument('--field_strength', type=str, default='3T', help='Field strength.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers.')
    parser.add_argument('--num_samples', type=int, default=10, help='Image samples to save per model.')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID.')
    args = parser.parse_args()

    setup_test_logging(args.output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Models to evaluate: {len(args.model_weights_paths)}")

    # --- Data Loader ---
    transform = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_t1_dir = os.path.join(args.dataroot, args.field_strength, 'T1')
    test_t2_dir = os.path.join(args.dataroot, args.field_strength, 'T2')
    test_pd_dir = os.path.join(args.dataroot, args.field_strength, 'PD')
    try:
        logging.info(f"Loading test data from: {test_t1_dir}, {test_t2_dir}, {test_pd_dir}")
        test_dataset = MRIImageDataset(test_t1_dir, test_t2_dir, test_pd_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    except Exception as e:
        logging.error(f"Failed to create test dataloader: {e}", exc_info=True)
        return

    # --- Evaluation Loop ---
    all_models_results = []
    
    for idx, path in enumerate(args.model_weights_paths):
        result = test_single_model(path, test_loader, device, args.output_dir, args.num_samples, idx)
        if result is not None:
            all_models_results.append(result)

    if not all_models_results:
        logging.error("No models were successfully evaluated.")
        return

    # --- Aggregate Statistics ---
    logging.info("Calculating Aggregate Statistics...")
    keys = all_models_results[0].keys()
    aggregated_stats = {}
    
    for key in keys:
        values = [res[key] for res in all_models_results]
        aggregated_stats[key] = (np.mean(values), np.std(values))

    # --- Report ---
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("    REPRODUCIBILITY REPORT (Mean ± Std Dev)")
    report_lines.append("="*60)
    report_lines.append(f"Models Evaluated: {len(all_models_results)}")
    for p in args.model_weights_paths:
        report_lines.append(f" - {p}")
    report_lines.append("-" * 60)
    
    directions = ['T1_to_T2', 'T1_to_PD', 'T2_to_PD', 'T2_to_T1', 'PD_to_T1', 'PD_to_T2']
    
    for d in directions:
        report_lines.append(f"Direction: {d}")
        psnr_m, psnr_s = aggregated_stats.get(f'psnr_{d}', (0,0))
        ssim_m, ssim_s = aggregated_stats.get(f'ssim_{d}', (0,0))
        lpips_m, lpips_s = aggregated_stats.get(f'lpips_{d}', (0,0))
        fid_m, fid_s = aggregated_stats.get(f'fid_{d}', (0,0))
        
        report_lines.append(f"  PSNR : {psnr_m:.4f} ± {psnr_s:.4f}")
        report_lines.append(f"  SSIM : {ssim_m:.4f} ± {ssim_s:.4f}")
        report_lines.append(f"  LPIPS: {lpips_m:.4f} ± {lpips_s:.4f}")
        report_lines.append(f"  FID  : {fid_m:.4f} ± {fid_s:.4f}")
        report_lines.append("-" * 20)

    report_text = "\n".join(report_lines)
    print(report_text)
    
    final_report_path = os.path.join(args.output_dir, 'final_reproducibility_report.txt')
    with open(final_report_path, 'w') as f:
        f.write(report_text)
    logging.info(f"Final report saved to {final_report_path}")

if __name__ == '__main__':
    main()

##################################################
# BRATS METHODS 
##################################################

import os
import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# --- NEW IMPORT FOR FID ---
from torchmetrics.image.fid import FrechetInceptionDistance

# --- CUSTOM MODULES (Assumed to exist in your directory) ---
from generator import Generator_BRATS as Generator
from dataset import BraTSDataset
from metrics import calculate_psnr, calculate_ssim, MetricCalculator

# Initialize global metric calculator
metric_calculator = MetricCalculator(device="cuda:2" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn.functional as F



def smart_collate(batch):
    """
    Generalized collate for dictionary-based datasets.
    Pads all modalities (t1, t2, flair) in a batch to the same size
    and ensures dimensions are multiples of 4.
    """
    # 1. Filter out failed/None samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    
    # 2. Get the keys from the first sample (e.g., ['t1', 't2', 'flair'])
    keys = batch[0].keys()
    
    # 3. Find the maximum height and width across all samples and all modalities
    max_h = 0
    max_w = 0
    for sample in batch:
        for k in keys:
            img = sample[k]
            # Assumes img is a tensor of shape [C, H, W]
            max_h = max(max_h, img.shape[1])
            max_w = max(max_w, img.shape[2])
    
    # 4. Ensure dimensions are multiples of 4 for the Generator
    pad_h = (4 - max_h % 4) % 4
    pad_w = (4 - max_w % 4) % 4
    target_h, target_w = max_h + pad_h, max_w + pad_w

    output_dict = {}
    for k in ['t1', 't2', 'flair']:
        if k not in keys: continue
        modality_batch = []
        for sample in batch:
            img = sample[k]
            
            # Calculate padding for this specific image to reach target size
            diff_h = target_h - img.shape[1]
            diff_w = target_w - img.shape[2]
            
            # Pad: (left, right, top, bottom)
            padded_img = F.pad(img, (0, diff_w, 0, diff_h), mode='constant', value=0)
            modality_batch.append(padded_img)
            
        # Stack into a batch tensor: [B, C, H, W]
        output_dict[k] = torch.stack(modality_batch)

    if 'name' in keys:
        output_dict['name'] = [sample['name'] for sample in batch]

    return output_dict


def setup_test_logging(output_dir):
    """Sets up logging for the test script."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'test_run.log')
    # Clear existing handlers to prevent duplicate logs
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])

def tensor_to_pil(tensor):
    """Converts a single PyTorch image tensor to a PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    img_np = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)
    return Image.fromarray(img_np)

def prepare_for_fid(tensor):
    """
    Prepares a batch of tensors for FID calculation.
    FID expects: uint8 [0, 255], 3 channels (RGB).
    Input: [-1, 1], 1 channel (Grayscale).
    """
    # 1. Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    # 2. Convert to [0, 255] uint8
    tensor = (tensor * 255).to(dtype=torch.uint8)
    # 3. Expand 1 channel to 3 channels (Grayscale -> RGB)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor

def save_test_outputs(real_input, fake_step1, real_step1, fake_step2, real_step2,
                      input_modality, step1_modality, step2_modality,
                      output_dir, batch_idx):
    """Saves a well-labeled grid of images for visual comparison."""
    pil_real_input = tensor_to_pil(real_input)
    pil_fake_step1 = tensor_to_pil(fake_step1)
    pil_real_step1 = tensor_to_pil(real_step1)
    pil_fake_step2 = tensor_to_pil(fake_step2)
    pil_real_step2 = tensor_to_pil(real_step2)
    
    img_w, img_h = pil_real_input.size
    padding = 10
    label_height = 30
    grid_w = img_w * 2 + padding
    grid_h = (img_h + label_height) * 3 
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        
    y_offset = 0
    draw.text((padding, y_offset + 5), f"Input: {input_modality}", font=font, fill="black")
    grid.paste(pil_real_input, ((grid_w - img_w) // 2, y_offset + label_height))
    
    y_offset += img_h + label_height
    draw.text((padding, y_offset + 5), f"Generated: {step1_modality}", font=font, fill="blue")
    draw.text((padding + img_w, y_offset + 5), f"Ground Truth: {step1_modality}", font=font, fill="green")
    grid.paste(pil_fake_step1, (0, y_offset + label_height))
    grid.paste(pil_real_step1, (img_w + padding, y_offset + label_height))
    
    y_offset += img_h + label_height
    draw.text((padding, y_offset + 5), f"Generated: {step2_modality}", font=font, fill="blue")
    draw.text((padding + img_w, y_offset + 5), f"Ground Truth: {step2_modality}", font=font, fill="green")
    grid.paste(pil_fake_step2, (0, y_offset + label_height))
    grid.paste(pil_real_step2, (img_w + padding, y_offset + label_height))
    
    filename = os.path.join(output_dir, f"sample_batch_{batch_idx}_input_{input_modality}.png")
    grid.save(filename)

def test_single_model(model_path, test_loader, device, output_dir, num_samples_to_save, model_idx):
    """
    Evaluates a SINGLE set of model weights.
    Returns a dictionary of average metrics.
    """
    logging.info(f"--- Testing Model {model_idx+1}: {os.path.basename(model_path)} ---")
    
    # Initialize Generators
    G_T1_to_T2 = Generator().to(device)
    G_T2_to_PD = Generator().to(device)
    G_PD_to_T1 = Generator().to(device)
    
    # Load Weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        G_T1_to_T2.load_state_dict(checkpoint['G_T1_to_T2'])
        G_T2_to_PD.load_state_dict(checkpoint['G_T2_to_PD'])
        G_PD_to_T1.load_state_dict(checkpoint['G_PD_to_T1'])
    except Exception as e:
        logging.error(f"Failed to load weights from {model_path}: {e}")
        return None

    G_T1_to_T2.eval()
    G_T2_to_PD.eval()
    G_PD_to_T1.eval()
    
    # Create sub-directory for samples specific to this model
    samples_dir = os.path.join(output_dir, f'samples_model_{model_idx+1}')
    os.makedirs(samples_dir, exist_ok=True)

    # Initialize Metrics storage
    metrics = defaultdict(list)
    
    # --- FID INITIALIZATION ---
    # We need 6 FID calculators for the 6 translation directions
    fid_calcs = {
        'fid_T1_to_T2': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T1_to_FLAIR': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T2_to_FLAIR': FrechetInceptionDistance(feature=2048).to(device),
        'fid_T2_to_T1': FrechetInceptionDistance(feature=2048).to(device),
        'fid_FLAIR_to_T1': FrechetInceptionDistance(feature=2048).to(device),
        'fid_FLAIR_to_T2': FrechetInceptionDistance(feature=2048).to(device),
    }

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None: continue
            
            # Load Data
            real_T1 = batch["t1"].to(device)
            real_T2 = batch["t2"].to(device)
            real_FLAIR = batch["flair"].to(device)
            
            # --- GENERATION STEPS ---
            # Path 1: T1 -> T2 -> FLAIR
            fake_T2_from_T1, _ = G_T1_to_T2(real_T1)
            fake_FLAIR_from_T2_from_T1, _ = G_T2_to_PD(fake_T2_from_T1)
            
            # Path 2: T2 -> FLAIR -> T1
            fake_FLAIR_from_T2, _ = G_T2_to_PD(real_T2)
            fake_T1_from_FLAIR_from_T2, _ = G_PD_to_T1(fake_FLAIR_from_T2)
            
            # Path 3: FLAIR -> T1 -> T2
            fake_T1_from_FLAIR, _ = G_PD_to_T1(real_FLAIR)
            fake_T2_from_T1_from_FLAIR, _ = G_T1_to_T2(fake_T1_from_FLAIR)
            
            # --- CALCULATE PER-BATCH METRICS (PSNR, SSIM, LPIPS) ---
            # Helper to append metrics
            def record_metrics(real, fake, prefix):
                metrics[f'psnr_{prefix}'].append(calculate_psnr(real, fake))
                metrics[f'ssim_{prefix}'].append(calculate_ssim(real, fake))
                metrics[f'lpips_{prefix}'].append(metric_calculator.calculate_lpips(real, fake))

            record_metrics(real_T2, fake_T2_from_T1, 'T1_to_T2')
            record_metrics(real_FLAIR, fake_FLAIR_from_T2_from_T1, 'T1_to_FLAIR')
            record_metrics(real_FLAIR, fake_FLAIR_from_T2, 'T2_to_FLAIR')
            record_metrics(real_T1, fake_T1_from_FLAIR_from_T2, 'T2_to_T1')
            record_metrics(real_T1, fake_T1_from_FLAIR, 'FLAIR_to_T1')
            record_metrics(real_T2, fake_T2_from_T1_from_FLAIR, 'FLAIR_to_T2')

            # --- UPDATE FID STATS ---
            # Note: real=True/False. Updates internal statistics, does not return score yet.
            fid_calcs['fid_T1_to_T2'].update(prepare_for_fid(real_T2), real=True)
            fid_calcs['fid_T1_to_T2'].update(prepare_for_fid(fake_T2_from_T1), real=False)
            
            fid_calcs['fid_T1_to_FLAIR'].update(prepare_for_fid(real_FLAIR), real=True)
            fid_calcs['fid_T1_to_FLAIR'].update(prepare_for_fid(fake_FLAIR_from_T2_from_T1), real=False)
            
            fid_calcs['fid_T2_to_FLAIR'].update(prepare_for_fid(real_FLAIR), real=True)
            fid_calcs['fid_T2_to_FLAIR'].update(prepare_for_fid(fake_FLAIR_from_T2), real=False)
            
            fid_calcs['fid_T2_to_T1'].update(prepare_for_fid(real_T1), real=True)
            fid_calcs['fid_T2_to_T1'].update(prepare_for_fid(fake_T1_from_FLAIR_from_T2), real=False)
            
            fid_calcs['fid_FLAIR_to_T1'].update(prepare_for_fid(real_T1), real=True)
            fid_calcs['fid_FLAIR_to_T1'].update(prepare_for_fid(fake_T1_from_FLAIR), real=False)
            
            fid_calcs['fid_FLAIR_to_T2'].update(prepare_for_fid(real_T2), real=True)
            fid_calcs['fid_FLAIR_to_T2'].update(prepare_for_fid(fake_T2_from_T1_from_FLAIR), real=False)

            # Save Visual Samples
            if i < num_samples_to_save:
                save_test_outputs(real_T1, fake_T2_from_T1, real_T2, fake_FLAIR_from_T2_from_T1, real_FLAIR, "T1", "T2", "FLAIR", samples_dir, i)
                save_test_outputs(real_T2, fake_FLAIR_from_T2, real_FLAIR, fake_T1_from_FLAIR_from_T2, real_T1, "T2", "FLAIR", "T1", samples_dir, i)
                save_test_outputs(real_FLAIR, fake_T1_from_FLAIR, real_T1, fake_T2_from_T1_from_FLAIR, real_T2, "FLAIR", "T1", "T2", samples_dir, i)

    # --- COMPUTE FINAL METRICS ---
    # Average PSNR/SSIM/LPIPS
    final_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    # Compute FID (scalar)
    logging.info("Computing FID scores (this might take a moment)...")
    for key, fid_obj in fid_calcs.items():
        final_metrics[key] = fid_obj.compute().item()
        
    logging.info(f"Finished Model {model_idx+1}")
    return final_metrics

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def main_brats():
    parser = argparse.ArgumentParser(description="Multi-Model Test Script for Tri-Modal GAN on BraTS.")
    # MODIFIED: Accepts multiple paths now
    parser.add_argument('--model_weights_paths', nargs='+', required=True, 
                        help='List of paths to best_model.pth files (e.g. path1.pth path2.pth path3.pth).')
    parser.add_argument('--output_dir', type=str, default='test_results_reproducibility', help='Directory to save results.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of image samples to save per model.')
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID.')
    
    args = parser.parse_args()

    setup_test_logging(args.output_dir)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Found {len(args.model_weights_paths)} models to evaluate.")

    # # --- DATALOADER SETUP ---
    # transform = transforms.Compose([
    #     transforms.Resize((128, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])

    transform = transforms.Compose([
    # If the image is 240x240, this does nothing.
    # If the image is 500x500, it takes the middle 240x240.
    # This prevents the "stretching" (squashing) look.
    # transforms.CenterCrop(210), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    # test_path = "/home/m24csa032/.cache/kagglehub/datasets/varunraskar/brats-patient-split/versions/1/split_brats/test"
    test_path = "/DATA1/RINKU/.cache/kagglehub/datasets/varunraskar/brats-patient-split/versions/1/split_brats/test"
    try:
        logging.info(f"Loading Test Data from: {test_path}")
        if "OASIS" in test_path:
            from oasis_dataset import OasisNpyDataset
            test_dataset = OasisNpyDataset(root_dir=test_path)
        else:
            test_dataset = BraTSDataset(root_dir=test_path, transform=transform)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
        #                          num_workers=4, pin_memory=True, collate_fn=collate_fn)
        test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=smart_collate # Add this line
)
    
    except Exception as e:
        logging.error(f"Dataloader error: {e}", exc_info=True)
        return

    # --- EVALUATION LOOP ---
    all_models_results = []
    
    for idx, path in enumerate(args.model_weights_paths):
        result = test_single_model(path, test_loader, device, args.output_dir, args.num_samples, idx)
        if result is not None:
            all_models_results.append(result)

    if not all_models_results:
        logging.error("No models were successfully evaluated.")
        return

    # --- AGGREGATE STATISTICS (Mean +/- Std) ---
    logging.info("Calculating Aggregate Statistics...")
    
    keys = all_models_results[0].keys()
    aggregated_stats = {}
    
    for key in keys:
        values = [res[key] for res in all_models_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        aggregated_stats[key] = (mean_val, std_val)

    # --- REPORT GENERATION ---
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("    REPRODUCIBILITY REPORT (Mean ± Std Dev over 3 Runs)")
    report_lines.append("="*60)
    report_lines.append(f"Models Evaluated: {len(all_models_results)}")
    for p in args.model_weights_paths:
        report_lines.append(f" - {p}")
    report_lines.append("-" * 60)
    
    # Organize by translation direction
    directions = ['T1_to_T2', 'T1_to_FLAIR', 'T2_to_FLAIR', 'T2_to_T1', 'FLAIR_to_T1', 'FLAIR_to_T2']
    
    for d in directions:
        report_lines.append(f"Direction: {d}")
        
        # Add individual model performances to pinpoint the source of high variance
        for i, res in enumerate(all_models_results):
            psnr_val = res.get(f'psnr_{d}', 0)
            ssim_val = res.get(f'ssim_{d}', 0)
            lpips_val = res.get(f'lpips_{d}', 0)
            fid_val = res.get(f'fid_{d}', 0)
            report_lines.append(f"  Model {i+1}: PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, FID: {fid_val:.4f}")
            
        # Retrieve stats
        psnr_m, psnr_s = aggregated_stats.get(f'psnr_{d}', (0,0))
        ssim_m, ssim_s = aggregated_stats.get(f'ssim_{d}', (0,0))
        lpips_m, lpips_s = aggregated_stats.get(f'lpips_{d}', (0,0))
        fid_m, fid_s = aggregated_stats.get(f'fid_{d}', (0,0))
        
        report_lines.append(f"  [AGGREGATE] PSNR : {psnr_m:.4f} ± {psnr_s:.4f}")
        report_lines.append(f"  [AGGREGATE] SSIM : {ssim_m:.4f} ± {ssim_s:.4f}")
        report_lines.append(f"  [AGGREGATE] LPIPS: {lpips_m:.4f} ± {lpips_s:.4f}")
        report_lines.append(f"  [AGGREGATE] FID  : {fid_m:.4f} ± {fid_s:.4f}")
        report_lines.append("-" * 20)

    report_text = "\n".join(report_lines)
    print(report_text)
    
    final_report_path = os.path.join(args.output_dir, 'final_reproducibility_report.txt')
    with open(final_report_path, 'w') as f:
        f.write(report_text)
    logging.info(f"Final report saved to {final_report_path}")

if __name__ == '__main__':
    main()
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CycleGAN Models")
    parser.add_argument('--dataset', type=str, required=True, choices=['pmc', 'brats'], help="Select dataset to test.")
    
    # We parse known args so that dataset specific parsers can still work inside main_pmc or main_brats
    args, unknown = parser.parse_known_args()
    
    import sys
    # Reconstruct sys.argv for the downstream arg parsers
    sys.argv = [sys.argv[0]] + unknown
    
    if args.dataset == 'pmc':
        main_pmc()
    elif args.dataset == 'brats':
        main_brats()
