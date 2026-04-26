import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime

# Import merged logic
from train import (
    train_model_pmc, initialize_loss_optimizers_pmc, weights_init as weights_init_pmc,
    train_model_brats, initialize_loss_optimizers_brats
)
from generator import Generator_PMC, Generator_BRATS
from discriminator import Discriminator
from dataset import MRIImageDataset, BraTSDataset
from metrics import MetricCalculator

import torch.nn.functional as F

# Helper function for BRATS collation
def smart_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    keys = batch[0].keys()
    max_h = 0
    max_w = 0
    for sample in batch:
        for k in keys:
            img = sample[k]
            max_h = max(max_h, img.shape[1])
            max_w = max(max_w, img.shape[2])
    pad_h = (4 - max_h % 4) % 4
    pad_w = (4 - max_w % 4) % 4
    target_h, target_w = max_h + pad_h, max_w + pad_w
    output_dict = {}
    for k in keys:
        modality_batch = []
        for sample in batch:
            img = sample[k]
            diff_h = target_h - img.shape[1]
            diff_w = target_w - img.shape[2]
            padded_img = F.pad(img, (0, diff_w, 0, diff_h), mode='constant', value=0)
            modality_batch.append(padded_img)
        output_dict[k] = torch.stack(modality_batch)
    return output_dict

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN for MRI Image Translation")
    parser.add_argument('--dataset', type=str, required=True, choices=['pmc', 'brats'],
                        help='Which dataset to use: pmc or brats')
    parser.add_argument('--field_strength', type=str, default='3T',
                        help='Field strength for PMC dataset (e.g., 1.5T or 3T)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output models and logs')
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of epochs for training')
    args = parser.parse_args()

    BATCH_SIZE = 16
    today = datetime.now().strftime('%Y-%m-%d')
    experiment_dir = os.path.join(args.output_dir, f"best_trial_{today}")
    os.makedirs(experiment_dir, exist_ok=True)
    setup_logging(experiment_dir)

    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if args.dataset == 'pmc':
        train_pmc(args, device, experiment_dir, BATCH_SIZE)
    elif args.dataset == 'brats':
        train_brats(args, device, experiment_dir, BATCH_SIZE)


def train_pmc(args, device, experiment_dir, BATCH_SIZE):
    metric_calculator = MetricCalculator(device=device)

    best_hyperparams = {
        "lr_G_T1_to_T2": 0.0002774699967880093,
        "lr_G_T2_to_PD": 0.0003272728936133143,
        "lr_G_PD_to_T1": 0.00041420693929415865,
        "lr_D_T1": 0.00015799131283236812,
        "lr_D_T2": 7.853281407067347e-05,
        "lr_D_PD": 8.59323952929778e-05,
        "lambda_GAN_T1_T2": 0.5821156093873405,
        "lambda_GAN_T2_PD": 1.2321606797886726,
        "lambda_GAN_PD_T1": 0.6754468292750628,
        "lambda_cycle_T1": 10.214697984886651,
        "lambda_cycle_PD": 5.104570841026641 ,
        "lambda_cycle_T2": 14.669133358300012,
        "lambda_identity_T1": 4.421440994798656 ,
        "lambda_identity_T2": 0.6764764601554305,
        "lambda_identity_PD": 9.74841502432547  ,
        "lambda_fm_D_T1":12.96522503592844,
        "lambda_fm_D_T2":12.959832198680976,
        "lambda_fm_D_PD":5.866689123670891
    }

    lambda_dict = {
        'lambda_GAN_T1_T2': best_hyperparams["lambda_GAN_T1_T2"],
        'lambda_GAN_T2_PD': best_hyperparams["lambda_GAN_T2_PD"],
        'lambda_GAN_PD_T1': best_hyperparams["lambda_GAN_PD_T1"],
        'lambda_cycle_T1': best_hyperparams["lambda_cycle_T1"],
        'lambda_cycle_PD': best_hyperparams["lambda_cycle_PD"],
        'lambda_cycle_T2': best_hyperparams["lambda_cycle_T2"],
        'lambda_identity_T1': best_hyperparams["lambda_identity_T1"],
        'lambda_identity_T2': best_hyperparams["lambda_identity_T2"],
        'lambda_identity_PD': best_hyperparams["lambda_identity_PD"],
        'lambda_fm_D_T1': best_hyperparams["lambda_fm_D_T1"],
        'lambda_fm_D_T2': best_hyperparams["lambda_fm_D_T2"],
        'lambda_fm_D_PD': best_hyperparams["lambda_fm_D_PD"],
    }

    transform = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_t1_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/train/{args.field_strength}/T1'
    train_t2_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/train/{args.field_strength}/T2'
    train_pd_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/train/{args.field_strength}/PD'
 
    test_t1_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/validation/{args.field_strength}/T1'
    test_t2_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/validation/{args.field_strength}/T2'
    test_pd_dir = f'/home/PMCdataset/PMC dataset/2D_PNG_Format/validation/{args.field_strength}/PD'

    train_dataset = MRIImageDataset(train_t1_dir, train_t2_dir, train_pd_dir, transform)
    test_dataset = MRIImageDataset(test_t1_dir, test_t2_dir, test_pd_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    G_T1_to_T2 = Generator_PMC().to(device)
    G_T2_to_PD = Generator_PMC().to(device)
    G_PD_to_T1 = Generator_PMC().to(device)
    D_T1 = Discriminator().to(device)
    D_T2 = Discriminator().to(device)
    D_PD = Discriminator().to(device)

    G_T1_to_T2.apply(weights_init_pmc)
    G_T2_to_PD.apply(weights_init_pmc)
    G_PD_to_T1.apply(weights_init_pmc)
    D_T1.apply(weights_init_pmc)
    D_T2.apply(weights_init_pmc)
    D_PD.apply(weights_init_pmc)

    (criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching, optimizer_G, optimizer_D, scheduler_G, scheduler_D) = initialize_loss_optimizers_pmc(
        G_T1_to_T2, G_T2_to_PD, G_PD_to_T1, D_T1, D_T2, D_PD,
        lr_G_T1_to_T2=best_hyperparams["lr_G_T1_to_T2"], 
        lr_G_T2_to_PD=best_hyperparams["lr_G_T2_to_PD"],
        lr_G_PD_to_T1=best_hyperparams["lr_G_PD_to_T1"],
        lr_D_T1=best_hyperparams["lr_D_T1"],
        lr_D_T2=best_hyperparams["lr_D_T2"],    
        lr_D_PD=best_hyperparams["lr_D_PD"]
    )

    best_ssim, best_psnr, best_lpips = train_model_pmc(
        n_epochs=args.n_epochs, train_loader=train_loader, val_loader=val_loader,
        G_T1_to_T2=G_T1_to_T2, G_T2_to_PD=G_T2_to_PD, G_PD_to_T1=G_PD_to_T1,
        D_T1=D_T1, D_T2=D_T2, D_PD=D_PD,
        device=device, criterion_GAN=criterion_GAN, criterion_cycle=criterion_cycle,
        criterion_identity=criterion_identity, criterion_feature_matching=criterion_feature_matching,
        optimizer_G=optimizer_G, optimizer_D=optimizer_D, scheduler_G=scheduler_G, scheduler_D=scheduler_D,
        experiment_dir=experiment_dir, lambda_dict=lambda_dict, metric_calculator=metric_calculator
    )
    logging.info(f"PMC Training completed with Best SSIM: {best_ssim:.4f}, Best PSNR: {best_psnr:.4f}, Best LPIPS: {best_lpips:.4f}")


def train_brats(args, device, experiment_dir, BATCH_SIZE):
    metric_calculator = MetricCalculator(device=device)

    best_hyperparams = {
        'lr_G_T1_to_T2': 0.0001772463180372168, 'lr_G_T2_to_PD': 0.00023610711556838363, 'lr_G_PD_to_T1': 0.00013182074240921824, 
        'lr_D_T1': 3.9667359561883534e-05, 'lr_D_T2': 2.3643168411963974e-05, 'lr_D_PD': 0.00012002769808376257, 
        'lambda_GAN_T1_T2': 1.309997347255265, 'lambda_GAN_T2_PD': 1.0377150812588023, 'lambda_GAN_PD_T1': 1.9699437632916301, 
        'lambda_cycle_T1': 10.982405267566914, 'lambda_cycle_PD': 11.812475625018056, 'lambda_cycle_T2': 11.926207141792489, 
        'lambda_identity_T1': 3.184981584775155, 'lambda_identity_T2': 1.9862693610122726, 'lambda_identity_PD': 7.162400416065927, 
        'lambda_fm_D_T1': 8.312380392757333, 'lambda_fm_D_T2': 9.854661924561093, 'lambda_fm_D_PD': 14.275700217863216
    }
    
    lambda_dict = {
        'lambda_GAN_T1_T2': best_hyperparams["lambda_GAN_T1_T2"],
        'lambda_GAN_T2_PD': best_hyperparams["lambda_GAN_T2_PD"],
        'lambda_GAN_PD_T1': best_hyperparams["lambda_GAN_PD_T1"],
        'lambda_cycle_T1': best_hyperparams["lambda_cycle_T1"],
        'lambda_cycle_PD': best_hyperparams["lambda_cycle_PD"],
        'lambda_cycle_T2': best_hyperparams["lambda_cycle_T2"],
        'lambda_identity_T1': best_hyperparams["lambda_identity_T1"],
        'lambda_identity_T2': best_hyperparams["lambda_identity_T2"],
        'lambda_identity_PD': best_hyperparams["lambda_identity_PD"],
        'lambda_fm_D_T1': best_hyperparams["lambda_fm_D_T1"],
        'lambda_fm_D_T2': best_hyperparams["lambda_fm_D_T2"],
        'lambda_fm_D_PD': best_hyperparams["lambda_fm_D_PD"],
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data_path = "/home/brats/train"
    val_data_path = "/home/brats/val"
    train_dataset = BraTSDataset(root_dir=train_data_path, transform=transform)
    val_dataset = BraTSDataset(root_dir=val_data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=smart_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=smart_collate)

    G_T1_to_T2 = Generator_BRATS().to(device)
    G_T2_to_FLAIR = Generator_BRATS().to(device)
    G_FLAIR_to_T1 = Generator_BRATS().to(device)
    D_T1 = Discriminator().to(device)
    D_T2 = Discriminator().to(device)
    D_FLAIR = Discriminator().to(device)

    G_T1_to_T2.apply(weights_init_pmc)  # Using the general weights_init function
    G_T2_to_FLAIR.apply(weights_init_pmc)
    G_FLAIR_to_T1.apply(weights_init_pmc)
    D_T1.apply(weights_init_pmc)
    D_T2.apply(weights_init_pmc)
    D_FLAIR.apply(weights_init_pmc)

    (criterion_GAN, criterion_cycle, criterion_identity, criterion_feature_matching, optimizer_G, optimizer_D, scheduler_G, scheduler_D) = initialize_loss_optimizers_brats(
        G_T1_to_T2, G_T2_to_FLAIR, G_FLAIR_to_T1, D_T1, D_T2, D_FLAIR,
        lr_G_T1_to_T2=best_hyperparams["lr_G_T1_to_T2"],
        lr_G_T2_to_PD=best_hyperparams["lr_G_T2_to_PD"],
        lr_G_PD_to_T1=best_hyperparams["lr_G_PD_to_T1"],
        lr_D_T1=best_hyperparams["lr_D_T1"],
        lr_D_T2=best_hyperparams["lr_D_T2"],
        lr_D_PD=best_hyperparams["lr_D_PD"]
    )

    best_ssim, best_psnr, best_lpips = train_model_brats(
        n_epochs=args.n_epochs, train_loader=train_loader, val_loader=val_loader,
        G_T1_to_T2=G_T1_to_T2, G_T2_to_PD=G_T2_to_FLAIR, G_PD_to_T1=G_FLAIR_to_T1,
        D_T1=D_T1, D_T2=D_T2, D_PD=D_FLAIR,
        device=device, criterion_GAN=criterion_GAN, criterion_cycle=criterion_cycle,
        criterion_identity=criterion_identity, criterion_feature_matching=criterion_feature_matching,
        optimizer_G=optimizer_G, optimizer_D=optimizer_D, scheduler_G=scheduler_G, scheduler_D=scheduler_D,
        experiment_dir=experiment_dir, lambda_dict=lambda_dict, metric_calculator=metric_calculator
    )

    logging.info(f"BRATS Training completed with Best SSIM: {best_ssim:.4f}, Best PSNR: {best_psnr:.4f}, Best LPIPS: {best_lpips:.4f}")

if __name__ == "__main__":
    main()
