# Controlling Error Propagation in One-to-Many MRI Sequence Synthesis via Intermediate Distribution Alignment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Stable-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for the paper: **"Controlling Error Propagation in One-to-Many MRI Sequence Synthesis via Intermediate Distribution Alignment."** This framework proposes a unified generative chain capable of stable, one-to-many synthesis of all missing MRI sequences from a single acquired input sequence. By leveraging **Structured Distribution-Aligned Curriculum Learning**, the model successfully mitigates the compounding error propagation typically seen in cascaded multi-hop generators.

## 📌 Repository Overview

Previously, separate sets of files were used for logic tied to PMC (T1, T2, PD) and BraTS 2021 (T1, T2, FLAIR). These have now been merged so that a single unified command-line parameter dictates which specific configuration and architecture logic runs.

* `main.py`: The merged main execution entry point. Uses `--dataset` to run either PMC or BraTS.
* `dataset.py`: Merged dataset logic comprising both `MRIImageDataset` (for PMC) and `BraTSDataset`.
* `generator.py`: Contains both `Generator_PMC` and `Generator_BRATS` under a unified generative chain architecture.
* `discriminator.py`: Implementation of the patch-based adversarial discriminators.
* `train.py`: Contains separated, conditionally invoked training loops (`train_model_pmc` and `train_model_brats`) integrating the curriculum learning schedule.
* `test.py`: A unified test execution script for inference and metric computation.
* `metrics.py` & `visualization.py`: Utilities for computing PSNR, SSIM, LPIPS, FID, and generating FFT/error maps.

## ⚙️ Environment Setup

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment or Conda.

```bash
# Clone the repository
git clone [(https://github.com/RINKUSADH/Multi-sequence_MRI.git)]
cd Multi-sequence_MRI

# Install required dependencies
pip install -r requirements.txt
```

*(Dependencies include `torch`, `torchvision`, `numpy`, `opencv-python`, `nibabel`, `SimpleITK`, and `scipy`)*

## 📂 Supported Dataset Formats

To train or evaluate the model, your datasets must be structured as follows:

### 1. PMC Dataset Format (3T / 1.5T)
The PMC Dataset logic expects three distinct modality folders under the specified field strength:

```text
PMC_Dataset/
├── T1/
│   ├── Patient001_slice15_T1.png
├── T2/
│   ├── Patient001_slice15_T2.png
├── PD/
│   ├── Patient001_slice15_PD.png
```

*Note: Each triplet must share a common `[ID]` prefix, with the final suffix reflecting the modality.*

### 2. BraTS 2021 Dataset Format
The BraTS dataset logic handles triplets composed of T1, T2, and FLAIR imaging.

```text
BraTS_Dataset/
├── t1/
│   ├── t1_slice010.png
├── t2/
│   ├── t2_slice010.png
├── flair/
│   ├── flair_slice010.png
```

*Note: The dataloader determines the common identifier by dropping the predefined prefixes.*

## 🚀 Usage

### Training the Model
You can initiate training by selecting the target dataset via the `--dataset` argument. The curriculum learning schedule is automatically applied during training.

**Train on PMC Dataset (e.g., 3T Field Strength):**
```bash
python main.py --dataset pmc --field_strength 3T --output_dir ./models_pmc --n_epochs 200
```

**Train on BraTS 2021 Dataset:**
```bash
python main.py --dataset brats --output_dir ./models_brats --n_epochs 100
```

### Inference and Testing
To generate synthetic sequences and evaluate performance metrics (PSNR, SSIM, LPIPS, FID), use the unified testing script:

```bash
# Test PMC Model
python test.py --dataset pmc --model_path ./models_pmc/best_generator.pth

# Test BraTS Model
python test.py --dataset brats --model_path ./models_brats/best_generator.pth
```

### Few-Shot Domain Adaptation
For cross-institutional (UTSW-Glioma) or cross-scanner (PMC 1.5T) generalization using the 5% few-shot methodology described in the paper, append the `--few_shot 0.05` flag to the training command along with the target dataset path.


## 📬 Contact
For any inquiries regarding the code, implementation details, or the dataset processing scripts (including the 3D-to-2D NIfTI slicer), please open an issue in this repository.
