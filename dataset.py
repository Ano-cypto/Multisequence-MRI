import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

class MRIImageDataset(Dataset):
    def __init__(self, t1_images_dir, t2_images_dir, pd_images_dir, transform):
        """
        Dataset class for loading T1, T2, and PD MRI image triplets.
        
        Args:
            t1_images_dir (str): Directory containing T1-weighted images
            t2_images_dir (str): Directory containing T2-weighted images
            pd_images_dir (str): Directory containing PD-weighted images
            transform (callable, optional): Optional transform to be applied to images
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        # Extract identifiers from filenames (assuming format: [ID]_[MODALITY].ext)
        def extract_id(file_path, modality):
            base = os.path.basename(file_path)
            # Remove modality suffix and extension
            return base.replace(f"_{modality}", "").rsplit(".", 1)[0]

        # Build dictionaries of {id: file_path} for each modality
        t1_files = {
            extract_id(f, "T1"): os.path.join(t1_images_dir, f)
            for f in os.listdir(t1_images_dir) 
            if f.lower().endswith(valid_extensions)
        }
        
        t2_files = {
            extract_id(f, "T2"): os.path.join(t2_images_dir, f)
            for f in os.listdir(t2_images_dir)
            if f.lower().endswith(valid_extensions)
        }
        
        pd_files = {
            extract_id(f, "PD"): os.path.join(pd_images_dir, f)
            for f in os.listdir(pd_images_dir)
            if f.lower().endswith(valid_extensions)
        }

        # Debugging: Print sample counts and identifiers
        print(f"Found {len(t1_files)} T1 images, {len(t2_files)} T2 images, {len(pd_files)} PD images")
        print("T1 Sample IDs:", sorted(t1_files.keys())[:5])
        print("T2 Sample IDs:", sorted(t2_files.keys())[:5])
        print("PD Sample IDs:", sorted(pd_files.keys())[:5])

        # Find common identifiers across all three modalities
        common_ids = set(t1_files.keys()) & set(t2_files.keys()) & set(pd_files.keys())
        common_ids = sorted(common_ids)
        
        print(f"Found {len(common_ids)} matching triplets")
        print("Common IDs Sample:", common_ids[:5])

        if not common_ids:
            raise ValueError(
                "No matching T1-T2-PD image triplets found! "
                "Check that files follow naming convention: [ID]_[MODALITY].ext"
            )

        # Store only matched triplets
        self.t1_image_paths = [t1_files[id] for id in common_ids]
        self.t2_image_paths = [t2_files[id] for id in common_ids]
        self.pd_image_paths = [pd_files[id] for id in common_ids]
        self.transform = transform
        self.common_ids = common_ids  # Store for reference

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
        {
            't1': T1-weighted image,
            't2': T2-weighted image,
            'pd': PD-weighted image,
            'id': patient/scan identifier
        }
        """
        try:
            # Load all three modalities
            t1_img = Image.open(self.t1_image_paths[idx]).convert('L')
            t2_img = Image.open(self.t2_image_paths[idx]).convert('L')
            pd_img = Image.open(self.pd_image_paths[idx]).convert('L')

            if self.transform:
                t1_img = self.transform(t1_img)
                t2_img = self.transform(t2_img)
                pd_img = self.transform(pd_img)

            return {
                't1': t1_img,
                't2': t2_img,
                'pd': pd_img,
                'id': self.common_ids[idx]  # Include identifier for tracking
            }

        except Exception as e:
            print(f"Error loading image triplet {idx}: {e}")
            print(f"T1 path: {self.t1_image_paths[idx]}")
            print(f"T2 path: {self.t2_image_paths[idx]}")
            print(f"PD path: {self.pd_image_paths[idx]}")
            return None  # Skip corrupted images

    def get_sample_ids(self):
        """Returns list of all sample IDs in the dataset"""
        return self.common_ids

    def verify_image(self, idx):
        """Verifies that an image triplet can be loaded successfully"""
        try:
            sample = self.__getitem__(idx)
            if sample is None:
                return False
            # Check all images have same dimensions
            shapes = [sample['t1'].shape, sample['t2'].shape, sample['pd'].shape]
            if len(set(shapes)) != 1:
                print(f"Shape mismatch in triplet {idx}: T1 {shapes[0]}, T2 {shapes[1]}, PD {shapes[2]}")
                return False
            return True
        except:
            return False

    def remove_corrupted(self, verbose=True):
        """Remove corrupted image triplets from the dataset"""
        valid_indices = []
        for idx in range(len(self)):
            if self.verify_image(idx):
                valid_indices.append(idx)
            elif verbose:
                print(f"Removing corrupted triplet {idx}")

        # Update paths and IDs
        self.t1_image_paths = [self.t1_image_paths[i] for i in valid_indices]
        self.t2_image_paths = [self.t2_image_paths[i] for i in valid_indices]
        self.pd_image_paths = [self.pd_image_paths[i] for i in valid_indices]
        self.common_ids = [self.common_ids[i] for i in valid_indices]

        if verbose:
            print(f"Kept {len(valid_indices)} valid triplets after removal")


class BraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.modalities = ['t1', 't2', 'flair']
        self.transform = transform
        self.samples = []
        self._build_file_list()

    def _build_file_list(self):
        # We will use the 't1' folder as the master list of files
        t1_dir = os.path.join(self.root_dir,'t1')
        if not os.path.isdir(t1_dir):
            raise FileNotFoundError(f"t1 directory not found at {t1_dir}")

        # Get all filenames from the t1 directory
        all_t1_files = sorted([f for f in os.listdir(t1_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        logging.info(f"Found {len(all_t1_files)} potential T1 slices.")

        # Check for corresponding files in other modalities
        for file_name in all_t1_files:
            # CORRECTED LOGIC: Remove the 't1_' prefix to get the base filename
            base_file_name = file_name.replace('t1_', '')

            # Construct paths for T2 and FLAIR using the corrected base filename
            t2_path = os.path.join(self.root_dir, 't2', f't2_{base_file_name}')
            flair_path = os.path.join(self.root_dir, 'flair', f'flair_{base_file_name}')

            # Check if all corresponding files exist
            if os.path.exists(t2_path) and os.path.exists(flair_path):
                self.samples.append({
                    't1': os.path.join(t1_dir, file_name),
                    't2': t2_path,
                    'flair': flair_path
                })
            else:
                logging.warning(f"Skipping file {file_name} due to missing corresponding files.")
        
        if not self.samples:
            raise RuntimeError(f"No valid slice triplets found in {self.root_dir}. Check file names and structure.")
        
        logging.info(f"Successfully built dataset with {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]
        try:
            image_t1 = Image.open(sample_paths['t1']).convert('L')
            image_t2 = Image.open(sample_paths['t2']).convert('L')
            image_flair = Image.open(sample_paths['flair']).convert('L')

            if self.transform:
                image_t1 = self.transform(image_t1)
                image_t2 = self.transform(image_t2)
                image_flair = self.transform(image_flair)

            return {
                't1': image_t1,
                't2': image_t2,
                'flair': image_flair, 
            }
        except Exception as e:
            logging.error(f"Error loading image files from sample {idx}: {e}")
            return None