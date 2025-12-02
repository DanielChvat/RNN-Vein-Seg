import os
import numpy as np
import albumentations as A
import random

# Function to generate a random seed
def get_random_seed():
    return random.randint(0, 2**32 - 1)  # Generate a random 32-bit integer seed

# Define your 2D augmentations
def get_augmentations(seed):
    return A.Compose([
        A.HorizontalFlip(p=1.0),  # Always apply
         # Small smooth geometric distortions
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=10,
            border_mode=0,  # reflect/constant ok
            p=1.0
        ),
        # Elastic deformation
        A.ElasticTransform(
            alpha=20,
            sigma=3,
            alpha_affine=10,
            p=1.0
        ),
        # Intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=1.0
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.GaussianNoise(var_limit=(5, 25), p=1.0),
        # Local histogram equalization
        A.CLAHE(clip_limit=2.0, p=1.0),
    ], additional_targets={'mask': 'mask'}, seed=seed)  # Set seed for reproducibility

def augment_npz_volume(npz_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    # Get all .npz files in the folder
    npz_files = sorted([f for f in os.listdir(npz_folder) if f.endswith(".npz")])

    # Generate a random seed for the augmentations
    seed = get_random_seed()

    # Get the augmentations with the random seed
    augment = get_augmentations(seed)

    # Apply the same transform to every slice
    for npz_f in npz_files:
        path = os.path.join(npz_folder, npz_f)
        data = np.load(path)
        img = data['image']
        mask = data['label']

        augmented = augment(image=img, mask=mask)

        transformed = augmented["image"]
        transformed_mask = augmented["mask"]

        # Save as new NPZ
        save_path = os.path.join(out_folder, npz_f)
        np.savez_compressed(save_path, image=transformed, label=transformed_mask)


if __name__ == "__main__":
    ROOT = "./filtered_data"
    DATASETS = ["OA", "ICA", "ICA2", "Cube96", "Cube15", "Cube95", "Cube16"]

    for name in DATASETS:
        print(f"Augmenting volume (2D consistent) for: {name}")
        npz_dir = os.path.join(ROOT, name)
        out_dir = os.path.join(ROOT, f"{name}_AUG")
        augment_npz_volume(npz_dir, out_dir)
