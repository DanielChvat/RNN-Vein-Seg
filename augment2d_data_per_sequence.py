import os
import numpy as np
import albumentations as A
import random

# Function to generate a random seed
def get_random_seed():
    return random.randint(0, 2**32 - 1)  # Generate a random 32-bit integer seed

# Define your 2D augmentations
def get_augmentations(seed):
    random.seed(seed)
    np.random.seed(seed)

    return A.Compose([
        A.Affine(
            scale=(0.9, 1.0),
            translate_percent=(0.0, 0.05),
            rotate=0,
            fit_output=False,
            p=1
        ),

        A.ElasticTransform(
            alpha=5,
            sigma=3,
            approximate=True,
            p=1
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=1
        ),

        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0,
            hue=0,
            p=1
        ),

        A.GaussNoise(
            std_range=(0.005, 0.1),
            mean_range=(0, 0),
            per_channel=False,
            noise_scale_factor=1.0,
            p=1
        ),

    ], additional_targets={'mask': 'mask'}, seed=seed)

def augment_npz_volume(npz_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(npz_folder) if f.endswith(".npz")])

    seed = get_random_seed()

    augment = get_augmentations(seed)

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
    OUT_ROOT = "./filtered_data_augmented"

    DATASETS = ["OA", "ICA", "ICA2", "Cube96", "Cube15", "Cube95", "Cube16"]
    NUM_AUGMENTS = 5

    for name in DATASETS:
        print(f"=== Augmenting: {name} ===")

        npz_dir = os.path.join(ROOT, name)

        # Make K augmented copies
        for i in range(1, NUM_AUGMENTS + 1):
            out_dir = os.path.join(OUT_ROOT, f"{name}_AUG_{i}")
            print(f" → Generating augmented dataset #{i}: {out_dir}")
            augment_npz_volume(npz_dir, out_dir)