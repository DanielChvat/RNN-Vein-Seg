from PIL import Image
import os
import numpy as np

TARGET_SIZE = (224, 224)

def resize_to_target_size(img_array: np.ndarray, target_size=TARGET_SIZE, is_label=False):
    """Resizes a 2D array to the target size using interpolation."""
    img = Image.fromarray(img_array)

    if is_label:
        img = img.resize((target_size[1], target_size[0]), Image.NEAREST)
    else:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    return np.array(img, dtype=img_array.dtype)


def create_npz_slices(img_folder: str, label_folder: str, output_folder: str, prefix: str):
    """
    Loads image/label slices, resizes them, normalizes image slices,
    and saves each pair as an .npz file: {image, label}.
    """

    os.makedirs(output_folder, exist_ok=True)

    img_paths = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder)])
    lbl_paths = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder)])

    print(f"Found {len(img_paths)} image slices and {len(lbl_paths)} label slices.")

    for idx, (img_p, lbl_p) in enumerate(zip(img_paths, lbl_paths)):

        # Load + resize
        img = resize_to_target_size(np.array(Image.open(img_p), dtype=np.float32), is_label=False)
        lbl = resize_to_target_size(np.array(Image.open(lbl_p), dtype=np.float32), is_label=True)

        # Clip & normalize image slice
        img = np.clip(img, -125, 275)
        min_v, max_v = img.min(), img.max()
        img = (img - min_v) / (max_v - min_v + 1e-8)

        save_path = os.path.join(output_folder, f"{prefix}_slice_{idx:04d}.npz")
        np.savez_compressed(save_path, image=img, label=lbl)

    print(f"Saved {idx+1} NPZ slices to {output_folder}")


if __name__ == "__main__":

    datasets = {
        "OA": ("./raw_data/OA/imgs", "./raw_data/OA/masks"),
        "ICA": ("./raw_data/ICA/imgs", "./raw_data/ICA/masks"),
        "ICA2": ("./raw_data/ICA2/imgs", "./raw_data/ICA2/masks"),
        "Cube96": ("./raw_data/Cube96/imgs", "./raw_data/Cube96/masks"),
        "Cube15": ("./raw_data/Cube15/imgs", "./raw_data/Cube15/masks"),
        "Cube16": ("./raw_data/Cube16/imgs", "./raw_data/Cube16/masks"),
        "Cube95": ("./raw_data/Cube95/imgs", "./raw_data/Cube95/masks")
    }

    for name, (img_folder, lbl_folder) in datasets.items():
        print(f"\nProcessing dataset: {name}")

        output_folder = f"./processed_data/{name}"

        create_npz_slices(
            img_folder=img_folder,
            label_folder=lbl_folder,
            output_folder=output_folder,
            prefix=f"CASE_{name}"
        )