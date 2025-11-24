import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

preprocessed_image_folders = {
    "OA": './processed_data/OA',
    "ICA": './processed_data/ICA',
    "ICA2": './processed_data/ICA2',
    "Cube96": "./processed_data/Cube96"
}

intensities = []
is_empty_mask = []
flat_images = []

for folder_name, folder_path in preprocessed_image_folders.items():
    for file in os.listdir(folder_path):
        data = np.load(os.path.join(folder_path, file))
        image = data['image']
        mask = data['label']

        average_pixel_intensity = image.mean()
        mask_empty = int(mask.sum() == 0)

        intensities.append(average_pixel_intensity)
        is_empty_mask.append(mask_empty)

        flat_images.append(image.flatten())

intensities = np.array(intensities)
is_empty_mask = np.array(is_empty_mask)
flat_images = np.array(flat_images)

corr, pval = pearsonr(intensities, is_empty_mask)

print("Correlation:", corr)
print("p-value:", pval)
print("Mean intensity for empty masks:", intensities[is_empty_mask == 1].mean())
print("Mean intensity for non-empty masks:", intensities[is_empty_mask == 0].mean())

plt.hist(intensities[is_empty_mask == 1], bins=40, alpha=0.6, label="empty mask")
plt.hist(intensities[is_empty_mask == 0], bins=40, alpha=0.6, label="non-empty mask")
plt.legend()
plt.xlabel("Avg pixel intensity")
plt.ylabel("Count")
plt.show()


flat_norm = (flat_images - flat_images.mean(axis=1, keepdims=True)) / \
            (flat_images.std(axis=1, keepdims=True) + 1e-6)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    verbose=1
)

tsne_emb = tsne.fit_transform(flat_images)

plt.figure(figsize=(8, 6))
plt.scatter(
    tsne_emb[:,0], tsne_emb[:,1],
    c=is_empty_mask,
    cmap="coolwarm",
    alpha=0.6
)
plt.title("t-SNE on OCT Images (red=empty mask, blue=non-empty)")
plt.show()


