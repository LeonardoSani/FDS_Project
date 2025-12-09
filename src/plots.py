import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np

import src.VaeA as vA


def plot_distribution(data, title='Data Distribution', ax =None):
    type_counts = pd.Series(data).value_counts()
    df_plot = pd.DataFrame({'Category': type_counts.index.astype(str), 'Count': type_counts.values})


    sns.barplot(x='Category', y='Count', hue='Category', data=df_plot, palette='viridis', legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.grid(axis='y', linestyle='--', alpha=0.7)



def visualize_augmentations(new_images, n_show=5, normalize='tanh'):
    plt.figure(figsize=(15, 3))
    for i in range(n_show):
        plt.subplot(1, n_show, i + 1)
        if normalize == 'tanh':
            plt.imshow(new_images[i], cmap='gray', vmin=-1, vmax=1)
        else:
            plt.imshow(new_images[i], cmap='gray')
        plt.axis('off')
        plt.title(f"Gen Sample {i+1}")
    plt.show()


def visualize_interpolation(model, X_train_vae, n_steps=10, t=5):
    """
    Visualize latent space SLERP interpolation between two random images.
    
    Args:
        model: VAE model with encoder and decoder
        X_train_vae: Training data array
        n_steps: Number of interpolation steps
        t: Threshold distance for selecting close enough latent vectors
    """
    
    # Repeat until mu1 and mu2 are close enough (distance < t)
    while True:
        # Select two random defect images
        idx1, idx2 = np.random.choice(len(X_train_vae), 2, replace=False)
        img1 = X_train_vae[idx1:idx1+1]
        img2 = X_train_vae[idx2:idx2+1]

        # Add channel dimension if needed and convert to tensor
        if img1.ndim == 3:
            img1 = torch.tensor(img1[:, np.newaxis, :, :], dtype=torch.float32)
            img2 = torch.tensor(img2[:, np.newaxis, :, :], dtype=torch.float32)
        else:
            img1 = torch.tensor(img1, dtype=torch.float32)
            img2 = torch.tensor(img2, dtype=torch.float32)

        # Encode both images to latent space
        with torch.no_grad():
            mu1, _ = model.encoder(img1)
            mu2, _ = model.encoder(img2)

        # Check the distance between mu1 and mu2
        if torch.norm(mu1 - mu2) < t:
            break

    # Create interpolation with alpha from 0 to 1
    alphas = np.linspace(0, 1, n_steps)
    interpolated_images = []

    with torch.no_grad():
        for alpha in alphas:
            z_interp = vA.slerp_torch(alpha, mu1, mu2)
            img_interp = model.decoder(z_interp)
            interpolated_images.append(img_interp.cpu().numpy()[0, 0])

    # Visualize the interpolation
    fig, axs = plt.subplots(1, n_steps, figsize=(16, 2))
    for i, (img, alpha) in enumerate(zip(interpolated_images, alphas)):
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'α={alpha:.1f}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()