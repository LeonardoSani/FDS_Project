import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def plot_distribution(data, title='Data Distribution', ax =None):
    type_counts = pd.Series(data).value_counts()
    df_plot = pd.DataFrame({'Category': type_counts.index.astype(str), 'Count': type_counts.values})

    # Use hue and legend=False to address FutureWarning
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